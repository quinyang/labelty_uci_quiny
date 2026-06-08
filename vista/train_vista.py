"""
DeepLabV3+ segmentation for dental X-ray multi-class segmentation.

K-FOLD CROSS-VALIDATION + ASPECT-RATIO-PRESERVING (LETTERBOX) RESIZE.
Ported to match train_unet.py's design (2026-05-26):
  * 5-fold CV trains on every image across folds; the K fold-models are
    ensembled at inference (nnU-Net's recipe). Val is held-out CLEAN originals.
  * LetterboxResized scales the longest edge to a square canvas and pads the
    remainder, preserving geometry (no stretching). Content sits at the top-left
    (origin); padding is on the bottom/right, so the inverse at inference is a
    plain crop of [:nA, :nB] then resize (see test_vista.py).
  * Leakage guard: an image and its augmentations never straddle the split.

Inherited fixes:
  1. 3-channel input (grayscale repeated x3) preserves ImageNet encoder weights.
  2. Decay class (11) has 0 pixels across all data — zeroed in the loss weight.
  3. WeightedRandomSampler oversamples images with rare classes.
  4. Tversky(0.3/0.7)+Focal loss; copy-paste of rare-class crops (4, 7).
  Note: no in-pipeline CLAHE — augmentation happens once, offline in Dataset102.
"""
import csv
import re
import cv2
import torch.nn.functional as F
import os
import glob
import numpy as np
import torch
import segmentation_models_pytorch as smp
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd,
    RepeatChanneld, MapTransform,
)
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from monai.data import Dataset, DataLoader, decollate_batch
from monai.losses import TverskyLoss, FocalLoss
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from sklearn.model_selection import KFold
from tqdm import tqdm

NUM_CLASSES = 12
CANVAS = 1024          # square letterbox canvas; longest image edge maps here (1024/32=32, valid for the encoder)
NUM_FOLDS = 5
BATCH_SIZE = 4         # raise if your GPU allows
MAX_EPOCHS = 300
WARMUP_EPOCHS = 5
PATIENCE = 25          # stop a fold if val mDice doesn't improve for this many epochs
CLASS_NAMES = [
    "BG", "Apical", "MainRoot", "MainCanal", "MesialRoot", "MesialCanal",
    "DistalRoot", "DistalCanal", "PalatalRoot", "PalatalCanal", "RCFilling", "Decay",
]

ORIG_DIR = "/home/jiakuny1/Projects/nnUNet_data/nnUNet_raw/Dataset101_Dental"
AUG_DIR  = "/home/jiakuny1/Projects/nnUNet_data/nnUNet_raw/Dataset102_Dental_Aug"
SAVE_ROOT = "/home/jiakuny1/Projects"

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"Training on device: {device}")


def _base_stem(img_path):
    """Strip _aug<N> suffix to recover the original image stem."""
    s = os.path.basename(img_path).replace("_0000.png", "")
    return re.sub(r"_aug\d+$", "", s)


class LetterboxResized(MapTransform):
    """
    Aspect-ratio-preserving resize to a square canvas.

    Scales the longest spatial edge to `canvas`, keeps aspect ratio, and pads the
    remainder. Content is placed at the ORIGIN (top-left); padding is appended on
    the bottom/right with `pad_value`. Placing content at the origin makes the
    inverse at inference a plain crop of [:nA, :nB] followed by a resize.

    Operates on channel-first tensors (C, A, B). Bilinear for images, nearest for
    integer label maps. Pad value 0 = background for labels, black for images.
    """

    def __init__(self, keys, canvas, modes, pad_value=0.0):
        super().__init__(keys)
        self.canvas = canvas
        self.modes = dict(zip(keys, modes))
        self.pad_value = pad_value

    @staticmethod
    def scaled_size(A, B, canvas):
        scale = canvas / max(A, B)
        nA = min(int(round(A * scale)), canvas)
        nB = min(int(round(B * scale)), canvas)
        return nA, nB

    def __call__(self, data):
        d = dict(data)
        for k in self.key_iterator(d):
            x = d[k]
            t = x if isinstance(x, torch.Tensor) else torch.as_tensor(np.asarray(x, dtype=np.float32))
            t = t.float()
            C, A, B = t.shape
            nA, nB = self.scaled_size(A, B, self.canvas)
            mode = self.modes[k]
            kw = {} if mode == "nearest" else {"align_corners": False}
            r = F.interpolate(t.unsqueeze(0), size=(nA, nB), mode=mode, **kw).squeeze(0)
            out = torch.full((C, self.canvas, self.canvas), self.pad_value, dtype=r.dtype)
            out[:, :nA, :nB] = r
            d[k] = out
        return d


RARE_CLASSES = [4, 7]  # MesialRoot, DistalCanal


class CopyPasteDistalCanald(MapTransform):
    """
    Paste rare-class crops (MesialRoot + DistalCanal) from donor images into
    training images. Collects one patch per rare class per donor image so both
    classes are represented in the patch library. Activates with prob per sample.
    """

    def __init__(self, image_key, label_key, all_image_paths, all_label_paths, prob=0.5):
        super().__init__([image_key, label_key])
        self.image_key = image_key
        self.label_key = label_key
        self.prob = prob
        self.patches = self._collect_patches(all_image_paths, all_label_paths)
        print(f"CopyPaste: {len(self.patches)} rare-class donor patches ready")

    def _collect_patches(self, img_paths, lbl_paths):
        patches = []
        for ip, lp in zip(img_paths, lbl_paths):
            lbl = cv2.imread(lp, cv2.IMREAD_GRAYSCALE)
            if lbl is None:
                continue
            present = set(np.unique(lbl).tolist())
            rare_present = [c for c in RARE_CLASSES if c in present]
            if not rare_present:
                continue
            img = cv2.imread(ip, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img_f = img.astype(np.float32) / 255.0
            for cls in rare_present:
                mask = lbl == cls
                ys, xs = np.where(mask)
                pad = 40
                y1 = max(0, int(ys.min()) - pad)
                y2 = min(lbl.shape[0], int(ys.max()) + pad + 1)
                x1 = max(0, int(xs.min()) - pad)
                x2 = min(lbl.shape[1], int(xs.max()) + pad + 1)
                patches.append({
                    "img":  img_f[y1:y2, x1:x2],
                    "lbl":  lbl[y1:y2, x1:x2],
                    "mask": mask[y1:y2, x1:x2],
                })
        return patches

    def __call__(self, data):
        d = dict(data)
        if not self.patches or np.random.random() > self.prob:
            return d

        img = d[self.image_key]
        lbl = d[self.label_key]

        is_tensor_img = isinstance(img, torch.Tensor)
        is_tensor_lbl = isinstance(lbl, torch.Tensor)
        img_np = img.detach().cpu().numpy().copy() if is_tensor_img else np.asarray(img, dtype=np.float32).copy()
        lbl_np = lbl.detach().cpu().numpy().copy() if is_tensor_lbl else np.asarray(lbl).copy()

        patch = self.patches[np.random.randint(len(self.patches))]
        ph, pw = patch["img"].shape
        _, H, W = img_np.shape

        if ph > H or pw > W:
            return d

        y_off = np.random.randint(0, H - ph + 1)
        x_off = np.random.randint(0, W - pw + 1)
        pmask = patch["mask"]

        img_np[0, y_off:y_off+ph, x_off:x_off+pw] = np.where(
            pmask, patch["img"], img_np[0, y_off:y_off+ph, x_off:x_off+pw]
        )
        lbl_np[0, y_off:y_off+ph, x_off:x_off+pw] = np.where(
            pmask, patch["lbl"], lbl_np[0, y_off:y_off+ph, x_off:x_off+pw]
        )

        d[self.image_key] = torch.from_numpy(img_np) if is_tensor_img else img_np
        d[self.label_key] = torch.from_numpy(lbl_np) if is_tensor_lbl else lbl_np
        return d


class TverskyFocalLoss(torch.nn.Module):
    """
    Tversky (alpha=0.3, beta=0.7): penalizes FN 2.3x more than FP.
    For thin canal structures, missing them (FN) is far worse than over-predicting.
    Focal (gamma=2.0) + class weights: mines hard examples and handles imbalance.
    """

    def __init__(self, focal_weights, alpha=0.3, beta=0.7, gamma=2.0):
        super().__init__()
        self.tversky = TverskyLoss(
            include_background=False,
            to_onehot_y=True,
            softmax=True,
            alpha=alpha,
            beta=beta,
        )
        self.focal = FocalLoss(
            include_background=False,
            to_onehot_y=True,
            gamma=gamma,
            weight=focal_weights,
            use_softmax=True,
        )

    def forward(self, pred, target):
        return 0.5 * self.tversky(pred, target) + 0.5 * self.focal(pred, target)


def compute_class_weights(label_paths, num_classes, device):
    counts = np.zeros(num_classes, dtype=np.float64)
    for lbl_path in tqdm(label_paths, desc="Computing class weights"):
        lbl = cv2.imread(lbl_path, cv2.IMREAD_GRAYSCALE)
        if lbl is None:
            continue
        for c in range(num_classes):
            counts[c] += int(np.sum(lbl == c))

    freq = counts / (counts.sum() + 1e-10)
    weights = 1.0 / np.sqrt(freq + 1e-10)
    weights = np.clip(weights, 0.0, 20.0)
    weights[11] = 0.0  # Decay: 0 pixels in all training images — exclude from loss
    weights = weights / (weights[weights > 0].mean() + 1e-10)
    return torch.tensor(weights, dtype=torch.float32).to(device)


def make_weighted_sampler(data_dicts, num_classes=12):
    """Oversample training images that contain rare anatomical classes."""
    class_image_counts = np.zeros(num_classes, dtype=np.float64)
    label_class_sets = []

    for d in tqdm(data_dicts, desc="Building weighted sampler"):
        lbl = cv2.imread(d["label"], cv2.IMREAD_GRAYSCALE)
        present = set(int(v) for v in np.unique(lbl)) if lbl is not None else {0}
        label_class_sets.append(present)
        for c in present:
            if c < num_classes:
                class_image_counts[c] += 1

    class_image_counts[11] = max(class_image_counts[11], len(data_dicts))

    sample_weights = []
    for present in label_class_sets:
        valid = [c for c in present if c < num_classes]
        w = max(len(data_dicts) / (class_image_counts[c] + 1e-6) for c in valid)
        sample_weights.append(w)

    return torch.utils.data.WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.float64),
        num_samples=len(sample_weights),
        replacement=True,
    )


def make_transforms(copy_paste):
    train_t = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityd(keys=["image"]),
        copy_paste,
        RepeatChanneld(keys=["image"], repeats=3),
        LetterboxResized(keys=["image", "label"], canvas=CANVAS,
                         modes=("bilinear", "nearest")),
    ])
    val_t = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityd(keys=["image"]),
        RepeatChanneld(keys=["image"], repeats=3),
        LetterboxResized(keys=["image", "label"], canvas=CANVAS,
                         modes=("bilinear", "nearest")),
    ])
    return train_t, val_t


def build_model():
    return smp.DeepLabV3Plus(
        encoder_name="resnet50",
        encoder_weights="imagenet",
        in_channels=3,
        classes=NUM_CLASSES,
        decoder_dropout=0.2,
    ).to(device)


def train_one_fold(fold, train_dicts, val_dicts):
    print(f"\n{'#'*70}\n# FOLD {fold}/{NUM_FOLDS - 1}  "
          f"train={len(train_dicts)} (orig+aug)  val={len(val_dicts)} (orig)\n{'#'*70}")

    class_weights = compute_class_weights([d["label"] for d in train_dicts], NUM_CLASSES, device)
    copy_paste = CopyPasteDistalCanald(
        image_key="image", label_key="label",
        all_image_paths=[d["image"] for d in train_dicts],
        all_label_paths=[d["label"] for d in train_dicts],
        prob=0.5,
    )
    train_t, val_t = make_transforms(copy_paste)

    train_ds = Dataset(data=train_dicts, transform=train_t)
    val_ds = Dataset(data=val_dicts, transform=val_t)
    sampler = make_weighted_sampler(train_dicts)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=4, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = build_model()

    loss_fn = TverskyFocalLoss(focal_weights=class_weights[1:], alpha=0.3, beta=0.7, gamma=2.0)
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=WARMUP_EPOCHS)
    cosine = CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS - WARMUP_EPOCHS, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[WARMUP_EPOCHS])
    scaler = GradScaler()

    post_pred = AsDiscrete(argmax=True, to_onehot=NUM_CLASSES)
    post_label = AsDiscrete(to_onehot=NUM_CLASSES)
    dice_metric = DiceMetric(include_background=False, reduction="mean_batch")

    best_path = os.path.join(SAVE_ROOT, f"vista2d_fold{fold}.pth")
    log_path = os.path.join(SAVE_ROOT, f"vista_fold{fold}_val_log.csv")
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss"] + CLASS_NAMES[1:] + ["mDice_excl_decay"])

    best_val_dice, no_improve = 0.0, 0
    for epoch in range(MAX_EPOCHS):
        model.train()
        epoch_loss, step = 0.0, 0
        pbar = tqdm(train_loader, desc=f"Fold {fold} Epoch {epoch+1}/{MAX_EPOCHS}", dynamic_ncols=True)
        for batch in pbar:
            step += 1
            inputs = batch["image"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            with autocast():
                outputs = model(inputs)
                if outputs.shape[2:] != labels.shape[2:]:
                    outputs = F.interpolate(outputs, size=labels.shape[2:], mode="bilinear", align_corners=False)
                loss = loss_fn(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        epoch_loss /= max(step, 1)
        scheduler.step()
        print(f"Fold {fold} Epoch {epoch+1} | train_loss={epoch_loss:.4f} | lr={optimizer.param_groups[0]['lr']:.6f}")

        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                for vb in val_loader:
                    vi = vb["image"].to(device)
                    vl = vb["label"].to(device)
                    with autocast():
                        vo = model(vi)
                        if vo.shape[2:] != vl.shape[2:]:
                            vo = F.interpolate(vo, size=vl.shape[2:], mode="bilinear", align_corners=False)
                    op = [post_pred(i) for i in decollate_batch(vo)]
                    lp = [post_label(i) for i in decollate_batch(vl)]
                    dice_metric(y_pred=op, y=lp)
            per_class = dice_metric.aggregate().cpu().numpy()
            dice_metric.reset()
            mDice = float(per_class[:10].mean())
            print(f"  >> Fold {fold} val mDice (excl BG+Decay): {mDice:.4f}")
            with open(log_path, "a", newline="") as f:
                csv.writer(f).writerow([epoch + 1, round(epoch_loss, 6)] + per_class.tolist() + [round(mDice, 6)])

            if mDice > best_val_dice:
                best_val_dice = mDice
                no_improve = 0
                torch.save(model.state_dict(), best_path)
                print(f"  >> New best for fold {fold} (mDice={best_val_dice:.4f}) -> {best_path}")
            else:
                no_improve += 5
                if no_improve >= PATIENCE:
                    print(f"  >> Early stopping fold {fold} at epoch {epoch+1}")
                    break

    del model
    torch.cuda.empty_cache()
    print(f"Fold {fold} done. Best val mDice: {best_val_dice:.4f}")
    return best_val_dice


def main():
    orig_images = sorted(glob.glob(os.path.join(ORIG_DIR, "imagesTr", "*_0000.png")))
    orig_dicts = [
        {"image": p, "label": p.replace("imagesTr", "labelsTr").replace("_0000.png", ".png")}
        for p in orig_images
    ]
    orig_stems = [_base_stem(d["image"]) for d in orig_dicts]
    all_aug = sorted(glob.glob(os.path.join(AUG_DIR, "imagesTr", "*_0000.png")))
    print(f"Originals: {len(orig_dicts)}   Augmented pool: {len(all_aug)}")

    kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
    fold_scores = []
    for fold, (tr_idx, va_idx) in enumerate(kf.split(orig_dicts)):
        # Resume: if this fold's checkpoint already exists, skip retraining and
        # recover its best mDice from the val log for the CV summary.
        ckpt = os.path.join(SAVE_ROOT, f"vista2d_fold{fold}.pth")
        if os.path.exists(ckpt):
            log_path = os.path.join(SAVE_ROOT, f"vista_fold{fold}_val_log.csv")
            best = 0.0
            if os.path.exists(log_path):
                with open(log_path) as f:
                    best = max((float(r["mDice_excl_decay"]) for r in csv.DictReader(f)), default=0.0)
            print(f"[resume] fold {fold} checkpoint exists -> skip (best mDice={best:.4f})")
            fold_scores.append(best)
            continue
        # Val = held-out originals; train = augmented copies of the OTHER folds'
        # stems only. The held-out fold's stems are excluded from training to
        # prevent leakage (an image and its augmentations never straddle the split).
        val_dicts = [orig_dicts[i] for i in va_idx]
        train_stems = {orig_stems[i] for i in tr_idx}
        train_dicts = [
            {"image": p, "label": p.replace("imagesTr", "labelsTr").replace("_0000.png", ".png")}
            for p in all_aug if _base_stem(p) in train_stems
        ]
        fold_scores.append(train_one_fold(fold, train_dicts, val_dicts))

    print(f"\n{'='*70}\nCV complete. Per-fold best val mDice: "
          f"{[round(s, 4) for s in fold_scores]}")
    print(f"Mean CV mDice = {np.mean(fold_scores):.4f} +/- {np.std(fold_scores):.4f}")
    print("Fold checkpoints: vista2d_fold0.pth .. vista2d_fold{}.pth".format(NUM_FOLDS - 1))


if __name__ == "__main__":
    main()
