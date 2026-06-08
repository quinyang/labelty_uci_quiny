"""
LAYER 1 of the two-layer dental cascade — root-STRUCTURE segmentation.

6-class target (see make_layer1_labels.py): BG, Main, Mesial, Distal, Palatal,
Apical. Each tooth structure is root UNION canal (+ absorbed RC filling), so the
mesial/distal/palatal/main disambiguation — the hard, *positional* problem — is
learned here, once. Layer 2 later finds the canal *inside* each region and
inherits the region's identity.

This is a deliberate clone of p_unet/train_unet.py (same EfficientNet-B2 U-Net,
letterbox @1024, 5-fold CV, Tversky+Focal, weighted sampler, copy-paste). Diffs
from train_unet.py, all because the label space changed 12 -> 6:
  * NUM_CLASSES 12 -> 6; CLASS_NAMES are the 6 structures; no Decay class.
  * Labels read from labelsTr_layer1/ (remapped) instead of labelsTr/.
  * compute_class_weights drops the hard-coded weights[11]=0 (no decay column).
  * RARE_CLASSES = [3] (Distal) — DistalCanal was in only 19 imgs, so the merged
    Distal region is the small/rare one; copy-paste oversamples it.
  * Checkpoints/logs: layer1_fold{f}.pth / layer1_fold{f}_val_log.csv.
  * mDice = mean over all 5 foreground structures (no decay to exclude).
"""
import argparse
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
from monai.networks.utils import one_hot
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from monai.data import Dataset, DataLoader, decollate_batch
from monai.losses import FocalLoss
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from sklearn.model_selection import KFold
from tqdm import tqdm

NUM_CLASSES = 6
CANVAS = 1024
NUM_FOLDS = 5
BATCH_SIZE = 4
MAX_EPOCHS = 300
WARMUP_EPOCHS = 5
PATIENCE = 25
CLASS_NAMES = ["BG", "Main", "Mesial", "Distal", "Palatal", "Apical"]

ORIG_DIR = "/home/jiakuny1/Projects/nnUNet_data/nnUNet_raw/Dataset101_Dental"
AUG_DIR  = "/home/jiakuny1/Projects/nnUNet_data/nnUNet_raw/Dataset102_Dental_Aug"
SAVE_ROOT = "/home/jiakuny1/Projects"
LABEL_SUBDIR = "labelsTr_layer1"   # remapped 6-class labels

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"Training on device: {device}")


def _label_path(image_path):
    """imagesTr/<stem>_0000.png -> <its dataset>/labelsTr_layer1/<stem>.png"""
    return (image_path
            .replace("imagesTr", LABEL_SUBDIR)
            .replace("_0000.png", ".png"))


def _base_stem(img_path):
    s = os.path.basename(img_path).replace("_0000.png", "")
    return re.sub(r"_aug\d+$", "", s)


class LetterboxResized(MapTransform):
    """Aspect-ratio-preserving resize to a square canvas (content at top-left)."""

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


RARE_CLASSES = [3]  # Distal structure (DistalCanal was in only 19/436 images)
RARE_BOOST = 3.0    # extra sampler weight for images containing a rare class
COPY_PASTE_PROB = 0.8   # was 0.5 — Distal is the recall bottleneck, paste harder
MAX_PASTE = 2           # paste up to this many rare patches per triggered image


class CopyPasteStructd(MapTransform):
    """Paste rare-structure crops from donor images into training images."""

    def __init__(self, image_key, label_key, all_image_paths, all_label_paths,
                 prob=COPY_PASTE_PROB, max_paste=MAX_PASTE):
        super().__init__([image_key, label_key])
        self.image_key = image_key
        self.label_key = label_key
        self.prob = prob
        self.max_paste = max_paste
        self.patches = self._collect_patches(all_image_paths, all_label_paths)
        print(f"CopyPaste: {len(self.patches)} rare-structure donor patches ready "
              f"(prob={prob}, up to {max_paste} pastes/img)")

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
        _, H, W = img_np.shape

        for _ in range(np.random.randint(1, self.max_paste + 1)):
            patch = self.patches[np.random.randint(len(self.patches))]
            ph, pw = patch["img"].shape
            if ph > H or pw > W:
                continue
            y_off = np.random.randint(0, H - ph + 1)
            x_off = np.random.randint(0, W - pw + 1)
            pmask = patch["mask"]
            img_np[0, y_off:y_off+ph, x_off:x_off+pw] = np.where(
                pmask, patch["img"], img_np[0, y_off:y_off+ph, x_off:x_off+pw])
            lbl_np[0, y_off:y_off+ph, x_off:x_off+pw] = np.where(
                pmask, patch["lbl"], lbl_np[0, y_off:y_off+ph, x_off:x_off+pw])

        d[self.image_key] = torch.from_numpy(img_np) if is_tensor_img else img_np
        d[self.label_key] = torch.from_numpy(lbl_np) if is_tensor_lbl else lbl_np
        return d


class FocalTverskyClDiceLoss(torch.nn.Module):
    """Region + topology loss tuned for the rare, thin Distal structure.

    total = (1 - lambda_cl) * (w_ft * FocalTversky + w_focal * Focal)
            + lambda_cl * clDice

    - FocalTversky (alpha=0.3, beta=0.7, gamma_ft=4/3): Tversky penalises false
      negatives 0.7:0.3 over false positives (Distal is recall-limited), and the
      gamma_ft exponent focuses gradient on low-overlap classes — i.e. Distal —
      instead of the already-easy Apical/Main. Replaces the plain Tversky term.
    - Focal (gamma=2) + inverse-freq class weights: pixel-level imbalance, kept
      from the previous loss.
    - clDice (centerline Dice): rewards skeleton overlap so thin Distal/canal
      regions keep connectivity instead of being eroded to background. Computed
      at half-res (512) in fp32 for stability/memory; background excluded.

    Replaces TverskyFocalLoss (plain Tversky+Focal, 0.5/0.5).
    """

    def __init__(self, focal_weights, num_classes,
                 alpha=0.3, beta=0.7, gamma_ft=4.0 / 3.0, gamma_focal=2.0,
                 w_ft=0.6, w_focal=0.4, lambda_cl=0.25, cl_iters=8, smooth=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.alpha, self.beta, self.gamma_ft = alpha, beta, gamma_ft
        self.w_ft, self.w_focal, self.lambda_cl = w_ft, w_focal, lambda_cl
        self.cl_iters, self.smooth = cl_iters, smooth
        self.focal = FocalLoss(include_background=False, to_onehot_y=True,
                               gamma=gamma_focal, weight=focal_weights, use_softmax=True)

    @staticmethod
    def _soft_erode(x):
        p1 = -F.max_pool2d(-x, (3, 1), 1, (1, 0))
        p2 = -F.max_pool2d(-x, (1, 3), 1, (0, 1))
        return torch.min(p1, p2)

    @staticmethod
    def _soft_dilate(x):
        return F.max_pool2d(x, (3, 3), 1, 1)

    def _soft_open(self, x):
        return self._soft_dilate(self._soft_erode(x))

    def _soft_skel(self, x, iters):
        x1 = self._soft_open(x)
        skel = F.relu(x - x1)
        for _ in range(iters):
            x = self._soft_erode(x)
            x1 = self._soft_open(x)
            delta = F.relu(x - x1)
            skel = skel + F.relu(delta - skel * delta)
        return skel

    def _cldice(self, probs_fg, tgt_fg):
        s = self.smooth
        skel_pred = self._soft_skel(probs_fg, self.cl_iters)
        skel_true = self._soft_skel(tgt_fg, self.cl_iters)
        tprec = (torch.sum(skel_pred * tgt_fg, dim=(2, 3)) + s) / (torch.sum(skel_pred, dim=(2, 3)) + s)
        tsens = (torch.sum(skel_true * probs_fg, dim=(2, 3)) + s) / (torch.sum(skel_true, dim=(2, 3)) + s)
        cl = 2.0 * tprec * tsens / (tprec + tsens + 1e-8)
        return 1.0 - cl.mean()

    def _focal_tversky(self, probs_fg, tgt_fg):
        dims = (2, 3)
        tp = torch.sum(probs_fg * tgt_fg, dim=dims)
        fp = torch.sum(probs_fg * (1 - tgt_fg), dim=dims)
        fn = torch.sum((1 - probs_fg) * tgt_fg, dim=dims)
        ti = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return torch.pow(1.0 - ti, self.gamma_ft).mean()

    def forward(self, logits, target):
        focal = self.focal(logits, target)
        # region/topology terms in fp32 (soft-skeleton is unstable in fp16)
        with torch.cuda.amp.autocast(enabled=False):
            probs = torch.softmax(logits.float(), dim=1)[:, 1:]          # foreground
            tgt = one_hot(target.float(), self.num_classes)[:, 1:]
            ftl = self._focal_tversky(probs, tgt)
            probs_lo = F.interpolate(probs, scale_factor=0.5, mode="bilinear", align_corners=False)
            tgt_lo = F.interpolate(tgt, scale_factor=0.5, mode="bilinear", align_corners=False)
            cldice = self._cldice(probs_lo, tgt_lo)
        region = self.w_ft * ftl + self.w_focal * focal
        return (1.0 - self.lambda_cl) * region + self.lambda_cl * cldice


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
    weights = weights / (weights[weights > 0].mean() + 1e-10)
    return torch.tensor(weights, dtype=torch.float32).to(device)


def make_weighted_sampler(data_dicts, num_classes=NUM_CLASSES):
    """Oversample training images that contain rare structures."""
    class_image_counts = np.zeros(num_classes, dtype=np.float64)
    label_class_sets = []
    for d in tqdm(data_dicts, desc="Building weighted sampler"):
        lbl = cv2.imread(d["label"], cv2.IMREAD_GRAYSCALE)
        present = set(int(v) for v in np.unique(lbl)) if lbl is not None else {0}
        label_class_sets.append(present)
        for c in present:
            if c < num_classes:
                class_image_counts[c] += 1
    class_image_counts = np.maximum(class_image_counts, 1.0)

    sample_weights = []
    for present in label_class_sets:
        valid = [c for c in present if c < num_classes] or [0]
        w = max((len(data_dicts) / (class_image_counts[c] + 1e-6))
                * (RARE_BOOST if c in RARE_CLASSES else 1.0)
                for c in valid)
        sample_weights.append(w)
    return torch.utils.data.WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.float64),
        num_samples=len(sample_weights), replacement=True)


def make_transforms(copy_paste):
    train_t = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityd(keys=["image"]),
        copy_paste,
        RepeatChanneld(keys=["image"], repeats=3),
        LetterboxResized(keys=["image", "label"], canvas=CANVAS, modes=("bilinear", "nearest")),
    ])
    val_t = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityd(keys=["image"]),
        RepeatChanneld(keys=["image"], repeats=3),
        LetterboxResized(keys=["image", "label"], canvas=CANVAS, modes=("bilinear", "nearest")),
    ])
    return train_t, val_t


def train_one_fold(fold, train_dicts, val_dicts):
    print(f"\n{'#'*70}\n# LAYER1 FOLD {fold}/{NUM_FOLDS - 1}  "
          f"train={len(train_dicts)} (orig+aug)  val={len(val_dicts)} (orig)\n{'#'*70}")

    class_weights = compute_class_weights([d["label"] for d in train_dicts], NUM_CLASSES, device)
    copy_paste = CopyPasteStructd(
        image_key="image", label_key="label",
        all_image_paths=[d["image"] for d in train_dicts],
        all_label_paths=[d["label"] for d in train_dicts],
    )
    train_t, val_t = make_transforms(copy_paste)

    train_ds = Dataset(data=train_dicts, transform=train_t)
    val_ds = Dataset(data=val_dicts, transform=val_t)
    sampler = make_weighted_sampler(train_dicts)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = smp.Unet(
        encoder_name="efficientnet-b2",
        encoder_weights="imagenet",
        in_channels=3,
        classes=NUM_CLASSES,
        decoder_dropout=0.2,
    ).to(device)

    loss_fn = FocalTverskyClDiceLoss(focal_weights=class_weights[1:], num_classes=NUM_CLASSES)
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=WARMUP_EPOCHS)
    cosine = CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS - WARMUP_EPOCHS, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[WARMUP_EPOCHS])
    scaler = GradScaler()

    post_pred = AsDiscrete(argmax=True, to_onehot=NUM_CLASSES)
    post_label = AsDiscrete(to_onehot=NUM_CLASSES)
    dice_metric = DiceMetric(include_background=False, reduction="mean_batch")

    best_path = os.path.join(SAVE_ROOT, f"layer1_fold{fold}.pth")
    log_path = os.path.join(SAVE_ROOT, f"layer1_fold{fold}_val_log.csv")
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss"] + CLASS_NAMES[1:] + ["mDice"])

    best_val_dice, no_improve = 0.0, 0
    for epoch in range(MAX_EPOCHS):
        model.train()
        epoch_loss, step = 0.0, 0
        pbar = tqdm(train_loader, desc=f"L1 Fold {fold} Epoch {epoch+1}/{MAX_EPOCHS}", dynamic_ncols=True)
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
        print(f"L1 Fold {fold} Epoch {epoch+1} | train_loss={epoch_loss:.4f} | lr={optimizer.param_groups[0]['lr']:.6f}")

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
            per_class = dice_metric.aggregate().cpu().numpy()   # length 5 (Main..Apical)
            dice_metric.reset()
            mDice = float(per_class.mean())
            print(f"  >> L1 Fold {fold} val mDice (5 structures): {mDice:.4f}  "
                  + "  ".join(f"{n}={v:.3f}" for n, v in zip(CLASS_NAMES[1:], per_class)))
            with open(log_path, "a", newline="") as f:
                csv.writer(f).writerow([epoch + 1, round(epoch_loss, 6)] + per_class.tolist() + [round(mDice, 6)])

            if mDice > best_val_dice:
                best_val_dice = mDice
                no_improve = 0
                torch.save(model.state_dict(), best_path)
                print(f"  >> New best for L1 fold {fold} (mDice={best_val_dice:.4f}) -> {best_path}")
            else:
                no_improve += 5
                if no_improve >= PATIENCE:
                    print(f"  >> Early stopping L1 fold {fold} at epoch {epoch+1}")
                    break

    del model
    torch.cuda.empty_cache()
    print(f"L1 Fold {fold} done. Best val mDice: {best_val_dice:.4f}")
    return best_val_dice


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="+", default=list(range(NUM_FOLDS)),
                    help="which fold indices to train (default: all 5)")
    args = ap.parse_args()

    orig_images = sorted(glob.glob(os.path.join(ORIG_DIR, "imagesTr", "*_0000.png")))
    orig_dicts = [{"image": p, "label": _label_path(p)} for p in orig_images]
    orig_stems = [_base_stem(d["image"]) for d in orig_dicts]
    all_aug = sorted(glob.glob(os.path.join(AUG_DIR, "imagesTr", "*_0000.png")))
    print(f"Originals: {len(orig_dicts)}   Augmented pool: {len(all_aug)}")

    # fail fast if the remapped labels are missing
    if orig_dicts and not os.path.exists(orig_dicts[0]["label"]):
        raise FileNotFoundError(
            f"Remapped label not found: {orig_dicts[0]['label']}\n"
            f"Run  python make_layer1_labels.py  first.")

    kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
    requested = set(args.folds)
    fold_scores = {}
    for fold, (tr_idx, va_idx) in enumerate(kf.split(orig_dicts)):
        if fold not in requested:
            continue
        val_dicts = [orig_dicts[i] for i in va_idx]
        train_stems = {orig_stems[i] for i in tr_idx}
        train_dicts = [{"image": p, "label": _label_path(p)}
                       for p in all_aug if _base_stem(p) in train_stems]
        fold_scores[fold] = train_one_fold(fold, train_dicts, val_dicts)

    print(f"\n{'='*70}\nLAYER1 CV complete. Per-fold best val mDice: "
          f"{ {f: round(s, 4) for f, s in fold_scores.items()} }")
    if fold_scores:
        print(f"Mean CV mDice = {np.mean(list(fold_scores.values())):.4f} "
              f"+/- {np.std(list(fold_scores.values())):.4f}")


if __name__ == "__main__":
    main()
