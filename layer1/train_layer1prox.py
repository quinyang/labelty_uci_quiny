"""
LAYER 1 PROX — root-STRUCTURE segmentation with Mesial+Distal MERGED.

5-class target (see make_layer1prox_labels.py): BG, Main, Proximal, Palatal,
Apical. Mesial and Distal are merged into one "Proximal" class because the
mesial/distal distinction is under-determined from an isolated tooth crop
(laterality measured 51/49 wrt image position; no quadrant context in the data).
The split is deferred to Layer 2 (concavity + canal count inside the region).

Clone of train_layer1.py with the PROVEN baseline loss (Tversky+Focal, NO clDice —
clDice regressed Apical in the 6-class smoke test) and no rare-class oversampling
(Proximal is present in ~84% of images; nothing is rare anymore). Diffs vs the
6-class script:
  * NUM_CLASSES 6 -> 5; CLASS_NAMES = BG/Main/Proximal/Palatal/Apical.
  * Labels read from labelsTr_layer1prox/.
  * RARE_CLASSES = [] (copy-paste inert); sampler keeps mild inverse-freq weighting.
  * Checkpoints/logs: layer1prox_fold{f}.pth / layer1prox_fold{f}_val_log.csv.
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
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from monai.data import Dataset, DataLoader, decollate_batch
from monai.losses import TverskyLoss, FocalLoss
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from sklearn.model_selection import KFold
from tqdm import tqdm

NUM_CLASSES = 5
CANVAS = 1024
NUM_FOLDS = 5
BATCH_SIZE = 4
MAX_EPOCHS = 300
WARMUP_EPOCHS = 5
PATIENCE = 25
CLASS_NAMES = ["BG", "Main", "Proximal", "Palatal", "Apical"]

ORIG_DIR = "/home/jiakuny1/Projects/nnUNet_data/nnUNet_raw/Dataset101_Dental"
AUG_DIR  = "/home/jiakuny1/Projects/nnUNet_data/nnUNet_raw/Dataset102_Dental_Aug"
SAVE_ROOT = "/home/jiakuny1/Projects"
LABEL_SUBDIR = "labelsTr_layer1prox"   # remapped 5-class labels

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"Training on device: {device}")


def _label_path(image_path):
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


class TverskyFocalLoss(torch.nn.Module):
    """Tversky(0.3/0.7) penalises FN > FP; Focal(2.0)+class weights for imbalance.

    The proven Layer-1 baseline loss (the 0.664 6-class model used this). clDice
    was tried in the 6-class variant and REGRESSED Apical — not used here.
    """

    def __init__(self, focal_weights, alpha=0.3, beta=0.7, gamma=2.0):
        super().__init__()
        self.tversky = TverskyLoss(include_background=False, to_onehot_y=True,
                                   softmax=True, alpha=alpha, beta=beta)
        self.focal = FocalLoss(include_background=False, to_onehot_y=True,
                               gamma=gamma, weight=focal_weights, use_softmax=True)

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
    weights = weights / (weights[weights > 0].mean() + 1e-10)
    return torch.tensor(weights, dtype=torch.float32).to(device)


def make_weighted_sampler(data_dicts, num_classes=NUM_CLASSES):
    """Mild oversampling of images containing the less-frequent structures."""
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
        w = max(len(data_dicts) / (class_image_counts[c] + 1e-6) for c in valid)
        sample_weights.append(w)
    return torch.utils.data.WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.float64),
        num_samples=len(sample_weights), replacement=True)


def make_transforms():
    train_t = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityd(keys=["image"]),
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


def train_one_fold(fold, train_dicts, val_dicts, encoder="efficientnet-b2", tag="",
                   tversky_beta=0.7, palatal_boost=1.0):
    print(f"\n{'#'*70}\n# LAYER1PROX{tag.upper()} FOLD {fold}/{NUM_FOLDS - 1}  enc={encoder}  "
          f"beta={tversky_beta} palatal_boost={palatal_boost}  "
          f"train={len(train_dicts)} (orig+aug)  val={len(val_dicts)} (orig)\n{'#'*70}")

    class_weights = compute_class_weights([d["label"] for d in train_dicts], NUM_CLASSES, device)
    # Palatal is class 3 -> index 2 in the foreground-only focal weights [Main,Proximal,Palatal,Apical]
    focal_weights = class_weights[1:].clone()
    focal_weights[2] *= palatal_boost
    train_t, val_t = make_transforms()

    train_ds = Dataset(data=train_dicts, transform=train_t)
    val_ds = Dataset(data=val_dicts, transform=val_t)
    sampler = make_weighted_sampler(train_dicts)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = smp.Unet(
        encoder_name=encoder,
        encoder_weights="imagenet",
        in_channels=3,
        classes=NUM_CLASSES,
        decoder_dropout=0.2,
    ).to(device)

    loss_fn = TverskyFocalLoss(focal_weights=focal_weights, alpha=0.3, beta=tversky_beta, gamma=2.0)
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=WARMUP_EPOCHS)
    cosine = CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS - WARMUP_EPOCHS, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[WARMUP_EPOCHS])
    scaler = GradScaler()

    post_pred = AsDiscrete(argmax=True, to_onehot=NUM_CLASSES)
    post_label = AsDiscrete(to_onehot=NUM_CLASSES)
    dice_metric = DiceMetric(include_background=False, reduction="mean_batch")

    best_path = os.path.join(SAVE_ROOT, f"layer1prox{tag}_fold{fold}.pth")
    log_path = os.path.join(SAVE_ROOT, f"layer1prox{tag}_fold{fold}_val_log.csv")
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss"] + CLASS_NAMES[1:] + ["mDice"])

    best_val_dice, no_improve = 0.0, 0
    for epoch in range(MAX_EPOCHS):
        model.train()
        epoch_loss, step = 0.0, 0
        pbar = tqdm(train_loader, desc=f"L1P Fold {fold} Epoch {epoch+1}/{MAX_EPOCHS}", dynamic_ncols=True)
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
        print(f"L1P Fold {fold} Epoch {epoch+1} | train_loss={epoch_loss:.4f} | lr={optimizer.param_groups[0]['lr']:.6f}")

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
            per_class = dice_metric.aggregate().cpu().numpy()   # length 4 (Main..Apical)
            dice_metric.reset()
            mDice = float(per_class.mean())
            print(f"  >> L1P Fold {fold} val mDice (4 structures): {mDice:.4f}  "
                  + "  ".join(f"{n}={v:.3f}" for n, v in zip(CLASS_NAMES[1:], per_class)))
            with open(log_path, "a", newline="") as f:
                csv.writer(f).writerow([epoch + 1, round(epoch_loss, 6)] + per_class.tolist() + [round(mDice, 6)])

            if mDice > best_val_dice:
                best_val_dice = mDice
                no_improve = 0
                torch.save(model.state_dict(), best_path)
                print(f"  >> New best for L1P fold {fold} (mDice={best_val_dice:.4f}) -> {best_path}")
            else:
                no_improve += 5
                if no_improve >= PATIENCE:
                    print(f"  >> Early stopping L1P fold {fold} at epoch {epoch+1}")
                    break

    del model
    torch.cuda.empty_cache()
    print(f"L1P Fold {fold} done. Best val mDice: {best_val_dice:.4f}")
    return best_val_dice


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="+", default=list(range(NUM_FOLDS)),
                    help="which fold indices to train (default: all 5)")
    ap.add_argument("--encoder", default="efficientnet-b2",
                    help="smp encoder_name (e.g. efficientnet-b4). Checkpoints are tagged "
                         "by encoder so the b2 baseline is never overwritten.")
    ap.add_argument("--tversky-beta", type=float, default=0.7,
                    help="Tversky FN penalty (0.7 baseline). Raise to push recall.")
    ap.add_argument("--palatal-boost", type=float, default=1.0,
                    help="multiply Palatal's focal class weight (1.0 = baseline). >1 lifts Palatal recall.")
    ap.add_argument("--run-tag", default="",
                    help="extra checkpoint/log tag for an experiment, e.g. 'palrec'")
    args = ap.parse_args()
    # b2 keeps the untagged baseline names; encoder + run-tag both extend the tag
    tag = "" if args.encoder == "efficientnet-b2" else "_" + args.encoder.split("-")[-1]
    if args.run_tag:
        tag += "_" + args.run_tag

    orig_images = sorted(glob.glob(os.path.join(ORIG_DIR, "imagesTr", "*_0000.png")))
    orig_dicts = [{"image": p, "label": _label_path(p)} for p in orig_images]
    orig_stems = [_base_stem(d["image"]) for d in orig_dicts]
    all_aug = sorted(glob.glob(os.path.join(AUG_DIR, "imagesTr", "*_0000.png")))
    print(f"Originals: {len(orig_dicts)}   Augmented pool: {len(all_aug)}")

    if orig_dicts and not os.path.exists(orig_dicts[0]["label"]):
        raise FileNotFoundError(
            f"Remapped label not found: {orig_dicts[0]['label']}\n"
            f"Run  python make_layer1prox_labels.py  first.")

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
        fold_scores[fold] = train_one_fold(fold, train_dicts, val_dicts, args.encoder, tag,
                                           tversky_beta=args.tversky_beta,
                                           palatal_boost=args.palatal_boost)

    print(f"\n{'='*70}\nLAYER1PROX CV complete. Per-fold best val mDice: "
          f"{ {f: round(s, 4) for f, s in fold_scores.items()} }")
    if fold_scores:
        print(f"Mean CV mDice = {np.mean(list(fold_scores.values())):.4f} "
              f"+/- {np.std(list(fold_scores.values())):.4f}")


if __name__ == "__main__":
    main()
