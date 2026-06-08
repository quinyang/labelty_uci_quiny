"""
LAYER 1 scorecard — per-structure Dice/IoU/Prec/Recall against the remapped
6-class GT (resource/labels_mask_layer1/).

The decisive question for the cascade: can a dedicated structure model segment
the 4 root structures cleanly — well above the raw per-root numbers — AND beat
the existing 12-class model collapsed to the same 6 structures? The --remap-12
flag answers the second half: it remaps a 12-class prediction dir (e.g. nnU-Net)
into the 6 Layer-1 classes on the fly, so the comparison is apples-to-apples.

Usage:
    python evaluate_layer1.py                                  # eval layer1_predictions/
    python evaluate_layer1.py --pred-dir <dir> --remap-12      # baseline: collapse a 12-class model
"""
import argparse
import os
import glob
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

NUM_CLASSES = 6
CLASS_NAMES = ["Background", "Main", "Mesial", "Distal", "Palatal", "Apical"]
GT_DIR = "/home/jiakuny1/Projects/resource/labels_mask_layer1"
OUT_DIR = "/home/jiakuny1/Projects"

# same 12->6 mapping as make_layer1_labels.py, MINUS the spatial filling step.
# For a *prediction*, filling pixels (10) are simply dropped to BG: the structure
# Dice is unaffected since GT filling was absorbed into a structure region, and a
# model rarely predicts large filling blobs outside a root anyway.
REMAP_12 = {0: 0, 11: 0, 10: 0, 1: 5, 2: 1, 3: 1, 4: 2, 5: 2, 6: 3, 7: 3, 8: 4, 9: 4}

# raw per-root Dice from the current best 12-class models, for context only.
# (these are NOT directly comparable to merged-structure Dice — structures are an
# easier target — but they show the floor the cascade is trying to lift.)
RAW_ROOT_REFERENCE = {
    "Main":    ("MainRoot 0.676 / MainCanal 0.738", ),
    "Mesial":  ("MesialRoot 0.40 / MesialCanal 0.75", ),
    "Distal":  ("DistalRoot 0.36 / DistalCanal 0.12  <- the cascade's main target", ),
    "Palatal": ("PalatalRoot 0.74 / PalatalCanal 0.61", ),
    "Apical":  ("ApicalLesion 0.82", ),
}


def remap_pred(pred):
    out = np.zeros_like(pred)
    for src, dst in REMAP_12.items():
        out[pred == src] = dst
    return out


def load_pairs(pred_dir, remap_12):
    pairs = []
    for pred_path in sorted(glob.glob(os.path.join(pred_dir, "*.png"))):
        fname = os.path.basename(pred_path)
        gt_path = os.path.join(GT_DIR, fname)
        if not os.path.exists(gt_path):
            continue
        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        if pred is None or gt is None:
            continue
        if remap_12:
            pred = remap_pred(pred)
        if pred.shape != gt.shape:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
        pairs.append((gt.flatten(), pred.flatten()))
    return pairs


def compute(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    dice, iou, prec, rec = [], [], [], []
    for i in range(NUM_CLASSES):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        dice.append(2 * tp / (2 * tp + fp + fn + 1e-8))
        iou.append(tp / (tp + fp + fn + 1e-8))
        prec.append(tp / (tp + fp + 1e-8))
        rec.append(tp / (tp + fn + 1e-8))
    return list(map(np.array, (dice, iou, prec, rec))) + [cm]


def save_confusion_matrix(cm, png_path, csv_path, title):
    """Row-normalised confusion matrix (recall view): cm[i,j] = P(pred=j | gt=i)."""
    cm_norm = cm.astype(np.float64) / (cm.sum(axis=1, keepdims=True) + 1e-12)

    pd.DataFrame(cm, index=[f"gt_{n}" for n in CLASS_NAMES],
                 columns=[f"pred_{n}" for n in CLASS_NAMES]).to_csv(csv_path)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(NUM_CLASSES)); ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Ground truth")
    ax.set_title(title)
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            v = cm_norm[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="white" if v > 0.5 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="row-normalised")
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"Saved confusion matrix -> {png_path}\n              (counts) -> {csv_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", default="/home/jiakuny1/Projects/layer1_predictions")
    ap.add_argument("--remap-12", action="store_true",
                    help="treat predictions as 12-class and collapse them to 6 structures")
    ap.add_argument("--csv-out", default=os.path.join(OUT_DIR, "metrics_layer1.csv"))
    args = ap.parse_args()

    pairs = load_pairs(args.pred_dir, args.remap_12)
    if not pairs:
        print(f"No matching predictions/GT for {args.pred_dir} (GT: {GT_DIR}). "
              f"Did you run make_layer1_labels.py and test_layer1.py?")
        return

    y_true = np.concatenate([p[0] for p in pairs])
    y_pred = np.concatenate([p[1] for p in pairs])
    dice, iou, prec, rec, cm = compute(y_true, y_pred)
    mDice = dice[1:].mean()   # 5 structures, no decay/BG to exclude beyond BG

    tag = "12-class collapsed -> structures" if args.remap_12 else "Layer-1 structure model"
    print(f"\n{'='*72}\n{tag}   ({args.pred_dir})\n{'='*72}")
    print(f"{'Structure':<12}{'Dice':>7}{'IoU':>7}{'Prec':>7}{'Rec':>7}   raw per-root reference")
    print("-" * 72)
    for i, name in enumerate(CLASS_NAMES):
        ref = RAW_ROOT_REFERENCE.get(name, ("",))[0]
        print(f"{name:<12}{dice[i]:>7.3f}{iou[i]:>7.3f}{prec[i]:>7.3f}{rec[i]:>7.3f}   {ref}")
    print(f"\nmDice (5 structures): {mDice:.4f}")

    pd.DataFrame({"Class": CLASS_NAMES, "Dice": dice, "IoU": iou,
                  "Precision": prec, "Recall": rec}).to_csv(args.csv_out, index=False)
    print(f"Saved -> {args.csv_out}")

    suffix = "_remap12" if args.remap_12 else ""
    save_confusion_matrix(
        cm,
        png_path=os.path.join(OUT_DIR, f"confusion_matrix_layer1{suffix}.png"),
        csv_path=os.path.join(OUT_DIR, f"confusion_matrix_layer1{suffix}.csv"),
        title=f"{tag}\nmDice (5 structures) = {mDice:.4f}",
    )


if __name__ == "__main__":
    main()
