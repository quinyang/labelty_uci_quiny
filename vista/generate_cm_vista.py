import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os
import glob
import pandas as pd

NUM_CLASSES = 12
CLASS_NAMES = [
    "Background",    # 0
    "Apical Lesion", # 1
    "Main Root",     # 2
    "Main Canal",    # 3
    "Mesial Root",   # 4
    "Mesial Canal",  # 5
    "Distal Root",   # 6
    "Distal Canal",  # 7
    "Palatal Root",  # 8
    "Palatal Canal", # 9
    "RC Filling",    # 10
    "Decay",         # 11
]

pred_dir = "/home/jiakuny1/Projects/vista_predictions"
gt_dir   = "/home/jiakuny1/Projects/resource/labels_mask"

all_y_true = []
all_y_pred = []

pred_files = sorted(glob.glob(os.path.join(pred_dir, "*.png")))
print(f"Found {len(pred_files)} predictions. Loading images...")

for pred_path in pred_files:
    filename = os.path.basename(pred_path)
    gt_path  = os.path.join(gt_dir, filename)

    if not os.path.exists(gt_path):
        print(f"  -> Skipping {filename}: no matching GT.")
        continue

    pred_mask = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
    gt_mask   = cv2.imread(gt_path,   cv2.IMREAD_GRAYSCALE)
    if pred_mask is None or gt_mask is None:
        continue

    if pred_mask.shape != gt_mask.shape:
        pred_mask = cv2.resize(pred_mask, (gt_mask.shape[1], gt_mask.shape[0]),
                               interpolation=cv2.INTER_NEAREST)

    all_y_pred.append(pred_mask.flatten())
    all_y_true.append(gt_mask.flatten())

print("Concatenating pixels...")
y_true = np.concatenate(all_y_true)
y_pred = np.concatenate(all_y_pred)

print("Calculating confusion matrix...")
labels = list(range(NUM_CLASSES))
cm_raw = confusion_matrix(y_true, y_pred, labels=labels)


def compute_metrics(cm):
    dice, iou, precision, recall = [], [], [], []
    for i in range(cm.shape[0]):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        dice.append(2 * tp / (2 * tp + fp + fn + 1e-8))
        iou.append(tp / (tp + fp + fn + 1e-8))
        precision.append(tp / (tp + fp + 1e-8))
        recall.append(tp / (tp + fn + 1e-8))
    return (np.array(dice), np.array(iou),
            np.array(precision), np.array(recall))


dice_cls, iou_cls, prec_cls, rec_cls = compute_metrics(cm_raw)

print("\n=== Per-Class Metrics ===")
print(f"{'Class':<20} {'Dice':>7} {'IoU':>7} {'Prec':>7} {'Recall':>7}")
print("-" * 52)
for i, name in enumerate(CLASS_NAMES):
    print(f"{name:<20} {dice_cls[i]:>7.4f} {iou_cls[i]:>7.4f} "
          f"{prec_cls[i]:>7.4f} {rec_cls[i]:>7.4f}")
print(f"\nmDice (excl. bg) : {dice_cls[1:].mean():.4f}")
print(f"mIoU  (excl. bg) : {iou_cls[1:].mean():.4f}")

csv_path = "/home/jiakuny1/Projects/metrics_vista.csv"
pd.DataFrame({
    "Class": CLASS_NAMES,
    "Dice": dice_cls,
    "IoU": iou_cls,
    "Precision": prec_cls,
    "Recall": rec_cls,
}).to_csv(csv_path, index=False)
print(f"Saved metrics to {csv_path}")

row_sums = cm_raw.sum(axis=1, keepdims=True).astype(float)
row_sums[row_sums == 0] = 1
cm_norm = cm_raw / row_sums

fig = plt.figure(figsize=(22, 8))
gs  = gridspec.GridSpec(1, 2, width_ratios=[3, 2], figure=fig)

ax1 = fig.add_subplot(gs[0])
sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Purples",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            ax=ax1, annot_kws={"size": 7}, vmin=0, vmax=1)
ax1.set_xlabel("Predicted Class", fontsize=10)
ax1.set_ylabel("True Class (Ground Truth)", fontsize=10)
ax1.set_title("DeepLabV3+ (ResNet101) — Pixel-wise Confusion Matrix (row-normalised)", fontsize=12)
plt.setp(ax1.get_xticklabels(), rotation=45, ha="right", fontsize=8)
plt.setp(ax1.get_yticklabels(), rotation=0,  fontsize=8)

ax2 = fig.add_subplot(gs[1])
fg_names = CLASS_NAMES[1:]
fg_dice  = dice_cls[1:]
fg_iou   = iou_cls[1:]
x     = np.arange(len(fg_names))
width = 0.35
ax2.bar(x - width / 2, fg_dice, width, label="Dice", color="mediumpurple", alpha=0.85)
ax2.bar(x + width / 2, fg_iou,  width, label="IoU",  color="slateblue",   alpha=0.85)
ax2.axhline(fg_dice.mean(), color="mediumpurple", linestyle="--", alpha=0.7,
            label=f"mDice={fg_dice.mean():.3f}")
ax2.axhline(fg_iou.mean(),  color="slateblue",   linestyle="--", alpha=0.7,
            label=f"mIoU={fg_iou.mean():.3f}")
ax2.set_xticks(x)
ax2.set_xticklabels(fg_names, rotation=45, ha="right", fontsize=8)
ax2.set_ylabel("Score")
ax2.set_ylim(0, 1)
ax2.set_title("Per-Class Dice & IoU (background excluded)", fontsize=11)
ax2.legend(fontsize=8)

plt.tight_layout()
out_path = "/home/jiakuny1/Projects/confusion_matrix_vista.png"
plt.savefig(out_path, dpi=200, bbox_inches="tight")
print(f"\nSaved CM figure to {out_path}")
