import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

# ── Config ────────────────────────────────────────────────────────────────────
# All test-set image IDs that have predictions in every model directory.
# files to test with: 70 259 515 629 754 855 921
IMAGE_IDS = [259, 70, 515, 629, 754, 855, 921]

OUTPUT_DIR = "/home/jiakuny1/Projects/viz_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BASE = "/home/jiakuny1/Projects"
RAW_DIR  = f"{BASE}/nnUNet_data/nnUNet_raw/Dataset101_Dental/imagesTs"

MODELS = {
    "nnU-Net":       f"{BASE}/nnUNet_data/model_predictions",
    "Pretrained UNet": f"{BASE}/unet_predictions",
    "VISTA2D":       f"{BASE}/vista_predictions",
    "Swin-UNet":     f"{BASE}/swin_predictions",
}

# 12-class dental colour map (class 0 = background stays transparent)
# Classes 1-11 get distinct colours; class 0 rendered transparent via alpha.
CLASS_COLORS = np.array([
    [0,   0,   0,   0  ],   # 0  background (transparent)
    [255, 0,   0,   200],   # 1  red
    [0,   200, 0,   200],   # 2  green
    [0,   80,  255, 200],   # 3  blue
    [255, 165, 0,   200],   # 4  orange
    [160, 32,  240, 200],   # 5  purple
    [0,   220, 220, 200],   # 6  cyan
    [255, 20,  147, 200],   # 7  deep pink
    [255, 255, 0,   200],   # 8  yellow
    [0,   128, 0,   200],   # 9  dark green
    [255, 105, 180, 200],   # 10 hot pink
    [100, 149, 237, 200],   # 11 cornflower blue
], dtype=np.uint8)


def mask_to_rgba(mask: np.ndarray) -> np.ndarray:
    """Convert a class-index mask (H×W uint8) to an RGBA overlay image."""
    n_classes = CLASS_COLORS.shape[0]
    mask_clipped = np.clip(mask, 0, n_classes - 1)
    return CLASS_COLORS[mask_clipped]          # shape H×W×4


def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot load: {path}")
    return img


def make_legend_patches():
    labels = [
        "1 Upper-R Molar", "2 Upper-R Premolar", "3 Upper-L Premolar",
        "4 Upper-L Molar", "5 Upper Anterior", "6 Lower-R Molar",
        "7 Lower-R Premolar", "8 Lower-L Premolar", "9 Lower-L Molar",
        "10 Lower Anterior", "11 Other",
    ]
    patches = []
    for i, lbl in enumerate(labels, start=1):
        c = CLASS_COLORS[i, :3] / 255.0
        patches.append(mpatches.Patch(color=c, label=lbl))
    return patches


# ── Per-image comparison figures ──────────────────────────────────────────────
for img_id in IMAGE_IDS:
    filename = f"IO000001_({img_id})"
    raw_path = f"{RAW_DIR}/{filename}_0000.png"

    try:
        raw = load_image(raw_path)
    except FileNotFoundError as e:
        print(f"  SKIP {filename}: {e}")
        continue

    n_cols = 1 + len(MODELS)
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 5))
    fig.patch.set_facecolor("#1a1a1a")

    # Column 0: original X-ray
    axes[0].imshow(raw, cmap="gray")
    axes[0].set_title("Original X-Ray", color="white", fontsize=11, pad=6)
    axes[0].axis("off")

    # Columns 1-4: model overlays
    for col, (model_name, pred_dir) in enumerate(MODELS.items(), start=1):
        mask_path = f"{pred_dir}/{filename}.png"
        ax = axes[col]
        ax.imshow(raw, cmap="gray")
        try:
            mask = load_image(mask_path)
            overlay = mask_to_rgba(mask)
            ax.imshow(overlay)
        except FileNotFoundError:
            ax.text(0.5, 0.5, "No prediction", color="red",
                    ha="center", va="center", transform=ax.transAxes)
        ax.set_title(model_name, color="white", fontsize=11, pad=6)
        ax.axis("off")

    fig.suptitle(f"Dental Segmentation — Image {img_id}",
                 color="white", fontsize=14, y=1.01)

    # Legend below the figure
    patches = make_legend_patches()
    fig.legend(handles=patches, loc="lower center", ncol=6,
               fontsize=7, framealpha=0.3, labelcolor="white",
               bbox_to_anchor=(0.5, -0.08))

    plt.tight_layout()
    out_path = f"{OUTPUT_DIR}/compare_{img_id}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── Combined grid (all images × all models) ───────────────────────────────────
n_rows = len(IMAGE_IDS)
n_cols = 1 + len(MODELS)

fig, axes = plt.subplots(n_rows, n_cols,
                         figsize=(3.5 * n_cols, 3.5 * n_rows))
fig.patch.set_facecolor("#1a1a1a")

col_titles = ["Original"] + list(MODELS.keys())
for col, title in enumerate(col_titles):
    axes[0, col].set_title(title, color="white", fontsize=10, pad=4)

for row, img_id in enumerate(IMAGE_IDS):
    filename = f"IO000001_({img_id})"
    raw_path = f"{RAW_DIR}/{filename}_0000.png"

    axes[row, 0].set_ylabel(f"#{img_id}", color="white",
                             fontsize=9, rotation=0, labelpad=30, va="center")

    try:
        raw = load_image(raw_path)
    except FileNotFoundError:
        for col in range(n_cols):
            axes[row, col].axis("off")
        continue

    axes[row, 0].imshow(raw, cmap="gray")
    axes[row, 0].axis("off")

    for col, (model_name, pred_dir) in enumerate(MODELS.items(), start=1):
        mask_path = f"{pred_dir}/{filename}.png"
        ax = axes[row, col]
        ax.imshow(raw, cmap="gray")
        try:
            mask = load_image(mask_path)
            ax.imshow(mask_to_rgba(mask))
        except FileNotFoundError:
            ax.text(0.5, 0.5, "N/A", color="red",
                    ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")

fig.suptitle("Dental Segmentation — All Models × All Images",
             color="white", fontsize=15, y=1.005)

patches = make_legend_patches()
fig.legend(handles=patches, loc="lower center", ncol=6,
           fontsize=7, framealpha=0.3, labelcolor="white",
           bbox_to_anchor=(0.5, -0.02))

plt.tight_layout()
grid_path = f"{OUTPUT_DIR}/compare_grid_all.png"
plt.savefig(grid_path, dpi=120, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close(fig)
print(f"\nSaved combined grid: {grid_path}")
print("Done.")
