"""
Generate LAYER-1 PROX label masks (12-class -> 5-class root-structure target).

Variant of make_layer1_labels.py. Mesial and Distal are MERGED into a single
"Proximal" root class, because mesial-vs-distal is under-determined from an
isolated tooth crop (measured 2026-06-01: laterality is 51/49 wrt image position,
and there is no quadrant/tooth-number context in the data to break the tie). The
model already localises the proximal-root REGION well (merged Dice 0.726 vs split
Mesial 0.708 / Distal 0.466) — only the naming was wrong. Layer 2 will attempt the
mesial/distal split *inside* the proximal region using concavity + canal count.

Remap (source 12-class value -> Layer-1-prox class):
  0 background / 11 decay      -> 0  BG
  2 Main Root / 3 Main Canal   -> 1  Main      (root UNION canal)
  4 Mesial Root / 5 Mesial Canal\
  6 Distal Root / 7 Distal Canal/ -> 2  Proximal   (mesial + distal merged)
  8 Palatal Root / 9 Palatal Canal -> 3  Palatal
  1 Apical Lesion              -> 4  Apical
  10 RC Filling -> nearest of {Main, Proximal, Palatal} (solidifies the region).

Usage:
    python make_layer1prox_labels.py            # remap all three label dirs
    python make_layer1prox_labels.py --sample   # also dump a few RGB overlays
"""
import argparse
import glob
import os
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm

# source(12-class) -> Layer-1-prox class, for everything except RC Filling (10)
BASE_MAP = {0: 0, 11: 0, 1: 4, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2, 7: 2, 8: 3, 9: 3}
FILLING_SRC = 10
STRUCT_LABELS = (1, 2, 3)               # absorb filling into one of these only (not Apical)

NUM_CLASSES_L1 = 5
CLASS_NAMES_L1 = ["BG", "Main", "Proximal", "Palatal", "Apical"]
PALETTE = np.array([                    # for --sample visualisation only
    [0, 0, 0], [220, 50, 50], [50, 200, 80], [240, 200, 40], [200, 80, 220],
], dtype=np.uint8)

# (input dir, output dir) pairs. labelsTr feeds val; the aug set feeds train;
# labels_mask is the test GT used by evaluate_layer1prox.py.
DIRS = [
    ("/home/jiakuny1/Projects/nnUNet_data/nnUNet_raw/Dataset101_Dental/labelsTr",
     "/home/jiakuny1/Projects/nnUNet_data/nnUNet_raw/Dataset101_Dental/labelsTr_layer1prox"),
    ("/home/jiakuny1/Projects/nnUNet_data/nnUNet_raw/Dataset102_Dental_Aug/labelsTr",
     "/home/jiakuny1/Projects/nnUNet_data/nnUNet_raw/Dataset102_Dental_Aug/labelsTr_layer1prox"),
    ("/home/jiakuny1/Projects/resource/labels_mask",
     "/home/jiakuny1/Projects/resource/labels_mask_layer1prox"),
]


def remap_to_layer1prox(lbl):
    """12-class mask -> 5-class Layer-1-prox mask. RC Filling -> nearest structure."""
    out = np.zeros_like(lbl, dtype=np.uint8)
    for src, dst in BASE_MAP.items():
        out[lbl == src] = dst

    fill = lbl == FILLING_SRC
    if fill.any():
        struct_mask = np.isin(out, STRUCT_LABELS)
        if struct_mask.any():
            idx = distance_transform_edt(~struct_mask, return_distances=False,
                                         return_indices=True)
            nearest = out[tuple(idx)]
            out[fill] = nearest[fill]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true", help="dump RGB sanity overlays")
    args = ap.parse_args()

    sample_dir = "/home/jiakuny1/Projects/layer1prox_label_samples"
    if args.sample:
        os.makedirs(sample_dir, exist_ok=True)

    grand_presence = np.zeros(NUM_CLASSES_L1, dtype=int)
    n_total = 0
    for in_dir, out_dir in DIRS:
        paths = sorted(glob.glob(os.path.join(in_dir, "*.png")))
        if not paths:
            print(f"!! no PNGs in {in_dir} — skipping")
            continue
        os.makedirs(out_dir, exist_ok=True)
        print(f"\n{in_dir}\n  -> {out_dir}   ({len(paths)} masks)")
        for i, p in enumerate(tqdm(paths, desc=os.path.basename(out_dir))):
            m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if m is None:
                print(f"  !! unreadable: {p}")
                continue
            r = remap_to_layer1prox(m)
            cv2.imwrite(os.path.join(out_dir, os.path.basename(p)), r)
            for c in np.unique(r):
                grand_presence[c] += 1
            n_total += 1
            if args.sample and out_dir.endswith("labels_mask_layer1prox") and i < 6:
                cv2.imwrite(os.path.join(sample_dir, os.path.basename(p)), PALETTE[r])

    print(f"\nDone. {n_total} masks remapped.")
    print("Per-class image presence (all dirs combined):")
    for c in range(NUM_CLASSES_L1):
        print(f"  {CLASS_NAMES_L1[c]:<10} {grand_presence[c]:>6}")
    if args.sample:
        print(f"Sample overlays -> {sample_dir}")


if __name__ == "__main__":
    main()
