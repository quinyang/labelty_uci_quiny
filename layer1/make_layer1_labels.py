"""
Generate LAYER-1 label masks (12-class -> 6-class root-structure target).

The two-layer cascade (2026-05-28):
  Layer 1 answers "WHERE and WHICH tooth structure" — it segments the whole
  root structure per position (root UNION canal, made solid) plus apical lesion.
  Layer 2 will later answer "is this pixel a canal" *inside* each Layer-1 region,
  and the canal inherits its identity (mesial/distal/...) from the parent region.
  This kills the worst error in the 12-class model (Distal Canal -> Palatal Root,
  0.87 in the nnU-Net confusion matrix): the mesial/distal/palatal disambiguation
  is solved ONCE, at the structure level.

Remap (source 12-class value -> Layer-1 class):
  0  background        -> 0   BG
  11 decay (0 pixels)  -> 0   BG
  2  Main Root    \
  3  Main Canal   / -> 1   Main      (root UNION canal)
  4  Mesial Root  \
  5  Mesial Canal / -> 2   Mesial
  6  Distal Root  \
  7  Distal Canal / -> 3   Distal
  8  Palatal Root \
  9  Palatal Canal/ -> 4   Palatal
  1  Apical Lesion    -> 5   Apical   (independent finding, sits at the apex)
  10 Root Canal Filling -> assigned to the NEAREST of {Main,Mesial,Distal,Palatal}.
       Rationale: a filling replaces the canal lumen (filling subset canal subset
       root), so in the GT it punches a hole in the structure region. Absorbing it
       into the nearest structure makes each Layer-1 region SOLID, which is exactly
       what Layer 2 needs to look "inside". Identity (which tooth) is recovered
       from the surrounding structure via a distance transform.

Usage:
    python make_layer1_labels.py            # remap all three label dirs
    python make_layer1_labels.py --sample   # also dump a few RGB sanity overlays
"""
import argparse
import glob
import os
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm

# source(12-class) -> Layer-1 class, for everything except RC Filling (10)
BASE_MAP = {0: 0, 11: 0, 1: 5, 2: 1, 3: 1, 4: 2, 5: 2, 6: 3, 7: 3, 8: 4, 9: 4}
FILLING_SRC = 10
STRUCT_LABELS = (1, 2, 3, 4)            # absorb filling into one of these only (not Apical)

NUM_CLASSES_L1 = 6
CLASS_NAMES_L1 = ["BG", "Main", "Mesial", "Distal", "Palatal", "Apical"]
PALETTE = np.array([                    # for --sample visualisation only
    [0, 0, 0], [220, 50, 50], [50, 200, 80], [60, 120, 240],
    [240, 200, 40], [200, 80, 220],
], dtype=np.uint8)

# (input dir, output dir) pairs. labelsTr feeds val; the aug set feeds train;
# labels_mask is the test GT used by evaluate_layer1.py.
DIRS = [
    ("/home/jiakuny1/Projects/nnUNet_data/nnUNet_raw/Dataset101_Dental/labelsTr",
     "/home/jiakuny1/Projects/nnUNet_data/nnUNet_raw/Dataset101_Dental/labelsTr_layer1"),
    ("/home/jiakuny1/Projects/nnUNet_data/nnUNet_raw/Dataset102_Dental_Aug/labelsTr",
     "/home/jiakuny1/Projects/nnUNet_data/nnUNet_raw/Dataset102_Dental_Aug/labelsTr_layer1"),
    ("/home/jiakuny1/Projects/resource/labels_mask",
     "/home/jiakuny1/Projects/resource/labels_mask_layer1"),
]


def remap_to_layer1(lbl):
    """12-class mask -> 6-class Layer-1 mask. RC Filling -> nearest structure."""
    out = np.zeros_like(lbl, dtype=np.uint8)
    for src, dst in BASE_MAP.items():
        out[lbl == src] = dst

    fill = lbl == FILLING_SRC
    if fill.any():
        struct_mask = np.isin(out, STRUCT_LABELS)
        if struct_mask.any():
            # for every pixel, index of the nearest structure pixel
            idx = distance_transform_edt(~struct_mask, return_distances=False,
                                         return_indices=True)
            nearest = out[tuple(idx)]
            out[fill] = nearest[fill]
        # else: no structure in the image -> leave filling as BG (vanishingly rare)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true", help="dump RGB sanity overlays")
    args = ap.parse_args()

    sample_dir = "/home/jiakuny1/Projects/layer1_label_samples"
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
            r = remap_to_layer1(m)
            cv2.imwrite(os.path.join(out_dir, os.path.basename(p)), r)
            for c in np.unique(r):
                grand_presence[c] += 1
            n_total += 1
            if args.sample and out_dir.endswith("labels_mask_layer1") and i < 6:
                cv2.imwrite(os.path.join(sample_dir, os.path.basename(p)), PALETTE[r])

    print(f"\nDone. {n_total} masks remapped.")
    print("Per-class image presence (all dirs combined):")
    for c in range(NUM_CLASSES_L1):
        print(f"  {CLASS_NAMES_L1[c]:<8} {grand_presence[c]:>6}")
    if args.sample:
        print(f"Sample overlays -> {sample_dir}")


if __name__ == "__main__":
    main()
