# Dental Radiograph Segmentation — Project Report

A semantic-segmentation effort on peri-apical dental X-rays. The goal: pixel-wise
segmentation of tooth root structures (roots, canals, apical lesions) from a small
hand-annotated dataset, and pushing the accuracy of the hardest structures as far
as the data allows.

This report documents the full arc — including the approaches that **did not** work,
because the negative results shaped every decision that followed.

---

## 1. Starting point — 12 classes, raw

- **Data:** ~436 hand-annotated training images (+ a held-out test set of 110),
  annotated as YOLO polygons converted to pixel masks (`yolo_2_mask.py`).
- **Label space:** 12 classes — Background, Apical Lesion, and Root+Canal pairs for
  each root: Main, Mesial, Distal, Palatal (Root + Canal each), plus Root-Canal
  Filling and Decay.
- **First models, trained directly:** a U-Net and a DeepLabV3+ baseline.

**Result: near-failure.** Baseline mean-Dice (excluding background) was ~**0.06** for
both. Root causes, found by inspection:
- models used single-channel input → threw away ImageNet pretraining;
- the *Decay* class had **0 pixels** in the training data yet a large loss weight,
  injecting pure noise;
- rare classes (esp. *Distal Canal*, present in only ~19 of 436 images) had no
  oversampling.

## 2. Testing different models & fixing the fundamentals

Architectures evaluated: **U-Net (EfficientNet encoder)**, **DeepLabV3+ (ResNet)**,
**SwinUNETR (transformer)**, and **nnU-Net** as a strong reference.

Fixes applied:
- 3-channel input (`RepeatChannel`) to preserve ImageNet encoder weights;
- zeroed the empty Decay class weight;
- 5-fold cross-validation; ensemble + horizontal-flip test-time augmentation;
- per-class Dice logged to CSV for error analysis.

### The decisive bug — coordinate transpose
A large, persistent val/test gap (val ~0.52, test ~0.06) looked like overfitting but
was **a coordinate-convention bug in the prediction export path**: MONAI loads 2-D
arrays as `(W, H)` while OpenCV and the ground-truth masks use `(H, W)`. Training and
validation were self-consistent (both transposed), so val Dice was real — but the
exported test masks were transposed relative to the GT, giving near-zero overlap.

- **Signature that localized it:** every class, *including background*, scored near
  zero (background Dice ~0.77 vs nnU-Net's ~0.97); class *fractions* were correct but
  spatial overlap was not. nnU-Net was immune (consistent I/O).
- **Fix:** transpose the prediction back before resizing to native resolution.
- **Impact:** U-Net test mean-Dice **0.068 → 0.54**.

**Lesson:** when one pipeline works on identical data/GT/eval and the custom ones get
uniform near-zero across *all* classes, suspect an orientation/pipeline bug, not the model.

## 3. Data augmentation & imbalance handling

- Geometric/intensity: elastic deformation, CLAHE, zoom (0.6–1.4), coarse dropout.
- **Weighted sampler** oversampling rare-class images.
- **Copy-paste augmentation:** rare-structure crops pasted into other training images.
- A larger augmented training pool (~5,668 images) generated offline.

## 4. Reorganizing the data — the two-layer cascade

The core difficulty was **which-tooth-structure** disambiguation (mesial vs distal vs
palatal), not finding the structures. The cascade idea:

- **Layer 1** — segment the whole root *structure* per position (root ∪ canal, made
  solid), 6 classes: BG, Main, Mesial, Distal, Palatal, Apical. The hard positional
  problem is solved once, at the structure level.
- **Layer 2** — find the canal *inside* each Layer-1 region; the canal inherits its
  identity from the parent region.

Labels were remapped 12→6 (`make_layer1_labels.py`), absorbing the root-canal filling
into the nearest structure via a distance transform so each region is solid.

**Layer-1 (6-class) result:** test mean-Dice **0.664** (5-fold ensemble + TTA).
Per-structure: Apical 0.80, Main 0.71, Mesial 0.71, **Distal 0.47** (the weak one),
Palatal 0.63, Background 0.97.

## 5. The hard problem — Distal vs Mesial roots

Distal was the worst class (0.47) and the cascade's main target. Error analysis showed
its dominant failure was **positional**: 32% of Distal leaked into Mesial (its
mirror-twin root), and it was recall-limited. A sequence of targeted attempts:

| # | Attempt | Outcome |
|---|---------|---------|
| 1 | **Focal-Tversky + clDice loss + heavier oversampling** | **Regressed.** Fold-0 val mean-Dice 0.61→0.58; Distal flat; Apical dropped ~0.05 (clDice — a centerline objective — is wrong for blob-like lesions). Rejected at the 1-fold gate. |
| 2 | **Swin Transformer (global context)** per-class | **Tie.** Collapsed-to-structure Distal 0.457 vs U-Net 0.466 — no gain, worse on every other class. Global context within a single crop did not help. *(Also caught a stale-prediction trap: on-disk Swin predictions predated the transpose fix and produced bogus scores until regenerated.)* |
| 3 | **Quantitative laterality measurement** | **Decisive.** Across 240 images with both roots, Mesial sat left-of-Distal in **51%** and right-of in 49% — a coin flip. With no tooth-numbering/quadrant context in the data, mesial-vs-distal is **under-determined from an isolated crop.** |

**Conclusion:** the distinction was *not learnable* from the available inputs — not a
model-capacity problem. The fix was a **reformulation**, not more ML:

### Merge Mesial + Distal → "Proximal" (deferred split to Layer 2)
The model already localized the *region* correctly (merged Dice 0.726 vs the 0.47 split
Distal); only the naming was wrong. Merging eliminates the impossible distinction; the
mesial/distal split is deferred to Layer 2, where a genuinely discriminative signal
exists — **canal count** (the mesial root typically has 2 canals, the distal 1) and
root concavity.

**Layer-1-prox (5-class) result:** test mean-Dice **0.727** (+0.063 over 0.664).
Per-structure: Apical 0.80, **Proximal 0.74** (now the 2nd-strongest class), Main 0.72,
Palatal 0.65, Background 0.97. Zero regressions; the catastrophic Distal class is gone.

## 6. Pushing further (on the 0.727 model)

With Palatal now the weakest class (0.65, recall-limited — it under-segments because the
palatal root superimposes behind the buccal roots in 2-D projection), three more levers
were tried:

| Attempt | Outcome |
|---------|---------|
| **Multi-scale test-time augmentation** (0.85/1.0/1.15) | **Null** (0.7272→0.7276). Cleanly ruled out resolution — the model is already scale-invariant from zoom augmentation, so the boundary leak is a capacity/threshold issue, not scale. |
| **Encoder capacity bump (EfficientNet b2→b4)** | **Underperformed.** b4 (2.2× params) tracked *below* b2 at matched epochs on the fold-0 gate; dropped. Suggests the 436-image dataset, not model size, is the ceiling. |
| **Palatal-targeted recall** (Tversky β 0.7→0.75, Palatal loss weight ×2) | **Tie** on the fold-0 gate (best val 0.6702 vs 0.6699); Palatal not clearly lifted. |

## 7. Results summary

| Stage | Classes | Test mean-Dice |
|-------|---------|----------------|
| Raw 12-class U-Net (initial) | 12 | ~0.06 |
| 12-class, after transpose fix | 12 | ~0.54 |
| **Layer-1 cascade (6-class)** | 6 | **0.664** |
| **Layer-1-prox (Mesial+Distal merged)** | 5 | **0.727** |

(mean-Dice over foreground structures, 5-fold ensemble + TTA, on the 110-image test set.)

## 8. Methodology notes

- **Smoke-test gating:** every expensive idea was validated on a single CV fold before
  committing all five, rejecting weak configs early and saving ~12 GPU-hours each.
- **Baselines preserved:** every experiment wrote tagged checkpoints so prior best
  models were never overwritten.
- **Measure before modeling:** the laterality study replaced a guessing game with a
  data-driven decision, and the confusion-matrix/background-Dice signatures localized
  bugs that looked like modeling failures.

## 9. Future work

- **Layer 2:** segment canals inside each Proximal region and recover the mesial/distal
  split via canal count + concavity (the discriminator unavailable to Layer 1).
- **Palatal:** the remaining headroom is a real (non-ill-posed) class; a diverse 2nd
  architecture in the ensemble, or more annotated maxillary-molar data, are the likely
  levers.
- **Tooth-position prior:** if tooth-numbering / full-arch context can be obtained, the
  mesial/distal naming becomes recoverable at the structure level.

---

## Repository layout

| Path | Contents |
|------|----------|
| `layer1/` | The cascade: `make_*_labels.py`, `train_*`, `test_*`, `evaluate_*` for the 6-class and 5-class (prox) Layer-1 models |
| `p_unet/` | U-Net (EfficientNet) train/test/confusion-matrix scripts |
| `vista/` | DeepLabV3+ scripts |
| `swin/` | SwinUNETR scripts |
| `nnunet/` | nnU-Net formatting / stats / confusion-matrix |
| `yolo_2_mask.py` | YOLO polygon → pixel-mask conversion |
| `visualize.py` | Overlay / visualization utilities |

> Model weights, datasets, metrics CSVs, confusion matrices and training logs are **not**
> in git (size). They are archived separately (Google Drive).
