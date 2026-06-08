"""
Inference for the DeepLabV3+ dental segmentation model.

Matches train_vista.py: LetterboxResized preprocessing (no in-pipeline CLAHE),
and a 5-fold ENSEMBLE — softmax probabilities are averaged across all available
fold checkpoints and horizontal-flip TTA. The letterbox is inverted exactly
(crop the [:nA, :nB] content region, then resize back to the original size).
"""
import os
import glob
import cv2
import torch
import torch.nn.functional as F
import numpy as np
import segmentation_models_pytorch as smp
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd,
    RepeatChanneld, MapTransform,
)

NUM_CLASSES = 12
CANVAS = 1024
NUM_FOLDS = 5
SAVE_ROOT = "/home/jiakuny1/Projects"

test_images_dir = "/home/jiakuny1/Projects/nnUNet_data/nnUNet_raw/Dataset101_Dental/imagesTs"
output_dir      = "/home/jiakuny1/Projects/vista_predictions"
os.makedirs(output_dir, exist_ok=True)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"Testing on device: {device}")

test_images = sorted(glob.glob(os.path.join(test_images_dir, "*_0000.png")))
print(f"Found {len(test_images)} test images.")


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


def build_model():
    return smp.DeepLabV3Plus(
        encoder_name="resnet50",
        encoder_weights=None,
        in_channels=3,
        classes=NUM_CLASSES,
        decoder_dropout=0.2,
    ).to(device)


fold_paths = [os.path.join(SAVE_ROOT, f"vista2d_fold{f}.pth") for f in range(NUM_FOLDS)]
fold_paths = [p for p in fold_paths if os.path.exists(p)]
if not fold_paths:
    raise FileNotFoundError(
        f"No fold checkpoints (vista2d_fold0.pth ..) in {SAVE_ROOT}. Run train_vista.py first."
    )
print(f"Ensembling {len(fold_paths)} fold checkpoints:")
models = []
for p in fold_paths:
    print(f"  {p}")
    m = build_model()
    m.load_state_dict(torch.load(p, map_location=device, weights_only=True))
    m.eval()
    models.append(m)

# image-only pipeline mirrors training (minus label + copy_paste)
shape_loader = Compose([LoadImaged(keys=["image"]), EnsureChannelFirstd(keys=["image"])])
test_transform = Compose([
    LoadImaged(keys=["image"]),
    EnsureChannelFirstd(keys=["image"]),
    ScaleIntensityd(keys=["image"]),
    RepeatChanneld(keys=["image"], repeats=3),
    LetterboxResized(keys=["image"], canvas=CANVAS, modes=("bilinear",)),
])

print(f"Running letterbox inference at {CANVAS}² with {len(models)}-fold ensemble + h-flip TTA...")

with torch.no_grad():
    for img_path in test_images:
        filename = os.path.basename(img_path).replace("_0000.png", ".png")

        original_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        original_h, original_w = original_img.shape

        # MONAI-frame loaded shape (handles its W/H transpose); used to size the crop
        A, B = shape_loader({"image": img_path})["image"].shape[1:]
        nA, nB = LetterboxResized.scaled_size(A, B, CANVAS)

        x = test_transform({"image": img_path})["image"].unsqueeze(0).to(device)
        xf = torch.flip(x, dims=[3])

        prob_sum = None
        for model in models:
            for inp, flip_back in ((x, False), (xf, True)):
                logits = model(inp)
                if logits.shape[2:] != inp.shape[2:]:
                    logits = F.interpolate(logits, size=inp.shape[2:],
                                           mode="bilinear", align_corners=False)
                prob = torch.softmax(logits, dim=1)
                if flip_back:
                    prob = torch.flip(prob, dims=[3])
                prob_sum = prob if prob_sum is None else prob_sum + prob

        prediction = torch.argmax(prob_sum, dim=1).squeeze().cpu().numpy().astype(np.uint8)

        # Invert letterbox: crop the real content [:nA, :nB], undo MONAI's (W,H)
        # transpose, then resize back to the native image size.
        prediction = prediction[:nA, :nB]
        prediction = prediction.T

        if prediction.shape != (original_h, original_w):
            prediction = cv2.resize(prediction, (original_w, original_h),
                                    interpolation=cv2.INTER_NEAREST)

        cv2.imwrite(os.path.join(output_dir, filename), prediction)

print(f"\nDone! Masks saved to: {output_dir}")
