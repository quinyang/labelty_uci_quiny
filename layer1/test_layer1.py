"""
LAYER 1 inference — 5-fold ensemble + h-flip TTA, letterbox-exact inverse.

Clone of p_unet/test_unet.py for the 6-class structure model. Same critical
orientation fix as test_unet.py: crop the [:nA,:nB] content, then `prediction.T`
to undo MONAI's (W,H) load transpose before resizing back (the transpose bug that
sank the 12-class test scores — see project memory 2026-05-22).

Outputs:
  layer1_predictions/<stem>.png        argmax structure maps (0..5) for evaluation
  layer1_probs/<stem>.npz   (optional)  float16 softmax volume (6,H,W) in the
                                        native image frame — the conditioning
                                        signal Layer 2 will ingest. Enable with
                                        --save-probs.

Usage:
    python test_layer1.py                 # write label maps only
    python test_layer1.py --save-probs    # also dump prob volumes for Layer 2
"""
import argparse
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

NUM_CLASSES = 6
CANVAS = 1024
NUM_FOLDS = 5
SAVE_ROOT = "/home/jiakuny1/Projects"

test_images_dir = "/home/jiakuny1/Projects/nnUNet_data/nnUNet_raw/Dataset101_Dental/imagesTs"
output_dir = "/home/jiakuny1/Projects/layer1_predictions"
prob_dir = "/home/jiakuny1/Projects/layer1_probs"

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


class LetterboxResized(MapTransform):
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
    return smp.Unet(encoder_name="efficientnet-b2", encoder_weights=None,
                    in_channels=3, classes=NUM_CLASSES, decoder_dropout=0.2).to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save-probs", action="store_true",
                    help="also save (6,H,W) float16 softmax volumes for Layer 2")
    args = ap.parse_args()

    os.makedirs(output_dir, exist_ok=True)
    if args.save_probs:
        os.makedirs(prob_dir, exist_ok=True)

    print(f"Testing on device: {device}")
    test_images = sorted(glob.glob(os.path.join(test_images_dir, "*_0000.png")))
    print(f"Found {len(test_images)} test images.")

    fold_paths = [os.path.join(SAVE_ROOT, f"layer1_fold{f}.pth") for f in range(NUM_FOLDS)]
    fold_paths = [p for p in fold_paths if os.path.exists(p)]
    if not fold_paths:
        raise FileNotFoundError(f"No layer1_fold*.pth in {SAVE_ROOT}. Run train_layer1.py first.")
    print(f"Ensembling {len(fold_paths)} fold checkpoints:")
    models = []
    for p in fold_paths:
        print(f"  {p}")
        m = build_model()
        m.load_state_dict(torch.load(p, map_location=device, weights_only=True))
        m.eval()
        models.append(m)

    shape_loader = Compose([LoadImaged(keys=["image"]), EnsureChannelFirstd(keys=["image"])])
    test_transform = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        ScaleIntensityd(keys=["image"]),
        RepeatChanneld(keys=["image"], repeats=3),
        LetterboxResized(keys=["image"], canvas=CANVAS, modes=("bilinear",)),
    ])

    print(f"Running letterbox inference @{CANVAS}² with {len(models)}-fold ensemble + h-flip TTA"
          + (" (+prob dump)" if args.save_probs else "") + " ...")

    with torch.no_grad():
        for img_path in test_images:
            filename = os.path.basename(img_path).replace("_0000.png", ".png")
            stem = filename[:-4]

            original_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            original_h, original_w = original_img.shape
            A, B = shape_loader({"image": img_path})["image"].shape[1:]
            nA, nB = LetterboxResized.scaled_size(A, B, CANVAS)

            x = test_transform({"image": img_path})["image"].unsqueeze(0).to(device)
            xf = torch.flip(x, dims=[3])

            prob_sum = None
            for model in models:
                for inp, flip_back in ((x, False), (xf, True)):
                    logits = model(inp)
                    if logits.shape[2:] != inp.shape[2:]:
                        logits = F.interpolate(logits, size=inp.shape[2:], mode="bilinear", align_corners=False)
                    prob = torch.softmax(logits, dim=1)
                    if flip_back:
                        prob = torch.flip(prob, dims=[3])
                    prob_sum = prob if prob_sum is None else prob_sum + prob

            prob_sum = prob_sum / (2 * len(models))            # mean softmax, (1,6,CANVAS,CANVAS)
            prob_np = prob_sum.squeeze(0).cpu().numpy()         # (6, CANVAS, CANVAS)

            # invert letterbox: crop content, undo (W,H) transpose, resize to native
            prob_np = prob_np[:, :nA, :nB]                      # (6, nA, nB)
            prob_np = np.transpose(prob_np, (0, 2, 1))          # per-channel .T -> (6, nB, nA)
            if prob_np.shape[1:] != (original_h, original_w):
                prob_np = np.stack([
                    cv2.resize(prob_np[c], (original_w, original_h), interpolation=cv2.INTER_LINEAR)
                    for c in range(NUM_CLASSES)
                ], axis=0)

            prediction = np.argmax(prob_np, axis=0).astype(np.uint8)
            cv2.imwrite(os.path.join(output_dir, filename), prediction)
            if args.save_probs:
                np.savez_compressed(os.path.join(prob_dir, stem + ".npz"),
                                    prob=prob_np.astype(np.float16))

    print(f"\nDone! Label maps -> {output_dir}"
          + (f"\n      Prob volumes -> {prob_dir}" if args.save_probs else ""))


if __name__ == "__main__":
    main()
