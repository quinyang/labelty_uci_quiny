"""
LAYER 1 PROX inference — 5-fold ensemble + h-flip TTA, letterbox-exact inverse.

Clone of test_layer1.py for the 5-class (Mesial+Distal merged -> Proximal) model.
Same critical orientation fix: crop [:nA,:nB], then prediction.T to undo MONAI's
(W,H) load transpose before resizing back (the bug that sank the 12-class scores).

Outputs:
  layer1prox_predictions/<stem>.png       argmax structure maps (0..4)
  layer1prox_probs/<stem>.npz  (optional)  float16 softmax (5,H,W) for Layer 2

Usage:
    python test_layer1prox.py                 # write label maps only
    python test_layer1prox.py --save-probs    # also dump prob volumes for Layer 2
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

NUM_CLASSES = 5
CANVAS = 1024
NUM_FOLDS = 5
SAVE_ROOT = "/home/jiakuny1/Projects"

test_images_dir = "/home/jiakuny1/Projects/nnUNet_data/nnUNet_raw/Dataset101_Dental/imagesTs"
output_dir = "/home/jiakuny1/Projects/layer1prox_predictions"
prob_dir = "/home/jiakuny1/Projects/layer1prox_probs"

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


def build_model(encoder="efficientnet-b2"):
    return smp.Unet(encoder_name=encoder, encoder_weights=None,
                    in_channels=3, classes=NUM_CLASSES, decoder_dropout=0.2).to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save-probs", action="store_true",
                    help="also save (5,H,W) float16 softmax volumes for Layer 2")
    ap.add_argument("--scales", type=float, nargs="+", default=[1.0],
                    help="multi-scale TTA factors applied to the 1024 letterbox input "
                         "(e.g. --scales 0.85 1.0 1.15); default [1.0] = single scale")
    ap.add_argument("--out-dir", default=None,
                    help="prediction output dir (override to compare without clobbering)")
    ap.add_argument("--encoder", default="efficientnet-b2",
                    help="smp encoder_name; must match the trained checkpoints")
    ap.add_argument("--run-tag", default="",
                    help="extra experiment tag, must match train_layer1prox.py --run-tag")
    args = ap.parse_args()

    tag = "" if args.encoder == "efficientnet-b2" else "_" + args.encoder.split("-")[-1]
    if args.run_tag:
        tag += "_" + args.run_tag
    out_dir = args.out_dir or (output_dir + tag)
    os.makedirs(out_dir, exist_ok=True)
    if args.save_probs:
        os.makedirs(prob_dir, exist_ok=True)

    print(f"Testing on device: {device}")
    test_images = sorted(glob.glob(os.path.join(test_images_dir, "*_0000.png")))
    print(f"Found {len(test_images)} test images.")

    fold_paths = [os.path.join(SAVE_ROOT, f"layer1prox{tag}_fold{f}.pth") for f in range(NUM_FOLDS)]
    fold_paths = [p for p in fold_paths if os.path.exists(p)]
    if not fold_paths:
        raise FileNotFoundError(f"No layer1prox{tag}_fold*.pth in {SAVE_ROOT}. Run train_layer1prox.py --encoder {args.encoder} first.")
    print(f"Ensembling {len(fold_paths)} fold checkpoints:")
    models = []
    for p in fold_paths:
        print(f"  {p}")
        m = build_model(args.encoder)
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
          f" + scales {args.scales}"
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

            prob_sum, n_aug = None, 0
            for model in models:
                for inp, flip_back in ((x, False), (xf, True)):
                    for s in args.scales:
                        inp_s = inp if s == 1.0 else F.interpolate(
                            inp, scale_factor=s, mode="bilinear", align_corners=False)
                        logits = model(inp_s)
                        # bring every scale's logits back to the canvas frame before averaging
                        if logits.shape[2:] != x.shape[2:]:
                            logits = F.interpolate(logits, size=x.shape[2:], mode="bilinear", align_corners=False)
                        prob = torch.softmax(logits, dim=1)
                        if flip_back:
                            prob = torch.flip(prob, dims=[3])
                        prob_sum = prob if prob_sum is None else prob_sum + prob
                        n_aug += 1

            prob_sum = prob_sum / n_aug
            prob_np = prob_sum.squeeze(0).cpu().numpy()         # (5, CANVAS, CANVAS)

            prob_np = prob_np[:, :nA, :nB]                      # (5, nA, nB)
            prob_np = np.transpose(prob_np, (0, 2, 1))          # per-channel .T -> (5, nB, nA)
            if prob_np.shape[1:] != (original_h, original_w):
                prob_np = np.stack([
                    cv2.resize(prob_np[c], (original_w, original_h), interpolation=cv2.INTER_LINEAR)
                    for c in range(NUM_CLASSES)
                ], axis=0)

            prediction = np.argmax(prob_np, axis=0).astype(np.uint8)
            cv2.imwrite(os.path.join(out_dir, filename), prediction)
            if args.save_probs:
                np.savez_compressed(os.path.join(prob_dir, stem + ".npz"),
                                    prob=prob_np.astype(np.float16))

    print(f"\nDone! Label maps -> {out_dir}"
          + (f"\n      Prob volumes -> {prob_dir}" if args.save_probs else ""))


if __name__ == "__main__":
    main()
