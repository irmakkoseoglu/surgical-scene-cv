"""Per-class evaluation on the held-out videos + qualitative overlays.

    python python/evaluate.py --data /content/cholecseg8k --ckpt runs/unet_r34/best.pt
Writes results/metrics.md, results/metrics.json and results/samples/*.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from cholecseg8k import CLASSES, NUM_CLASSES, CholecSeg8k, colorize, find_samples, split_by_video
from metrics import ConfusionMatrix
from model import ResNetUNet


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", default="results")
    p.add_argument("--num-samples", type=int, default=6)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model = ResNetUNet(NUM_CLASSES, ckpt["encoder"], pretrained=False)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    _, val_s = split_by_video(find_samples(args.data), ckpt["val_videos"])
    ds = CholecSeg8k(val_s, tuple(ckpt["size"]))
    cm = ConfusionMatrix(NUM_CLASSES)
    with torch.no_grad():
        for x, y in DataLoader(ds, batch_size=16):
            cm.update(model(x.to(device)).argmax(1), y)
    s = cm.scores()

    out = Path(args.out)
    (out / "samples").mkdir(parents=True, exist_ok=True)
    lines = [f"Held-out videos: {', '.join(ckpt['val_videos'])} ({len(ds)} frames)\n",
             "| Class | IoU | Dice |", "|---|---|---|"]
    for i, name in enumerate(CLASSES):
        if s["present"][i]:
            lines.append(f"| {name} | {s['iou'][i]:.3f} | {s['dice'][i]:.3f} |")
        else:
            lines.append(f"| {name} | – | – |")
    lines.append(f"| **Mean** | **{s['miou']:.3f}** | **{s['mdice']:.3f}** |")
    (out / "metrics.md").write_text("\n".join(lines) + "\n")
    (out / "metrics.json").write_text(json.dumps(
        {"miou": s["miou"], "mdice": s["mdice"], "pixel_acc": s["pixel_acc"],
         "per_class_iou": {c: (None if np.isnan(v) else float(v)) for c, v in zip(CLASSES, s["iou"])}}, indent=2))
    print("\n".join(lines))

    # image | ground truth | prediction
    idx = np.linspace(0, len(ds) - 1, args.num_samples).astype(int)
    for k, i in enumerate(idx):
        img, gt = ds.load(i)
        x, _ = ds[i]
        with torch.no_grad():
            pred = model(x[None].to(device)).argmax(1)[0].cpu().numpy().astype(np.uint8)
        blend = lambda m: cv2.addWeighted(img, 0.5, colorize(m), 0.5, 0)
        row = np.concatenate([img, blend(gt), blend(pred)], axis=1)
        cv2.imwrite(str(out / "samples" / f"sample_{k}.png"), cv2.cvtColor(row, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    main()
