"""Create a tiny fake dataset with the exact CholecSeg8k folder layout and mask encoding.

Only meant for smoke tests / CI (does the pipeline run end to end?), not for real training.

    python tools/make_dummy_cholecseg8k.py --out data/dummy
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from cholecseg8k import COLORS_RGB, WATERSHED_VALUES  # noqa: E402

GREY_OF = {c: g for g, c in WATERSHED_VALUES.items()}


def frame(rng):
    h, w = 480, 854
    label = np.full((h, w), 2, np.uint8)  # liver everywhere
    cv2.circle(label, (int(rng.uniform(200, 650)), int(rng.uniform(150, 330))), int(rng.uniform(60, 120)), 10, -1)
    x = int(rng.uniform(300, 600))
    cv2.line(label, (x, 0), (x + int(rng.uniform(-150, 150)), 300), 5, 30)  # grasper shaft
    label[:, :60] = 0
    label[:, -60:] = 0  # black endoscope border

    img = np.array(COLORS_RGB, np.uint8)[label].astype(np.float32)
    img += rng.normal(0, 12, img.shape)
    img = cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    grey = np.vectorize(GREY_OF.get)(label).astype(np.uint8)
    return img, cv2.merge([grey, grey, grey])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/dummy")
    ap.add_argument("--videos", nargs="+", default=["video01", "video09", "video12"])
    ap.add_argument("--frames", type=int, default=8)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    for v in a.videos:
        for k in range(a.frames):
            n = 80 + k
            d = Path(a.out) / v / f"{v}_{n:05d}"
            d.mkdir(parents=True, exist_ok=True)
            img, mask = frame(rng)
            cv2.imwrite(str(d / f"frame_{n}_endo.png"), img)
            cv2.imwrite(str(d / f"frame_{n}_endo_watershed_mask.png"), mask)
    print(f"wrote {len(a.videos) * a.frames} frames to {a.out}")
