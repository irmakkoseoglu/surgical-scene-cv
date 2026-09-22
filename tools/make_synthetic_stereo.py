"""Generate synthetic stereo data with *known ground truth* to test the C++ stereo tools.

1. calib/left, calib/right : chessboard pairs seen by a virtual stereo endoscope
                             (f = 400 px, baseline = 5 mm, 640x480)
2. scene/left.png, right.png: rectified pair of a textured scene with two depth planes
                             (background 80 mm, instrument-like box 50 mm)

    python tools/make_synthetic_stereo.py --out data/synthetic
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

W, H, F, BASELINE = 640, 480, 400.0, 5.0
K = np.array([[F, 0, W / 2], [0, F, H / 2], [0, 0, 1]])


def rodrigues(rx, ry, rz):
    return cv2.Rodrigues(np.array([rx, ry, rz], dtype=np.float64))[0]


def board_texture(cols=10, rows=7, px=60, margin=60):
    """Chessboard with cols x rows squares (-> (cols-1) x (rows-1) inner corners) + white margin."""
    img = np.full((rows * px + 2 * margin, cols * px + 2 * margin), 255, np.uint8)
    for r in range(rows):
        for c in range(cols):
            if (r + c) % 2 == 0:
                img[margin + r * px: margin + (r + 1) * px, margin + c * px: margin + (c + 1) * px] = 0
    return img, px, margin


def render(tex, px, margin, square_mm, R, t):
    """Project the planar board (Z=0, units mm) into a camera with pose (R, t)."""
    # texture pixel -> board mm; the first inner corner is the board origin
    S = np.array([[square_mm / px, 0, -(margin + px) * square_mm / px],
                  [0, square_mm / px, -(margin + px) * square_mm / px], [0, 0, 1]])
    Hm = K @ np.column_stack([R[:, 0], R[:, 1], t]) @ S
    out = cv2.warpPerspective(tex, Hm, (W, H), flags=cv2.INTER_LINEAR, borderValue=128)
    noise = np.random.normal(0, 2, out.shape)
    return np.clip(out + noise, 0, 255).astype(np.uint8)


def make_calibration(out: Path, n: int, square_mm: float):
    tex, px, margin = board_texture()
    (out / "calib/left").mkdir(parents=True, exist_ok=True)
    (out / "calib/right").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    # right camera sits BASELINE mm to the right of the left camera
    R_lr, T_lr = np.eye(3), np.array([-BASELINE, 0.0, 0.0])
    for i in range(n):
        R = rodrigues(*rng.uniform([-0.35, -0.35, -0.3], [0.35, 0.35, 0.3]))
        centre = np.array([4 * square_mm, 2.5 * square_mm, 0])
        t = np.array([rng.uniform(-6, 6), rng.uniform(-5, 5), rng.uniform(45, 70)]) - R @ centre
        cv2.imwrite(str(out / f"calib/left/{i:02d}.png"), render(tex, px, margin, square_mm, R, t))
        cv2.imwrite(str(out / f"calib/right/{i:02d}.png"), render(tex, px, margin, square_mm, R_lr @ R, R_lr @ t + T_lr))


def make_scene(out: Path):
    rng = np.random.default_rng(1)
    def texture():
        t = rng.random((H, W + 200, 3)).astype(np.float32)
        t = cv2.GaussianBlur(t, (0, 0), 2.0)
        t = (t - t.min()) / (t.max() - t.min())
        return (t * np.array([90, 110, 220]) + 20).astype(np.uint8)  # reddish, tissue-like
    bg, fg = texture(), (texture() * 0.4 + 120).astype(np.uint8)       # fg: greyish "instrument"

    z_bg, z_fg = 80.0, 50.0
    d_bg, d_fg = F * BASELINE / z_bg, F * BASELINE / z_fg  # 25 px, 40 px
    x0, x1, y0, y1 = 250, 420, 150, 330

    def view(shift_bg, shift_fg):
        img = bg[:, int(shift_bg): int(shift_bg) + W].copy()
        img[y0:y1, x0 - int(shift_fg - 100): x1 - int(shift_fg - 100)] = fg[y0:y1, x0:x1]
        return img

    # left camera: reference; right camera sees everything shifted left by its disparity
    left = view(100, 100)
    right = view(100 + d_bg, 100 + d_fg)
    gt = np.full((H, W), z_bg, np.float32)
    gt[y0:y1, x0:x1] = z_fg
    (out / "scene").mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / "scene/left.png"), left)
    cv2.imwrite(str(out / "scene/right.png"), right)
    np.save(out / "scene/gt_depth_mm.npy", gt)
    return {"z_background_mm": z_bg, "z_foreground_mm": z_fg, "foreground_box_xyxy": [x0, y0, x1, y1]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--pairs", type=int, default=15)
    ap.add_argument("--square", type=float, default=3.0)
    a = ap.parse_args()
    out = Path(a.out)
    make_calibration(out, a.pairs, a.square)
    scene = make_scene(out)
    gt = {"focal_px": F, "baseline_mm": BASELINE, "image_size": [W, H], "square_mm": a.square,
          "inner_corners": [9, 6], **scene}
    (out / "ground_truth.json").write_text(json.dumps(gt, indent=2))
    print(json.dumps(gt, indent=2))
