#!/usr/bin/env bash
# End-to-end smoke test on synthetic data: train -> evaluate -> export ONNX -> C++ inference,
# plus stereo calibration / depth against known ground truth. Used by CI.
set -euo pipefail
cd "$(dirname "$0")/.."
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

echo "== build C++ tools"
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release > /dev/null
cmake --build cpp/build -j"$(nproc)" > /dev/null

echo "== segmentation pipeline"
python tools/make_dummy_cholecseg8k.py --out "$TMP/data" --frames 6
python python/inspect_masks.py "$TMP/data" | tail -4
python python/train.py --data "$TMP/data" --out "$TMP/run" --encoder resnet18 --no-pretrained \
  --epochs 2 --batch 4 --height 128 --width 224 --val-videos video12 --workers 0
python python/evaluate.py --data "$TMP/data" --ckpt "$TMP/run/best.pt" --out "$TMP/results" --num-samples 2 > /dev/null
python python/export_onnx.py --ckpt "$TMP/run/best.pt" --out "$TMP/unet.onnx"
FRAME=$(find "$TMP/data/video12" -name '*_endo.png' | sort | head -1)
./cpp/build/segment --model "$TMP/unet.onnx" --input "$FRAME" --output "$TMP/overlay.png" --width 224 --height 128
./cpp/build/segment --model "$TMP/unet.onnx" --input "$(dirname "$FRAME")" --output "$TMP/overlays" --width 224 --height 128
test -f "$TMP/overlay.png"

echo "== stereo calibration + depth"
python tools/make_synthetic_stereo.py --out "$TMP/stereo" > /dev/null
./cpp/build/stereo_calibrate --left "$TMP/stereo/calib/left" --right "$TMP/stereo/calib/right" \
  --cols 9 --rows 6 --square 3 --out "$TMP/calib.yml" | tail -3
./cpp/build/stereo_depth --left "$TMP/stereo/scene/left.png" --right "$TMP/stereo/scene/right.png" \
  --focal 400 --baseline 5 --out "$TMP/depth" --seg-model "$TMP/unet.onnx" --seg-width 224 --seg-height 128
python - "$TMP" <<'EOF'
import sys, cv2, numpy as np, re
tmp = sys.argv[1]
calib = cv2.FileStorage(f"{tmp}/calib.yml", cv2.FILE_STORAGE_READ)
baseline = float(np.linalg.norm(calib.getNode("T").mat()))
d = cv2.imread(f"{tmp}/depth/depth_mm.png", cv2.IMREAD_UNCHANGED).astype(float)
gt = np.load(f"{tmp}/stereo/scene/gt_depth_mm.npy")
v = d > 0
mae = np.abs(d[v] - gt[v]).mean()
print(f"baseline {baseline:.3f} mm (gt 5.000) | depth MAE {mae:.2f} mm on {v.mean():.0%} of pixels")
assert abs(baseline - 5.0) < 0.1 and mae < 2.0
EOF
echo "== all smoke tests passed"
