"""Export the trained model to ONNX for the C++ / OpenCV DNN runtime and check parity.

    python python/export_onnx.py --ckpt runs/unet_r34/best.pt --out models/unet.onnx
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from cholecseg8k import NUM_CLASSES
from model import DeployWrapper, ResNetUNet


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", default="models/unet.onnx")
    p.add_argument("--opset", type=int, default=13)
    args = p.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    model = ResNetUNet(NUM_CLASSES, ckpt["encoder"], pretrained=False)
    model.load_state_dict(ckpt["model"])
    net = DeployWrapper(model).eval()

    h, w = ckpt["size"]
    dummy = torch.rand(1, 3, h, w)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(net, dummy, args.out, input_names=["image"], output_names=["logits"],
                      opset_version=args.opset, dynamo=False)

    # Parity check: PyTorch vs. OpenCV DNN (the runtime used by the C++ app)
    with torch.no_grad():
        ref = net(dummy).numpy()
    cvnet = cv2.dnn.readNetFromONNX(args.out)
    cvnet.setInput(dummy.numpy())
    got = cvnet.forward()
    diff = float(np.abs(ref - got).max())
    agree = float((ref.argmax(1) == got.argmax(1)).mean())
    print(f"exported {args.out} | input 1x3x{h}x{w} | max |diff| {diff:.2e} | argmax agreement {agree:.4%}")
    if agree < 0.999:
        raise SystemExit("OpenCV DNN output does not match PyTorch")


if __name__ == "__main__":
    main()
