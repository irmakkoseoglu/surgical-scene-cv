"""Train a ResNet-U-Net on CholecSeg8k.

Example (GPU, e.g. Google Colab):
    python python/train.py --data /content/cholecseg8k --epochs 30 --batch 16 --out runs/unet_r34
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from cholecseg8k import CLASSES, NUM_CLASSES, CholecSeg8k, find_samples, split_by_video
from metrics import ConfusionMatrix, DiceCELoss
from model import ResNetUNet

DEFAULT_VAL_VIDEOS = ["video12", "video20", "video48", "video55"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="CholecSeg8k root folder")
    p.add_argument("--out", default="runs/unet")
    p.add_argument("--encoder", default="resnet34", choices=["resnet18", "resnet34"])
    p.add_argument("--no-pretrained", action="store_true", help="do not download ImageNet weights")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--width", type=int, default=448)
    p.add_argument("--val-videos", nargs="+", default=DEFAULT_VAL_VIDEOS)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=0, help="limit steps per epoch (smoke tests)")
    return p.parse_args()


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    cm = ConfusionMatrix(NUM_CLASSES)
    for x, y in loader:
        cm.update(model(x.to(device)).argmax(1), y)
    return cm.scores()


def main() -> None:
    args = parse_args()
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    samples = find_samples(args.data)
    train_s, val_s = split_by_video(samples, args.val_videos)
    print(f"{len(train_s)} train / {len(val_s)} val frames | val videos: {args.val_videos} | device: {device}")

    size = (args.height, args.width)
    train_dl = DataLoader(CholecSeg8k(train_s, size, train=True), args.batch, shuffle=True,
                          num_workers=args.workers, pin_memory=device.type == "cuda", drop_last=True)
    val_dl = DataLoader(CholecSeg8k(val_s, size), args.batch, num_workers=args.workers)

    model = ResNetUNet(NUM_CLASSES, args.encoder, pretrained=not args.no_pretrained).to(device)
    loss_fn = DiceCELoss(NUM_CLASSES)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    best, history = -1.0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, total, n = time.time(), 0.0, 0
        for step, (x, y) in enumerate(train_dl):
            if args.max_steps and step >= args.max_steps:
                break
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                loss = loss_fn(model(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += loss.item()
            n += 1
        sched.step()

        s = evaluate(model, val_dl, device)
        row = {"epoch": epoch, "loss": total / max(n, 1), "val_miou": s["miou"], "val_mdice": s["mdice"]}
        history.append(row)
        print(f"epoch {epoch:3d} | loss {row['loss']:.4f} | val mIoU {s['miou']:.4f} | "
              f"val mDice {s['mdice']:.4f} | {time.time() - t0:.0f}s")
        if s["miou"] > best:
            best = s["miou"]
            torch.save({"model": model.state_dict(), "encoder": args.encoder, "size": size,
                        "classes": CLASSES, "val_videos": args.val_videos}, out / "best.pt")

    (out / "history.json").write_text(json.dumps(history, indent=2))
    print(f"best val mIoU: {best:.4f} -> {out / 'best.pt'}")


if __name__ == "__main__":
    main()
