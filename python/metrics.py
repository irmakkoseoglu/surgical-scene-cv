from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConfusionMatrix:
    def __init__(self, num_classes: int, ignore_index: int = 255):
        self.n = num_classes
        self.ignore = ignore_index
        self.mat = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        pred, target = pred.flatten().cpu().numpy(), target.flatten().cpu().numpy()
        keep = target != self.ignore
        idx = self.n * target[keep].astype(np.int64) + pred[keep]
        self.mat += np.bincount(idx, minlength=self.n ** 2).reshape(self.n, self.n)

    def scores(self) -> dict:
        tp = np.diag(self.mat).astype(np.float64)
        fp = self.mat.sum(0) - tp
        fn = self.mat.sum(1) - tp
        present = self.mat.sum(1) > 0  # classes that occur in the ground truth
        with np.errstate(divide="ignore", invalid="ignore"):
            iou = tp / (tp + fp + fn)
            dice = 2 * tp / (2 * tp + fp + fn)
        return {
            "iou": iou,
            "dice": dice,
            "present": present,
            "miou": float(np.nanmean(iou[present])),
            "mdice": float(np.nanmean(dice[present])),
            "pixel_acc": float(tp.sum() / max(self.mat.sum(), 1)),
        }


class DiceCELoss(nn.Module):
    """Cross-entropy + soft Dice. Dice counteracts the heavy class imbalance
    (small, clinically important structures like the cystic duct vs. large background)."""

    def __init__(self, num_classes: int, ignore_index: int = 255, dice_weight: float = 1.0):
        super().__init__()
        self.n, self.ignore, self.w = num_classes, ignore_index, dice_weight
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = self.ce(logits, target)
        valid = (target != self.ignore).unsqueeze(1)
        probs = logits.softmax(1) * valid
        onehot = F.one_hot(target.clamp(max=self.n - 1), self.n).permute(0, 3, 1, 2) * valid
        inter = (probs * onehot).sum((0, 2, 3))
        denom = probs.sum((0, 2, 3)) + onehot.sum((0, 2, 3))
        dice = 1 - ((2 * inter + 1) / (denom + 1)).mean()
        return ce + self.w * dice
