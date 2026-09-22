"""CholecSeg8k dataset utilities.

CholecSeg8k (Hong et al., 2020) contains 8,080 annotated frames from 17 laparoscopic
cholecystectomy videos of the Cholec80 dataset, labelled with 13 classes.

Expected folder layout (as downloaded from Kaggle):

    <root>/video01/video01_00080/frame_80_endo.png
    <root>/video01/video01_00080/frame_80_endo_watershed_mask.png
    ...

The *watershed* masks are used as ground truth because the colour masks contain
anti-aliasing artefacts at class borders. Run `python inspect_masks.py <root>` once to
verify that the grey values in your copy match WATERSHED_VALUES below.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

CLASSES = [
    "Background",
    "Abdominal Wall",
    "Liver",
    "Gastrointestinal Tract",
    "Fat",
    "Grasper",
    "Connective Tissue",
    "Blood",
    "Cystic Duct",
    "L-hook Electrocautery",
    "Gallbladder",
    "Hepatic Vein",
    "Liver Ligament",
]
NUM_CLASSES = len(CLASSES)
IGNORE_INDEX = 255

# Grey value in the watershed mask -> class index
WATERSHED_VALUES = {
    50: 0,   # Background
    11: 1,   # Abdominal Wall
    21: 2,   # Liver
    13: 3,   # Gastrointestinal Tract
    12: 4,   # Fat
    31: 5,   # Grasper
    23: 6,   # Connective Tissue
    24: 7,   # Blood
    25: 8,   # Cystic Duct
    32: 9,   # L-hook Electrocautery
    22: 10,  # Gallbladder
    33: 11,  # Hepatic Vein
    5: 12,   # Liver Ligament
}

# RGB colours used for visualisation (same palette as the dataset's colour masks)
COLORS_RGB = [
    (127, 127, 127), (210, 140, 140), (255, 114, 114), (231, 70, 156),
    (186, 183, 75), (170, 255, 0), (255, 85, 0), (255, 0, 0),
    (255, 255, 0), (169, 255, 184), (255, 160, 165), (0, 50, 128),
    (111, 74, 0),
]

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_LUT = np.full(256, IGNORE_INDEX, dtype=np.uint8)
for grey, idx in WATERSHED_VALUES.items():
    _LUT[grey] = idx


@dataclass(frozen=True)
class Sample:
    image: Path
    mask: Path
    video: str


def find_samples(root: str | Path) -> list[Sample]:
    root = Path(root)
    samples = []
    for img in sorted(root.rglob("*_endo.png")):
        mask = img.with_name(img.name.replace("_endo.png", "_endo_watershed_mask.png"))
        if not mask.exists():
            continue
        m = re.search(r"video\d+", str(img.relative_to(root)))
        samples.append(Sample(img, mask, m.group(0) if m else "unknown"))
    if not samples:
        raise FileNotFoundError(f"No '*_endo.png' frames with watershed masks found under {root}")
    return samples


def split_by_video(samples: list[Sample], val_videos: list[str]) -> tuple[list[Sample], list[Sample]]:
    """Split on the *video* (= patient) level.

    Consecutive frames of one video are almost identical. A random frame-level split
    would put near-duplicates into train and validation and inflate the scores
    (data leakage), so whole videos are held out instead.
    """
    val_set = set(val_videos)
    train = [s for s in samples if s.video not in val_set]
    val = [s for s in samples if s.video in val_set]
    if not train or not val:
        raise ValueError(f"Empty split. Videos available: {sorted({s.video for s in samples})}")
    return train, val


def encode_mask(mask_bgr: np.ndarray) -> np.ndarray:
    """Watershed mask (H, W, 3) -> class-index mask (H, W) uint8."""
    grey = mask_bgr[..., 0] if mask_bgr.ndim == 3 else mask_bgr
    return _LUT[grey]


def colorize(label: np.ndarray) -> np.ndarray:
    """Class-index mask -> RGB image."""
    palette = np.zeros((256, 3), dtype=np.uint8)
    palette[:NUM_CLASSES] = COLORS_RGB
    return palette[label]


class CholecSeg8k(Dataset):
    def __init__(self, samples: list[Sample], size: tuple[int, int] = (256, 448), train: bool = False):
        self.samples = samples
        self.size = size  # (H, W), both divisible by 32
        self.train = train

    def __len__(self) -> int:
        return len(self.samples)

    def load(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        s = self.samples[i]
        img = cv2.cvtColor(cv2.imread(str(s.image), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        mask = encode_mask(cv2.imread(str(s.mask), cv2.IMREAD_UNCHANGED))
        h, w = self.size
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        return img, mask

    def augment(self, img: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if random.random() < 0.5:
            img, mask = img[:, ::-1], mask[:, ::-1]
        if random.random() < 0.8:  # brightness / contrast jitter (endoscope light varies)
            alpha = random.uniform(0.75, 1.25)
            beta = random.uniform(-25, 25)
            img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
        if random.random() < 0.3:
            img = cv2.GaussianBlur(img, (5, 5), 0)
        return np.ascontiguousarray(img), np.ascontiguousarray(mask)

    def __getitem__(self, i: int):
        img, mask = self.load(i)
        if self.train:
            img, mask = self.augment(img, mask)
        x = (img.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        return torch.from_numpy(x.transpose(2, 0, 1)), torch.from_numpy(mask.astype(np.int64))
