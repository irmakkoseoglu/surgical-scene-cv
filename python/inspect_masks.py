"""Sanity check: list the grey values found in the watershed masks and how often they occur.

    python python/inspect_masks.py /content/cholecseg8k --limit 300
Any value printed as UNKNOWN is not in WATERSHED_VALUES and will be ignored during training.
"""
import argparse
from collections import Counter

import cv2
import numpy as np

from cholecseg8k import CLASSES, WATERSHED_VALUES, find_samples

p = argparse.ArgumentParser()
p.add_argument("root")
p.add_argument("--limit", type=int, default=300)
args = p.parse_args()

samples = find_samples(args.root)
step = max(1, len(samples) // args.limit)
counts: Counter = Counter()
for s in samples[::step]:
    m = cv2.imread(str(s.mask), cv2.IMREAD_UNCHANGED)
    m = m[..., 0] if m.ndim == 3 else m
    v, c = np.unique(m, return_counts=True)
    counts.update(dict(zip(v.tolist(), c.tolist())))

total = sum(counts.values())
print(f"{len(samples)} frames, {len({s.video for s in samples})} videos; checked every {step}th frame")
for v, c in sorted(counts.items()):
    name = CLASSES[WATERSHED_VALUES[v]] if v in WATERSHED_VALUES else "UNKNOWN"
    print(f"grey {v:3d}: {100 * c / total:6.2f}%  {name}")
