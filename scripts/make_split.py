"""Create a deterministic, family-stratified train/val split from train.csv.

Validation is held out purely for offline scoring with src/grading.py; we never
train on it. Split is seeded for reproducibility.
"""

import csv
import json
import os
import random
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from families import classify  # noqa: E402

SEED = 1234
VAL_PER_FAMILY = 100  # held-out eval problems per family (600 total)


def main():
    here = os.path.join(os.path.dirname(__file__), "..")
    rows = list(csv.DictReader(open(os.path.join(here, "data/train.csv"))))
    by_fam = defaultdict(list)
    for r in rows:
        r["family"] = classify(r["prompt"])
        by_fam[r["family"]].append(r)

    rng = random.Random(SEED)
    train, val = [], []
    for fam, items in by_fam.items():
        idx = list(range(len(items)))
        rng.shuffle(idx)
        val_idx = set(idx[:VAL_PER_FAMILY])
        for i, r in enumerate(items):
            (val if i in val_idx else train).append(r)

    outdir = os.path.join(here, "data/splits")
    os.makedirs(outdir, exist_ok=True)
    for name, part in [("train", train), ("val", val)]:
        with open(os.path.join(outdir, f"{name}.jsonl"), "w") as f:
            for r in part:
                f.write(json.dumps(r) + "\n")
    counts = {f: len(v) for f, v in by_fam.items()}
    print(f"families: {counts}")
    print(f"train={len(train)}  val={len(val)}  (val {VAL_PER_FAMILY}/family)")


if __name__ == "__main__":
    main()
