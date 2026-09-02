"""Measure read rate of each preprocessing pipeline over a labelled image set.

Ground truth is `data/labels.csv` with columns: filename,expected.
Prints a table of read rate and mean latency per pipeline, so you can say
exactly what each preprocessing step bought you.

    python evaluate.py
    python evaluate.py --images data/images --labels data/labels.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import time

import cv2

from scanner import decode
from scanner.validate import normalize_gtin
from scanner.preprocess import PIPELINES


def load_labels(path: str) -> dict[str, str]:
    with open(path, newline="") as f:
        return {row["filename"]: row["expected"] for row in csv.DictReader(f)}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--images", default="data/images")
    p.add_argument("--labels", default="data/labels.csv")
    p.add_argument("--pipelines", nargs="*", default=list(PIPELINES))
    a = p.parse_args()

    labels = load_labels(a.labels)
    if not labels:
        print("no labels found; run make_testset.py first")
        return 1

    images = {name: cv2.imread(os.path.join(a.images, name)) for name in labels}
    missing = [n for n, im in images.items() if im is None]
    if missing:
        print(f"warning: {len(missing)} labelled images missing: {missing[:5]}")

    print(f"{'pipeline':<10} {'read':>5} {'total':>6} {'rate':>7} {'ms/img':>8}")
    failures: dict[str, list[str]] = {}
    for name in a.pipelines:
        hits, elapsed = 0, 0.0
        failures[name] = []
        for fname, expected in labels.items():
            img = images[fname]
            if img is None:
                continue
            t0 = time.perf_counter()
            got = {normalize_gtin(r.text) for r in decode.decode(img, name)}
            elapsed += time.perf_counter() - t0
            if normalize_gtin(expected) in got:
                hits += 1
            else:
                failures[name].append(fname)
        n = len(labels) - len(missing)
        print(f"{name:<10} {hits:>5} {n:>6} {hits / n:>7.1%} {1000 * elapsed / n:>8.1f}")

    best = max(a.pipelines, key=lambda k: -len(failures[k]))
    if failures[best]:
        print(f"\nstill failing with '{best}': {failures[best]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
