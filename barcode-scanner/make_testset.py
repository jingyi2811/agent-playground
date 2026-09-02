"""Generate a synthetic labelled test set of barcodes under bad conditions.

Renders EAN-13 and Code128 barcodes, then applies blur, rotation, noise,
low contrast, and glare so the evaluator has something to measure.
Replace or extend with real photos when you have them.

    python make_testset.py --count 60
"""
from __future__ import annotations

import argparse
import csv
import os
import random

import cv2
import numpy as np
import zxingcpp

from scanner.validate import gs1_check_digit


def random_ean13(rng: random.Random) -> str:
    body = "".join(rng.choice("0123456789") for _ in range(12))
    return body + str(gs1_check_digit(body))


def render(text: str, fmt: zxingcpp.BarcodeFormat, rng: random.Random) -> np.ndarray:
    bits = zxingcpp.write_barcode(fmt, text, width=400, height=160, quiet_zone=10)
    gray = np.array(bits, dtype=np.uint8)
    # Place on a larger "label" with a margin so rotation does not clip.
    canvas = np.full((gray.shape[0] + 200, gray.shape[1] + 200), 235, np.uint8)
    y, x = 100, 100
    canvas[y : y + gray.shape[0], x : x + gray.shape[1]] = gray
    return cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)


def degrade(img: np.ndarray, rng: random.Random, severity: float = 1.0) -> tuple[np.ndarray, str]:
    """Apply random degradations. severity 1.0 is moderate, 2.0 is harsh."""
    tags = []
    p = min(0.95, 0.5 * severity)
    if rng.random() < p:
        angle = rng.uniform(-25, 25) * min(severity, 1.6)
        h, w = img.shape[:2]
        m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        img = cv2.warpAffine(img, m, (w, h), borderValue=(235, 235, 235))
        tags.append(f"rot{angle:+.0f}")
    if rng.random() < p:
        k = rng.choice([3, 5, 7] if severity < 1.5 else [5, 7, 9, 11])
        img = cv2.GaussianBlur(img, (k, k), 0)
        tags.append(f"blur{k}")
    if rng.random() < 0.4 * severity:
        alpha = rng.uniform(0.3, 0.6) / severity
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=rng.uniform(40, 90))
        tags.append("lowcontrast")
    if rng.random() < 0.4 * severity:
        noise = np.random.default_rng(rng.randrange(1 << 30)).normal(0, rng.uniform(5, 20) * severity, img.shape)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        tags.append("noise")
    if rng.random() < 0.3 * severity:
        h, w = img.shape[:2]
        glare = np.zeros_like(img, dtype=np.float32)
        cx, cy = rng.randint(0, w), rng.randint(0, h)
        cv2.circle(glare, (cx, cy), rng.randint(60, 140), (255, 255, 255), -1)
        glare = cv2.GaussianBlur(glare, (0, 0), 40)
        img = np.clip(img.astype(np.float32) + glare * 0.7, 0, 255).astype(np.uint8)
        tags.append("glare")
    if rng.random() < 0.3 * severity:
        scale = rng.uniform(0.35, 0.6) / severity
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        tags.append("small")
    return img, "-".join(tags) or "clean"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--count", type=int, default=60)
    p.add_argument("--out", default="data")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--severity", type=float, default=1.0, help="1.0 moderate, 2.0 harsh")
    a = p.parse_args()

    rng = random.Random(a.seed)
    img_dir = os.path.join(a.out, "images")
    os.makedirs(img_dir, exist_ok=True)

    rows = []
    for i in range(a.count):
        if rng.random() < 0.7:
            text, fmt = random_ean13(rng), zxingcpp.BarcodeFormat.EAN13
        else:
            text, fmt = f"LOT-{rng.randint(1000, 99999)}", zxingcpp.BarcodeFormat.Code128
        img, tag = degrade(render(text, fmt, rng), rng, a.severity)
        fname = f"{i:03d}_{tag}.png"
        cv2.imwrite(os.path.join(img_dir, fname), img)
        rows.append({"filename": fname, "expected": text})

    with open(os.path.join(a.out, "labels.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["filename", "expected"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} images to {img_dir} and labels to {a.out}/labels.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
