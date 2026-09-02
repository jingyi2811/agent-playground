"""Image preprocessing steps that improve barcode read rate.

Each step is a pure function: ndarray in, ndarray out. They are composed
by name in `PIPELINES` so the evaluator can measure what each one buys.
"""
from __future__ import annotations

import cv2
import numpy as np


def to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def denoise(gray: np.ndarray) -> np.ndarray:
    """Light Gaussian blur removes sensor noise without merging bars."""
    return cv2.GaussianBlur(gray, (3, 3), 0)


def sharpen(gray: np.ndarray) -> np.ndarray:
    """Unsharp mask. Helps slightly out-of-focus frames."""
    blur = cv2.GaussianBlur(gray, (0, 0), 3)
    return cv2.addWeighted(gray, 1.5, blur, -0.5, 0)


def equalize(gray: np.ndarray) -> np.ndarray:
    """CLAHE lifts contrast in dark or unevenly lit images."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def threshold(gray: np.ndarray) -> np.ndarray:
    """Adaptive threshold handles glare and shadows across the label."""
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 10
    )


def upscale(gray: np.ndarray, min_width: int = 800) -> np.ndarray:
    """Small crops decode badly. Upscale so bars are several pixels wide."""
    h, w = gray.shape[:2]
    if w >= min_width:
        return gray
    scale = min_width / w
    return cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def deskew(gray: np.ndarray) -> np.ndarray:
    """Rotate so the dominant edge direction is horizontal.

    Barcodes are made of parallel bars, so the strongest gradient
    orientation tells us how far the label is rotated.
    """
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mask = mag > np.percentile(mag, 95)
    if mask.sum() < 50:
        return gray
    angles = np.arctan2(gy[mask], gx[mask])
    # Bars produce gradients perpendicular to their length; fold to [-90, 90).
    deg = (np.degrees(angles) + 90) % 180 - 90
    hist, edges = np.histogram(deg, bins=180, range=(-90, 90))
    angle = float(edges[np.argmax(hist)])
    if abs(angle) < 1.0:
        return gray
    h, w = gray.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(gray, m, (w, h), flags=cv2.INTER_CUBIC, borderValue=255)


STEPS = {
    "gray": to_gray,
    "denoise": denoise,
    "sharpen": sharpen,
    "equalize": equalize,
    "threshold": threshold,
    "upscale": upscale,
    "deskew": deskew,
}

# Named pipelines the evaluator compares. Order matters.
PIPELINES: dict[str, list[str]] = {
    "raw": ["gray"],
    "basic": ["gray", "denoise"],
    "contrast": ["gray", "denoise", "equalize"],
    "full": ["gray", "upscale", "denoise", "equalize", "deskew", "sharpen"],
    "binary": ["gray", "upscale", "denoise", "equalize", "deskew", "threshold"],
}


def run_pipeline(img: np.ndarray, name: str) -> np.ndarray:
    out = img
    for step in PIPELINES[name]:
        out = STEPS[step](out)
    return out
