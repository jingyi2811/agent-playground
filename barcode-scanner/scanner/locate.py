"""Find the barcode region before decoding.

Decoding a small crop is faster and more reliable than decoding a whole
frame. Uses the classic gradient method: barcodes have strong horizontal
gradients and weak vertical ones, so the difference lights up the label.
"""
from __future__ import annotations

import cv2
import numpy as np

Box = tuple[int, int, int, int]  # x, y, w, h


def find_candidates(gray: np.ndarray, max_regions: int = 3) -> list[Box]:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=-1)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=-1)
    grad = cv2.subtract(cv2.convertScaleAbs(gx), cv2.convertScaleAbs(gy))

    blurred = cv2.blur(grad, (9, 9))
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Close gaps between bars so the label becomes one blob.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    closed = cv2.erode(closed, None, iterations=4)
    closed = cv2.dilate(closed, None, iterations=4)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:max_regions]

    h, w = gray.shape[:2]
    boxes: list[Box] = []
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        if bw * bh < 0.002 * w * h:
            continue
        pad_x, pad_y = int(bw * 0.15), int(bh * 0.4)
        x0, y0 = max(0, x - pad_x), max(0, y - pad_y)
        x1, y1 = min(w, x + bw + pad_x), min(h, y + bh + pad_y)
        boxes.append((x0, y0, x1 - x0, y1 - y0))
    return boxes


def crop(img: np.ndarray, box: Box) -> np.ndarray:
    x, y, w, h = box
    return img[y : y + h, x : x + w]
