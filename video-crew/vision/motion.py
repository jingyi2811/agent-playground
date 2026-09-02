"""Object detection by background subtraction, and tracking across a scene."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

NAMED = {  # BGR reference colours, matched by nearest distance
    "red": (40, 40, 220), "green": (60, 180, 60), "blue": (220, 90, 40),
    "yellow": (40, 210, 230), "white": (240, 240, 240), "black": (20, 20, 20),
    "grey": (128, 128, 128), "navy": (90, 40, 20), "cream": (200, 230, 240), "olive": (40, 110, 110),
}


@dataclass
class Obj:
    shape: str
    color: str
    centroid: tuple[int, int]
    area: float


def nearest_color(bgr) -> str:
    b, g, r = (int(x) for x in bgr)
    return min(NAMED, key=lambda k: sum((int(a) - int(c)) ** 2 for a, c in zip(NAMED[k], (b, g, r))))


def background_color(frame: np.ndarray) -> tuple:
    """Most common colour, from a coarse quantisation of the pixels."""
    small = cv2.resize(frame, (64, 48), interpolation=cv2.INTER_AREA).reshape(-1, 3) // 16
    vals, counts = np.unique(small, axis=0, return_counts=True)
    return tuple(int(v) * 16 + 8 for v in vals[np.argmax(counts)])


def classify_shape(contour) -> str:
    """Shape from how the blob fills its bounding box and enclosing circle.

    A square fills its bounding box; a triangle fills about half; a circle
    fills about 79% of its box and nearly all of its enclosing circle.
    Working from area ratios is robust to ragged, anti-aliased edges that
    make vertex counting unreliable on small shapes.
    """
    area = cv2.contourArea(contour)
    if area <= 0:
        return "unknown"
    _, _, bw, bh = cv2.boundingRect(contour)
    extent = area / float(bw * bh)
    (_, _), radius = cv2.minEnclosingCircle(contour)
    circ_fill = area / (np.pi * radius * radius)
    if extent > 0.88:
        return "square"
    if extent < 0.62:
        return "triangle"
    if circ_fill > 0.78:
        return "circle"
    return "unknown"


def label_map(frame: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Assign every pixel to its nearest named colour. Returns (labels, names)."""
    names = list(NAMED)
    refs = np.array([NAMED[n] for n in names], np.int32)                 # (K, 3)
    px = frame.reshape(-1, 3).astype(np.int32)                           # (N, 3)
    d = ((px[:, None, :] - refs[None, :, :]) ** 2).sum(axis=2)           # (N, K)
    return d.argmin(axis=1).reshape(frame.shape[:2]), names


def find_objects(frame: np.ndarray, min_area: int = 150) -> tuple[list[Obj], str]:
    """Segment by colour first, then find blobs per colour.

    Overlapping objects of different colours stay separate this way, which
    a single foreground mask cannot do.
    """
    labels, names = label_map(frame)
    counts = np.bincount(labels.ravel(), minlength=len(names))
    bg_idx = int(counts.argmax())
    objs = []
    for idx in np.nonzero(counts)[0]:
        if idx == bg_idx or counts[idx] < min_area:
            continue
        mask = (labels == idx).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area:
                continue
            m = cv2.moments(c)
            cx, cy = int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])
            objs.append(Obj(classify_shape(c), names[idx], (cx, cy), area))
    return objs, names[bg_idx]


def track_motion(frames: list[np.ndarray], start: int, end: int) -> dict[tuple[str, str], str]:
    """Direction of travel for each (shape, colour) between first and last frame of a scene."""
    first, _ = find_objects(frames[start])
    last, _ = find_objects(frames[end])
    result = {}
    for a in first:
        match = [b for b in last if (b.shape, b.color) == (a.shape, a.color)]
        if not match:
            continue
        b = min(match, key=lambda o: abs(o.centroid[0] - a.centroid[0]) + abs(o.centroid[1] - a.centroid[1]))
        dx, dy = b.centroid[0] - a.centroid[0], b.centroid[1] - a.centroid[1]
        if abs(dx) < 8 and abs(dy) < 8:
            m = "still"
        elif abs(dx) >= abs(dy):
            m = "right" if dx > 0 else "left"
        else:
            m = "down" if dy > 0 else "up"
        result[(a.shape, a.color)] = m
    return result
