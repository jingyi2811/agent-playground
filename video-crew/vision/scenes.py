"""Scene cut detection and keyframe extraction."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Scene:
    index: int
    start_frame: int
    end_frame: int

    @property
    def keyframe(self) -> int:
        return (self.start_frame + self.end_frame) // 2


def read_frames(path: str) -> tuple[list[np.ndarray], float]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames, fps


def _hist(frame: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    return cv2.normalize(h, h).flatten()


def detect_scenes(frames: list[np.ndarray], threshold: float = 0.5, min_len: int = 5) -> list[Scene]:
    """A cut is a frame whose HSV histogram differs sharply from the previous one.

    Bhattacharyya distance is near 0 for similar frames and near 1 for
    unrelated ones. Moving objects change the histogram a little; a
    background change moves it a lot.
    """
    if not frames:
        return []
    cuts = [0]
    prev = _hist(frames[0])
    for i in range(1, len(frames)):
        cur = _hist(frames[i])
        d = cv2.compareHist(prev, cur, cv2.HISTCMP_BHATTACHARYYA)
        if d > threshold and i - cuts[-1] >= min_len:
            cuts.append(i)
        prev = cur
    bounds = cuts + [len(frames)]
    return [Scene(k, bounds[k], bounds[k + 1] - 1) for k in range(len(cuts))]
