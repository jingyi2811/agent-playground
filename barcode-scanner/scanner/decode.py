"""Decode barcodes with zxing-cpp, trying the located crops first."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import zxingcpp

from . import locate, preprocess


@dataclass(frozen=True)
class Result:
    text: str
    format: str
    pipeline: str
    region: tuple[int, int, int, int] | None  # None means whole frame


def _decode_array(gray: np.ndarray) -> list[zxingcpp.Result]:
    return zxingcpp.read_barcodes(gray, try_rotate=True, try_downscale=True)


def decode(img: np.ndarray, pipeline: str = "full") -> list[Result]:
    """Decode every barcode in `img` using the named preprocessing pipeline.

    Tries located regions first, then the full frame as a fallback.
    """
    gray = preprocess.to_gray(img)
    seen: set[str] = set()
    results: list[Result] = []

    for box in locate.find_candidates(gray):
        processed = preprocess.run_pipeline(locate.crop(gray, box), pipeline)
        for r in _decode_array(processed):
            if r.text not in seen:
                seen.add(r.text)
                results.append(Result(r.text, str(r.format), pipeline, box))

    if not results:
        processed = preprocess.run_pipeline(gray, pipeline)
        for r in _decode_array(processed):
            if r.text not in seen:
                seen.add(r.text)
                results.append(Result(r.text, str(r.format), pipeline, None))

    return results
