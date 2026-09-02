import cv2
import numpy as np
import zxingcpp

from scanner import decode, preprocess


def _render(text: str) -> np.ndarray:
    bits = zxingcpp.write_barcode(zxingcpp.BarcodeFormat.EAN13, text, width=400, height=160, quiet_zone=10)
    gray = np.array(bits, dtype=np.uint8)
    canvas = np.full((gray.shape[0] + 200, gray.shape[1] + 200), 235, np.uint8)
    canvas[100 : 100 + gray.shape[0], 100 : 100 + gray.shape[1]] = gray
    return cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)


def test_decodes_clean_image_every_pipeline():
    img = _render("4006381333931")
    for name in preprocess.PIPELINES:
        got = {r.text for r in decode.decode(img, name)}
        assert "4006381333931" in got, name


def test_decodes_rotated_blurred_image():
    img = _render("4006381333931")
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), 18, 1.0)
    img = cv2.warpAffine(img, m, (w, h), borderValue=(235, 235, 235))
    img = cv2.GaussianBlur(img, (5, 5), 0)
    got = {r.text for r in decode.decode(img, "full")}
    assert "4006381333931" in got


def test_empty_image_returns_nothing():
    blank = np.full((300, 400, 3), 200, np.uint8)
    assert decode.decode(blank, "full") == []
