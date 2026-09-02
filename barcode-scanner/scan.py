"""Scan a single image file or a live camera feed.

    python scan.py image.jpg
    python scan.py --camera 0
"""
from __future__ import annotations

import argparse
import sys

import cv2

from scanner import decode, validate


def scan_image(path: str, pipeline: str) -> int:
    img = cv2.imread(path)
    if img is None:
        print(f"could not read {path}", file=sys.stderr)
        return 1
    results = decode.decode(img, pipeline)
    if not results:
        print("no barcode found")
        return 2
    for r in results:
        valid = validate.is_valid_gtin(r.text)
        print(f"{r.format:<12} {r.text:<20} gtin_valid={valid} region={r.region}")
    return 0


def scan_camera(index: int, pipeline: str) -> int:
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(f"camera {index} not available", file=sys.stderr)
        return 1
    last = ""
    print("press q to quit")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        for r in decode.decode(frame, pipeline):
            if r.region:
                x, y, w, h = r.region
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, r.text, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if r.text != last:
                last = r.text
                print(f"{r.format}: {r.text}")
        cv2.imshow("scan", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("image", nargs="?", help="image file to scan")
    p.add_argument("--camera", type=int, help="camera index for live scanning")
    p.add_argument("--pipeline", default="full", help="preprocessing pipeline name")
    a = p.parse_args()
    if a.camera is not None:
        return scan_camera(a.camera, a.pipeline)
    if a.image:
        return scan_image(a.image, a.pipeline)
    p.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
