"""Summarise a video with the crew.

    python summarise.py data/clip1/video.mp4 --out runs/clip1
    python summarise.py data/clip1/video.mp4 --mode claude --diffusion
"""
from __future__ import annotations

import argparse

from crew.run import run_crew


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("video")
    p.add_argument("--out", default="runs/latest")
    p.add_argument("--mode", choices=["classical", "claude"], default="classical")
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--diffusion", action="store_true", help="generate the poster with Stable Diffusion")
    p.add_argument("--model", default="claude-opus-5")
    a = p.parse_args()
    r = run_crew(a.video, a.out, a.mode, a.workers, a.diffusion, a.model)
    print(r["summary"]["text"])
    print(f"\nboard: {r['board']}  describers: {r['describer_work']}  poster: {r['poster']['method']} -> {r['poster']['path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
