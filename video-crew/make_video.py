"""Generate a synthetic video with known content, plus ground truth JSON.

Each scene has a solid background and a few coloured shapes moving in a
straight line. Scenes are joined by hard cuts. The ground truth records
every scene's frame range, background, and objects, so the crew's output
can be scored exactly.

    python make_video.py --out data/clip1 --scenes 4 --seed 1
"""
from __future__ import annotations

import argparse
import json
import os
import random

import cv2
import numpy as np

COLORS = {  # BGR
    "red": (40, 40, 220), "green": (60, 180, 60), "blue": (220, 90, 40),
    "yellow": (40, 210, 230), "white": (240, 240, 240), "black": (20, 20, 20),
}
BACKGROUNDS = {"grey": (128, 128, 128), "navy": (90, 40, 20), "cream": (200, 230, 240), "olive": (40, 110, 110)}
SHAPES = ["circle", "square", "triangle"]
MOTIONS = {"right": (1, 0), "left": (-1, 0), "down": (0, 1), "up": (0, -1), "still": (0, 0)}


def draw(frame: np.ndarray, shape: str, color: tuple, cx: int, cy: int, r: int) -> None:
    if shape == "circle":
        cv2.circle(frame, (cx, cy), r, color, -1, cv2.LINE_AA)
    elif shape == "square":
        cv2.rectangle(frame, (cx - r, cy - r), (cx + r, cy + r), color, -1)
    else:
        pts = np.array([[cx, cy - r], [cx - r, cy + r], [cx + r, cy + r]], np.int32)
        cv2.fillPoly(frame, [pts], color, cv2.LINE_AA)


def make(out: str, scenes: int, seed: int, fps: int = 15, seconds: float = 3.0, size=(320, 240)) -> dict:
    rng = random.Random(seed)
    os.makedirs(out, exist_ok=True)
    w, h = size
    path = os.path.join(out, "video.mp4")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    truth = {"fps": fps, "size": [w, h], "scenes": []}
    frame_idx = 0
    per_scene = int(fps * seconds)
    bg_names = rng.sample(list(BACKGROUNDS), k=min(scenes, len(BACKGROUNDS)))
    for s in range(scenes):
        bg = bg_names[s % len(bg_names)]
        # Avoid object colours that match the background family.
        # Skip low-contrast pairs the video codec blurs together.
        clashes = {"cream": {"white", "yellow"}, "navy": {"blue", "black"}, "olive": {"green"}, "grey": set()}
        palette = [c for c in COLORS if c not in clashes[bg]]
        n_obj = rng.randint(1, 3)
        objs = []
        used_colors = set()
        swept = []  # (x0, y0, x1, y1) box each object covers over the whole scene
        for _ in range(n_obj):
            color = rng.choice([c for c in palette if c not in used_colors])
            for _attempt in range(50):
                motion = rng.choice(list(MOTIONS))
                r = rng.randint(18, 30)
                dx, dy = MOTIONS[motion]
                # Cap speed so the object stays fully inside the frame.
                room = (w if dx else h) - 2 * r - 40
                speed = min(rng.randint(2, 4), max(1, room // per_scene)) if motion != "still" else 0
                travel = speed * per_scene
                # Sample the start so the whole path stays inside [r+10, size-r-10].
                x0 = rng.randint(r + 10 + (travel if dx < 0 else 0), w - r - 10 - (travel if dx > 0 else 0))
                y0 = rng.randint(r + 10 + (travel if dy < 0 else 0), h - r - 10 - (travel if dy > 0 else 0))
                x1, y1 = x0 + dx * travel, y0 + dy * travel
                box = (min(x0, x1) - r, min(y0, y1) - r, max(x0, x1) + r, max(y0, y1) + r)
                if all(box[2] < b[0] or b[2] < box[0] or box[3] < b[1] or b[3] < box[1] for b in swept):
                    break
            else:
                continue  # could not place without overlap; skip this object
            swept.append(box)
            used_colors.add(color)
            objs.append({"shape": rng.choice(SHAPES), "color": color, "motion": motion, "radius": r,
                         "start": [x0, y0], "speed": speed})
        for f in range(per_scene):
            frame = np.full((h, w, 3), BACKGROUNDS[bg], np.uint8)
            for o in objs:
                dx, dy = MOTIONS[o["motion"]]
                cx = o["start"][0] + dx * o["speed"] * f
                cy = o["start"][1] + dy * o["speed"] * f
                draw(frame, o["shape"], COLORS[o["color"]], int(cx), int(cy), o["radius"])
            writer.write(frame)
        truth["scenes"].append({
            "index": s, "start_frame": frame_idx, "end_frame": frame_idx + per_scene - 1,
            "background": bg,
            "objects": [{"shape": o["shape"], "color": o["color"], "motion": o["motion"]} for o in objs],
        })
        frame_idx += per_scene
    writer.release()
    truth["frames"] = frame_idx
    with open(os.path.join(out, "truth.json"), "w") as f:
        json.dump(truth, f, indent=2)
    return truth


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/clip1")
    p.add_argument("--scenes", type=int, default=4)
    p.add_argument("--seed", type=int, default=1)
    a = p.parse_args()
    t = make(a.out, a.scenes, a.seed)
    print(f"wrote {a.out}/video.mp4 with {len(t['scenes'])} scenes, {t['frames']} frames")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
