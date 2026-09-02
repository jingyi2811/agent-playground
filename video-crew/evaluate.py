"""Score the crew against synthetic ground truth, and test the verifier
with planted wrong captions.

    python evaluate.py --clips 5
"""
from __future__ import annotations

import argparse
import copy
import os
import random

from crew.roles import classical_verifier
from crew.run import run_crew
from make_video import COLORS, SHAPES, make
from vision import describe as ds
from vision import scenes as sc


def score_clip(truth: dict, report: dict) -> dict:
    t_scenes, r_scenes = truth["scenes"], report["summary"]["scenes"]
    scene_ok = len(t_scenes) == len(r_scenes)
    obj_total = obj_hit = motion_hit = bg_hit = 0
    for i, ts in enumerate(t_scenes):
        rs = r_scenes[i] if i < len(r_scenes) else {"objects": [], "background": None}
        bg_hit += rs.get("background") == ts["background"]
        found = {(o["shape"], o["color"]): o.get("motion") for o in rs["objects"]}
        for o in ts["objects"]:
            obj_total += 1
            key = (o["shape"], o["color"])
            if key in found:
                obj_hit += 1
                motion_hit += found[key] == o["motion"]
    return {"scene_count_ok": scene_ok, "objects": obj_total, "object_recall": obj_hit / max(obj_total, 1),
            "motion_acc": motion_hit / max(obj_hit, 1), "background_acc": bg_hit / len(t_scenes)}


def plant_errors(caption: dict, rng: random.Random) -> tuple[dict, list[str]]:
    """Corrupt one fact and return the caption plus the claim that is now false."""
    bad = copy.deepcopy(caption)
    if not bad["objects"]:
        return bad, []
    present = {(x["shape"], x["color"]) for x in bad["objects"]}
    o = rng.choice(bad["objects"])
    field = rng.choice(["color", "shape"])
    # Pick a replacement that does not describe another object that really is there.
    choices = [c for c in (list(COLORS) if field == "color" else SHAPES)
               if c != o[field] and ((c, o["color"]) if field == "shape" else (o["shape"], c)) not in present]
    if not choices:
        return bad, []
    o[field] = rng.choice(choices)
    return bad, [f"there is a {o['color']} {o['shape']}"]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--clips", type=int, default=5)
    p.add_argument("--out", default="eval_runs")
    a = p.parse_args()
    rng = random.Random(0)

    rows, planted, caught = [], 0, 0
    for i in range(a.clips):
        clip_dir = os.path.join(a.out, f"clip{i}")
        truth = make(clip_dir, scenes=rng.randint(2, 5), seed=100 + i)
        report = run_crew(os.path.join(clip_dir, "video.mp4"), os.path.join(clip_dir, "run"))
        s = score_clip(truth, report)
        rows.append(s)

        # Verifier test: feed it deliberately wrong captions and see if it objects.
        frames, _ = sc.read_frames(os.path.join(clip_dir, "video.mp4"))
        for sc_ in report["scenes"]:
            good = ds.classical(frames, sc_["start"], sc_["end"])
            bad, false_claims = plant_errors(good, rng)
            if not false_claims:
                continue
            claims = ds.claims_from(bad)
            verdicts = classical_verifier(frames, sc_["start"], sc_["end"], claims)
            planted += 1
            caught += all(not v for c, v in zip(claims, verdicts) if c in false_claims)
        print(f"clip{i}: scenes_ok={s['scene_count_ok']} object_recall={s['object_recall']:.0%} "
              f"motion_acc={s['motion_acc']:.0%} background_acc={s['background_acc']:.0%}")

    n = len(rows)
    print(f"\nover {n} clips: scene count correct {sum(r['scene_count_ok'] for r in rows)}/{n}, "
          f"object recall {sum(r['object_recall'] for r in rows) / n:.0%}, "
          f"motion accuracy {sum(r['motion_acc'] for r in rows) / n:.0%}, "
          f"background accuracy {sum(r['background_acc'] for r in rows) / n:.0%}")
    print(f"verifier caught {caught}/{planted} planted wrong captions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
