"""Run the whole crew on one video."""
from __future__ import annotations

import json
import os
import shutil
import time

from vision import describe as ds
from vision import poster as po
from vision import scenes as sc

from .board import Board
from .roles import Context, classical_verifier, composer, run_describers, splitter, verifier


def run_crew(video: str, out: str, mode: str = "classical", workers: int = 3,
             use_diffusion: bool = False, model: str = "claude-opus-5") -> dict:
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(out)
    frames, fps = sc.read_frames(video)

    if mode == "claude":
        cd = ds.ClaudeDescriber(model)
        describer, verify = cd.describe, cd.verify
    else:
        describer, verify = ds.classical, classical_verifier

    board = Board(os.path.join(out, "board"))
    ctx = Context(frames=frames, fps=fps, describer=describer, verifier=verify)
    timings = {}

    t = time.perf_counter(); scenes = splitter(board, ctx); timings["split"] = time.perf_counter() - t
    t = time.perf_counter(); work = run_describers(board, ctx, workers); timings["describe"] = time.perf_counter() - t
    t = time.perf_counter(); checked = verifier(board, ctx); timings["verify"] = time.perf_counter() - t
    t = time.perf_counter(); summary = composer(board, ctx); timings["compose"] = time.perf_counter() - t

    keyframes = [frames[s.keyframe] for s in scenes]
    title = f"{len(scenes)} scenes"
    prompt = "movie poster, bold flat graphic style: " + summary.get("text", "")[:400]
    poster_path, method = po.make_poster(keyframes, title, prompt, os.path.join(out, "poster.png"), use_diffusion)

    report = {
        "video": video, "mode": mode, "frames": len(frames), "fps": fps,
        "scenes": [{"index": s.index, "start": s.start_frame, "end": s.end_frame} for s in scenes],
        "describer_work": work, "verified_scenes": checked,
        "board": board.summary(), "summary": summary, "poster": {"path": poster_path, "method": method},
        "timings": {k: round(v, 3) for k, v in timings.items()},
    }
    with open(os.path.join(out, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    return report
