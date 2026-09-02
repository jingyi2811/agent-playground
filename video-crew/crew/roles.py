"""The four roles. Each is a function of (board, shared video context).

They never call each other. Describers run in parallel and only see their
own task. The verifier re-derives facts from the frames rather than
trusting the describer, so a wrong caption is caught by evidence, not by
a second opinion on the same text.
"""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from vision import describe as ds
from vision import scenes as sc

from .board import Board, Task

Describer = Callable[[list, int, int], dict]
Verifier = Callable[[list, int, int, list], list]


@dataclass
class Context:
    frames: list
    fps: float
    describer: Describer
    verifier: Verifier
    scenes: Optional[list] = None


# -- splitter -------------------------------------------------------------------

def splitter(board: Board, ctx: Context, name: str = "splitter") -> list[sc.Scene]:
    ctx.scenes = sc.detect_scenes(ctx.frames)
    for s in ctx.scenes:
        board.post(Task(id=f"describe-{s.index:02d}", kind="describe",
                        payload={"scene": s.index, "start": s.start_frame, "end": s.end_frame}), by=name)
        board.post(Task(id=f"verify-{s.index:02d}", kind="verify",
                        payload={"scene": s.index, "start": s.start_frame, "end": s.end_frame},
                        depends_on=[f"describe-{s.index:02d}"]), by=name)
    board.post(Task(id="compose", kind="compose", payload={},
                    depends_on=[f"verify-{s.index:02d}" for s in ctx.scenes]), by=name)
    return ctx.scenes


# -- describers (parallel workers) ---------------------------------------------

def describer_worker(board: Board, ctx: Context, name: str) -> int:
    """Claim describe tasks until none are left. Returns tasks completed."""
    done = 0
    while True:
        ready = board.ready("describe")
        if not ready:
            return done
        t = board.claim(ready[0].id, name)
        if t is None:
            continue  # someone else got it; look again
        p = t.payload
        try:
            caption = ctx.describer(ctx.frames, p["start"], p["end"])
            board.finish(t.id, name, {"caption": caption}, ok=True)
            done += 1
        except Exception as e:
            board.finish(t.id, name, {"error": str(e)}, ok=False)


def run_describers(board: Board, ctx: Context, workers: int = 3) -> dict[str, int]:
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {f"describer-{i}": ex.submit(describer_worker, board, ctx, f"describer-{i}") for i in range(workers)}
        return {k: f.result() for k, f in futures.items()}


# -- verifier -------------------------------------------------------------------

def verifier(board: Board, ctx: Context, name: str = "verifier") -> int:
    checked = 0
    for t in board.ready("verify"):
        if board.claim(t.id, name) is None:
            continue
        p = t.payload
        desc = board.get(f"describe-{p['scene']:02d}")
        caption = (desc.result or {}).get("caption")
        if not caption:
            board.finish(t.id, name, {"verdicts": [], "note": "no caption"}, ok=False)
            continue
        claims = ds.claims_from(caption)
        verdicts = ctx.verifier(ctx.frames, p["start"], p["end"], claims)
        rejected = [c for c, v in zip(claims, verdicts) if not v]
        board.finish(t.id, name, {"claims": claims, "verdicts": verdicts, "rejected": rejected}, ok=True)
        checked += 1
    return checked


def classical_verifier(frames: list, start: int, end: int, claims: list[str]) -> list[bool]:
    """Re-derive the scene from frames and check each claim against it."""
    truth = ds.classical(frames, start, end)
    facts = set(ds.claims_from(truth))
    return [c in facts for c in claims]


# -- composer -------------------------------------------------------------------

def composer(board: Board, ctx: Context, name: str = "composer") -> dict:
    ready = [t for t in board.ready("compose")]
    if not ready or board.claim("compose", name) is None:
        return {}
    lines, kept_scenes = [], []
    for s in ctx.scenes or []:
        desc = board.get(f"describe-{s.index:02d}").result or {}
        ver = board.get(f"verify-{s.index:02d}").result or {}
        caption = desc.get("caption", {})
        rejected = set(ver.get("rejected", []))
        objs = []
        for o in caption.get("objects", []):
            exists = f"there is a {o.get('color')} {o.get('shape')}"
            if exists in rejected:
                continue  # verifier says this object is not there
            motion_claim = (f"the {o.get('color')} {o.get('shape')} moves {o.get('motion')}"
                            if o.get("motion") != "still" else f"the {o.get('color')} {o.get('shape')} stays still")
            motion = o.get("motion") if motion_claim not in rejected else "unknown motion"
            objs.append({"shape": o.get("shape"), "color": o.get("color"), "motion": motion})
        bg = caption.get("background") if f"the background is {caption.get('background')}" not in rejected else "unverified"
        t0, t1 = s.start_frame / ctx.fps, (s.end_frame + 1) / ctx.fps
        desc_text = ", ".join(f"{o['color']} {o['shape']} ({o['motion']})" for o in objs) or "nothing verified"
        lines.append(f"Scene {s.index + 1} ({t0:.1f}s to {t1:.1f}s): {desc_text} on a {bg} background.")
        kept_scenes.append({"scene": s.index, "background": bg, "objects": objs, "rejected": sorted(rejected)})
    summary = {"text": "\n".join(lines), "scenes": kept_scenes}
    board.finish("compose", name, summary, ok=True)
    return summary
