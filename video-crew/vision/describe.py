"""Frame describers. Both return the same caption schema so the verifier
and composer do not care which one produced it.

Caption schema:
    {"background": "grey", "objects": [{"shape": "circle", "color": "red", "motion": "right"}, ...]}
"""
from __future__ import annotations

import base64
import json
import re

import cv2
import numpy as np

from . import motion as mo

SCHEMA_HINT = (
    'Respond with JSON only: {"background": <colour word>, "objects": [{"shape": "circle"|"square"|"triangle", '
    '"color": <colour word>, "motion": "left"|"right"|"up"|"down"|"still"}]}. '
    "Colour words: red, green, blue, yellow, white, black, grey, navy, cream, olive."
)


def classical(frames: list, start: int, end: int) -> dict:
    key = frames[(start + end) // 2]
    objs, bg = mo.find_objects(key)
    motions = mo.track_motion(frames, start, end)
    return {
        "background": bg,
        "objects": [
            {"shape": o.shape, "color": o.color, "motion": motions.get((o.shape, o.color), "still")}
            for o in objs
        ],
    }


def _png_b64(frame: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", frame)
    return base64.standard_b64encode(buf.tobytes()).decode()


class ClaudeDescriber:
    """Captions with Claude vision. Sends first, middle and last frame so
    motion can be judged, and asks for the shared JSON schema."""

    def __init__(self, model: str = "claude-opus-5"):
        import anthropic

        self.client = anthropic.Anthropic()
        self.model = model

    def _ask(self, frames: list, text: str) -> str:
        content = []
        for i, f in enumerate(frames):
            content.append({"type": "text", "text": f"Frame {i + 1} of {len(frames)}:"})
            content.append({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": _png_b64(f)}})
        content.append({"type": "text", "text": text})
        resp = self.client.beta.messages.create(
            model=self.model, max_tokens=2000,
            betas=["server-side-fallback-2026-07-01"],
            extra_body={"fallbacks": "default", "output_config": {"effort": "low"}},
            messages=[{"role": "user", "content": content}],
        )
        if resp.stop_reason == "refusal":
            raise RuntimeError("refused")
        return "".join(b.text for b in resp.content if b.type == "text")

    def describe(self, frames: list, start: int, end: int) -> dict:
        trio = [frames[start], frames[(start + end) // 2], frames[end]]
        text = self._ask(trio, "These are the first, middle and last frames of one scene. Describe the scene. " + SCHEMA_HINT)
        return parse_json(text)

    def verify(self, frames: list, start: int, end: int, claims: list) -> list:
        trio = [frames[start], frames[(start + end) // 2], frames[end]]
        numbered = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(claims))
        text = self._ask(trio, "For each numbered claim about this scene, answer true or false. "
                               f"Respond with JSON only: a list of booleans in order.\n{numbered}")
        out = parse_json(text)
        return [bool(x) for x in out] if isinstance(out, list) else [False] * len(claims)


def parse_json(text: str):
    m = re.search(r"\{.*\}|\[.*\]", text, re.S)
    return json.loads(m.group(0) if m else text)


def claims_from(caption: dict) -> list:
    """Flatten a caption into atomic, checkable sentences."""
    out = [f"the background is {caption.get('background')}"]
    for o in caption.get("objects", []):
        out.append(f"there is a {o.get('color')} {o.get('shape')}")
        out.append(f"the {o.get('color')} {o.get('shape')} moves {o.get('motion')}" if o.get("motion") != "still"
                   else f"the {o.get('color')} {o.get('shape')} stays still")
    return out
