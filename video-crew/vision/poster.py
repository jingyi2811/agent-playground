"""Poster generation. Diffusion when available, an OpenCV mosaic otherwise."""
from __future__ import annotations

import os

import cv2
import numpy as np


def mosaic_poster(keyframes: list[np.ndarray], title: str, out: str) -> str:
    """Placeholder poster: keyframe grid with a title band."""
    if not keyframes:
        raise ValueError("no keyframes")
    h, w = keyframes[0].shape[:2]
    cols = min(len(keyframes), 3)
    rows = (len(keyframes) + cols - 1) // cols
    canvas = np.full((rows * h + 60, cols * w, 3), 30, np.uint8)
    for i, k in enumerate(keyframes):
        r, c = divmod(i, cols)
        canvas[60 + r * h : 60 + (r + 1) * h, c * w : (c + 1) * w] = cv2.resize(k, (w, h))
    cv2.putText(canvas, title[:60], (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (240, 240, 240), 2, cv2.LINE_AA)
    cv2.imwrite(out, canvas)
    return out


def diffusion_poster(prompt: str, out: str, model_id: str = "stabilityai/sd-turbo",
                     steps: int = 2, guidance: float = 0.0, seed: int = 0) -> str:
    """Generate with a small Stable Diffusion model.

    sd-turbo is distilled for 1 to 4 steps with guidance 0. Increase steps
    and switch to a full SD model for higher quality at much higher cost.
    Requires: pip install torch diffusers transformers accelerate
    """
    import torch
    from diffusers import AutoPipelineForImage2Image, AutoPipelineForText2Image  # noqa: F401

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device != "cpu" else torch.float32
    pipe = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=dtype).to(device)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    image = pipe(prompt=prompt, num_inference_steps=steps, guidance_scale=guidance, generator=gen).images[0]
    image.save(out)
    return out


def make_poster(keyframes: list[np.ndarray], title: str, prompt: str, out: str, use_diffusion: bool) -> tuple[str, str]:
    """Returns (path, method)."""
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    if use_diffusion:
        try:
            return diffusion_poster(prompt, out), "diffusion"
        except ImportError as e:
            print(f"diffusion unavailable ({e}); falling back to mosaic")
    return mosaic_poster(keyframes, title, out), "mosaic"
