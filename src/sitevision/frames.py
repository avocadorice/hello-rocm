"""Frame sources.

Two ways to get frames:
  * ``synthetic_frames`` — generate plausible first-person job-site frames with
    no external data, so the pipeline runs on any laptop. Each frame embeds a
    hidden "ground truth" scene tag the stub VLM can recover deterministically.
  * ``video_frames`` — sample real frames from a video file (needs opencv).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterator

from PIL import Image, ImageDraw

# Scene templates the synthetic generator cycles through. The stub VLM is
# designed to recover these, so end-to-end output looks realistic offline.
SCENE_TEMPLATES = [
    {"activity": "operating angle grinder", "tools": ["angle grinder"],
     "missing_ppe": ["face shield"], "hazards": ["sparks", "no face shield"]},
    {"activity": "carrying lumber", "tools": [], "missing_ppe": [], "hazards": []},
    {"activity": "welding steel beam", "tools": ["welder"],
     "missing_ppe": [], "hazards": ["arc flash", "hot metal"]},
    {"activity": "climbing scaffold", "tools": [], "missing_ppe": ["harness"],
     "hazards": ["fall risk", "unsecured harness"]},
    {"activity": "pouring concrete", "tools": ["concrete hose"],
     "missing_ppe": [], "hazards": ["wet surface"]},
    {"activity": "idle / walking site", "tools": [], "missing_ppe": [], "hazards": []},
]


@dataclass
class Frame:
    index: int
    timestamp_s: float
    image: Image.Image
    truth: dict | None = None  # hidden ground-truth tag for synthetic frames


def _tint(seed: int) -> tuple[int, int, int]:
    h = hashlib.md5(str(seed).encode()).digest()
    return (60 + h[0] % 120, 60 + h[1] % 120, 60 + h[2] % 120)


def synthetic_frames(n: int, fps: float = 4.0, size: int = 384) -> Iterator[Frame]:
    """Yield ``n`` synthetic first-person frames with embedded ground truth."""
    for i in range(n):
        template = SCENE_TEMPLATES[i % len(SCENE_TEMPLATES)]
        img = Image.new("RGB", (size, size), _tint(i))
        d = ImageDraw.Draw(img)
        # crude "first-person" framing: a horizon + a foreground tool block
        d.rectangle([0, size * 2 // 3, size, size], fill=(40, 40, 45))
        d.rectangle([size // 3, size // 2, size * 2 // 3, size * 3 // 4],
                    fill=_tint(i + 7))
        d.text((8, 8), template["activity"], fill=(235, 235, 235))
        yield Frame(index=i, timestamp_s=i / fps, image=img, truth=template)


def video_frames(path: str, every_n: int = 8) -> Iterator[Frame]:
    """Sample every ``every_n``-th frame from a real video file."""
    import cv2  # opencv-python-headless; optional dependency

    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    idx = out = 0
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            if idx % every_n == 0:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                yield Frame(index=out, timestamp_s=idx / fps,
                            image=Image.fromarray(rgb))
                out += 1
            idx += 1
    finally:
        cap.release()
