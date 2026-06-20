"""End-to-end pipeline: frames in, structured scene records out."""

from __future__ import annotations

from typing import Iterable, Iterator

from .frames import Frame
from .schema import SceneRecord


def run(frames: Iterable[Frame], vlm) -> Iterator[SceneRecord]:
    """Analyze each frame with the given VLM backend, yielding records."""
    for frame in frames:
        yield vlm.analyze(frame)


def summarize(records: list[SceneRecord]) -> dict:
    """Roll up per-frame records into a site-level safety summary."""
    flagged = [r for r in records if not r.ppe_compliant or r.hazards]
    return {
        "frames_analyzed": len(records),
        "ppe_violations": sum(1 for r in records if not r.ppe_compliant),
        "frames_with_hazards": sum(1 for r in records if r.hazards),
        "flagged_examples": [
            {"t": round(r.timestamp_s, 1), "activity": r.activity,
             "missing_ppe": r.missing_ppe, "hazards": r.hazards}
            for r in flagged[:5]
        ],
    }
