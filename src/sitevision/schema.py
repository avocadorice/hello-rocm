"""Structured output schema for a single analyzed frame.

Modeled on what a construction-site egocentric-video system actually needs to
extract: what the worker is doing, what tools are in frame, whether required
PPE is present, and any visible hazards.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class SceneRecord:
    frame_index: int
    timestamp_s: float
    activity: str                       # e.g. "operating angle grinder"
    tools: list[str] = field(default_factory=list)
    ppe_compliant: bool = True
    missing_ppe: list[str] = field(default_factory=list)
    hazards: list[str] = field(default_factory=list)
    confidence: float = 0.0
    # Inference accounting, useful for the benchmark/cost layer.
    latency_ms: float = 0.0
    output_tokens: int = 0

    def to_dict(self) -> dict:
        return asdict(self)
