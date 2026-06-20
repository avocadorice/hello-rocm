"""Throughput, latency, and cost accounting.

The cost model is deliberately simple and transparent: measured frames/sec on a
device, divided into a published $/hour rate, gives $ / 1k frames. Rates below
are *illustrative on-demand list prices* and vary widely by cloud and contract —
override them with real quotes before quoting anyone real numbers.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

from .schema import SceneRecord

# Illustrative on-demand $/GPU-hour. Override before citing.
GPU_HOURLY_USD = {
    "amd_mi300x": 3.99,
    "amd_mi250": 2.50,
    "nvidia_h100": 5.99,
    "nvidia_a100": 2.99,
}


@dataclass
class BenchResult:
    backend: str
    device_kind: str
    device_name: str
    frames: int
    wall_s: float
    frames_per_s: float
    tokens_per_s: float
    p50_ms: float
    p95_ms: float
    p99_ms: float

    def cost_table(self, rates: dict[str, float] = GPU_HOURLY_USD) -> dict[str, float]:
        """$ per 1,000 frames at this throughput, per GPU rate."""
        if self.frames_per_s <= 0:
            return {k: float("inf") for k in rates}
        usd_per_frame = lambda hourly: hourly / 3600.0 / self.frames_per_s
        return {k: round(usd_per_frame(v) * 1000, 4) for k, v in rates.items()}


def measure(records: list[SceneRecord], wall_s: float, backend: str,
            device_kind: str, device_name: str) -> BenchResult:
    lat = sorted(r.latency_ms for r in records) or [0.0]
    tokens = sum(r.output_tokens for r in records)

    def pct(p: float) -> float:
        if len(lat) == 1:
            return lat[0]
        k = min(len(lat) - 1, int(round(p / 100 * (len(lat) - 1))))
        return lat[k]

    n = len(records)
    return BenchResult(
        backend=backend, device_kind=device_kind, device_name=device_name,
        frames=n, wall_s=round(wall_s, 3),
        frames_per_s=round(n / wall_s, 2) if wall_s else 0.0,
        tokens_per_s=round(tokens / wall_s, 1) if wall_s else 0.0,
        p50_ms=round(pct(50), 1), p95_ms=round(pct(95), 1),
        p99_ms=round(pct(99), 1),
    )
