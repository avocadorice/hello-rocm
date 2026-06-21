"""VLM backends.

Two implementations behind a common ``analyze(frame) -> SceneRecord`` interface:

  * ``StubVLM``      — deterministic, dependency-free. Recovers the synthetic
                       frame's ground truth (or hashes the image otherwise) and
                       simulates a realistic per-frame latency. Lets the entire
                       pipeline + benchmark run with no GPU and no weights.
  * ``HFVLM``        — a real small open VLM (Moondream2 by default) via 🤗
                       Transformers, placed on the detected device (ROCm/CUDA/
                       MPS/CPU).
"""

from __future__ import annotations

import hashlib
import json
import time

from .device import Device, detect
from .frames import Frame
from .schema import SceneRecord

# What we ask the model to extract — kept tight so output is parseable JSON.
PROMPT = (
    "You are analyzing a single first-person frame from a construction worker's "
    "wearable camera. Respond with JSON: {activity, tools[], ppe_compliant(bool), "
    "missing_ppe[], hazards[], confidence(0-1)}."
)


class StubVLM:
    """Deterministic offline backend. No torch, no weights, no GPU."""

    name = "stub"

    def __init__(self, sim_latency_ms: float = 18.0):
        self.device = detect()
        self.sim_latency_ms = sim_latency_ms

    def analyze(self, frame: Frame) -> SceneRecord:
        t0 = time.perf_counter()
        if frame.truth is not None:
            t = frame.truth
            rec = SceneRecord(
                frame_index=frame.index, timestamp_s=frame.timestamp_s,
                activity=t["activity"], tools=list(t["tools"]),
                ppe_compliant=not t["missing_ppe"],
                missing_ppe=list(t["missing_ppe"]), hazards=list(t["hazards"]),
                confidence=0.88,
            )
        else:
            # No ground truth (real video): hash the image to a stable label.
            h = hashlib.md5(frame.image.tobytes()).digest()[0]
            rec = SceneRecord(
                frame_index=frame.index, timestamp_s=frame.timestamp_s,
                activity=f"scene-{h % 6}", confidence=0.5,
            )
        # crude token estimate from the JSON payload
        rec.output_tokens = max(1, len(json.dumps(rec.to_dict())) // 4)
        # simulate compute so the benchmark produces a realistic shape
        spin_until = t0 + self.sim_latency_ms / 1000.0
        while time.perf_counter() < spin_until:
            pass
        rec.latency_ms = (time.perf_counter() - t0) * 1000.0
        return rec


class HFVLM:
    """Real VLM backend via 🤗 Transformers (default: Moondream2)."""

    name = "hf"

    def __init__(self, model_id: str = "vikhyatk/moondream2",
                 revision: str = "2024-08-26",
                 device: Device | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = device or detect()
        dtype = torch.float16 if self.device.is_accelerated else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, revision=revision, trust_remote_code=True,
            torch_dtype=dtype,
        ).to(self.device.torch_device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

    def analyze(self, frame: Frame) -> SceneRecord:
        t0 = time.perf_counter()
        enc = self.model.encode_image(frame.image)
        answer = self.model.answer_question(enc, PROMPT, self.tokenizer)
        rec = _parse(answer, frame)
        rec.output_tokens = len(self.tokenizer.encode(answer))
        rec.latency_ms = (time.perf_counter() - t0) * 1000.0
        return rec


def _parse(answer: str, frame: Frame) -> SceneRecord:
    """Best-effort JSON parse of a model answer into a SceneRecord."""
    rec = SceneRecord(frame_index=frame.index, timestamp_s=frame.timestamp_s,
                      activity="unparsed")
    try:
        start, end = answer.find("{"), answer.rfind("}")
        data = json.loads(answer[start:end + 1])
        rec.activity = data.get("activity", "unknown")
        rec.tools = data.get("tools", [])
        rec.missing_ppe = data.get("missing_ppe", [])
        rec.ppe_compliant = bool(data.get("ppe_compliant", not rec.missing_ppe))
        rec.hazards = data.get("hazards", [])
        rec.confidence = float(data.get("confidence", 0.0))
    except (ValueError, json.JSONDecodeError):
        rec.activity = answer.strip()[:80]
    return rec


def build(backend: str) -> StubVLM | HFVLM:
    if backend == "stub":
        return StubVLM()
    if backend == "hf":
        return HFVLM()
    raise ValueError(f"unknown backend: {backend!r} (use 'stub' or 'hf')")
