"""Device detection across ROCm / CUDA / Apple MPS / CPU.

The whole point of the project is that one code path runs on AMD and NVIDIA
alike. PyTorch+ROCm reuses the ``torch.cuda`` namespace and sets
``torch.version.hip``, so we distinguish the two by that attribute rather than
by the device string (which is ``"cuda"`` on both).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Device:
    """A resolved compute target.

    Attributes:
        kind: Vendor/runtime label — one of ``rocm``, ``cuda``, ``mps``, ``cpu``.
        torch_device: The string you actually pass to ``.to(...)`` in torch.
        name: Human-readable accelerator name, when discoverable.
    """

    kind: str
    torch_device: str
    name: str = ""

    @property
    def is_accelerated(self) -> bool:
        return self.kind in ("rocm", "cuda", "mps")


def detect() -> Device:
    """Resolve the best available device without importing torch eagerly.

    Falls back cleanly to CPU so the stub pipeline runs anywhere — including a
    Mac with no torch installed at all.
    """
    try:
        import torch
    except ImportError:
        return Device(kind="cpu", torch_device="cpu", name="cpu (no torch)")

    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        # ROCm builds carry torch.version.hip; CUDA builds carry torch.version.cuda.
        if getattr(torch.version, "hip", None):
            return Device(kind="rocm", torch_device="cuda", name=name)
        return Device(kind="cuda", torch_device="cuda", name=name)

    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return Device(kind="mps", torch_device="mps", name="Apple MPS")

    return Device(kind="cpu", torch_device="cpu", name="cpu")
