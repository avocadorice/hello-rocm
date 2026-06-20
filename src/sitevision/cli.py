"""Command-line entrypoint: ``python -m sitevision.cli {demo,bench}``."""

from __future__ import annotations

import argparse
import json
import time

from . import benchmark, frames, pipeline, vlm
from .device import detect


def _frames(args):
    if args.video:
        return list(frames.video_frames(args.video, every_n=args.every_n))
    return list(frames.synthetic_frames(args.frames))


def cmd_demo(args) -> None:
    dev = detect()
    print(f"device: {dev.kind} ({dev.name})  backend: {args.backend}")
    model = vlm.build(args.backend)
    records = list(pipeline.run(_frames(args), model))
    for r in records[:8]:
        print(f"  t={r.timestamp_s:5.1f}s  {r.activity:<22} "
              f"ppe={'OK ' if r.ppe_compliant else 'VIOL'} "
              f"hazards={r.hazards}")
    if len(records) > 8:
        print(f"  ... ({len(records) - 8} more)")
    print("\nsite summary:")
    print(json.dumps(pipeline.summarize(records), indent=2))


def cmd_bench(args) -> None:
    dev = detect()
    model = vlm.build(args.backend)
    fs = _frames(args)
    t0 = time.perf_counter()
    records = list(pipeline.run(fs, model))
    wall = time.perf_counter() - t0
    res = benchmark.measure(records, wall, args.backend, dev.kind, dev.name)

    print(f"\nSiteVision benchmark — {res.backend} on {res.device_kind} "
          f"({res.device_name})")
    print(f"  frames        : {res.frames}")
    print(f"  wall          : {res.wall_s}s")
    print(f"  throughput    : {res.frames_per_s} frames/s  "
          f"({res.tokens_per_s} tok/s)")
    print(f"  latency p50/95/99 : {res.p50_ms} / {res.p95_ms} / {res.p99_ms} ms")
    print("  $ / 1k frames (illustrative on-demand rates):")
    for gpu, cost in res.cost_table().items():
        print(f"    {gpu:<14}: ${cost}")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(res.__dict__ | {"cost_per_1k": res.cost_table()}, fh, indent=2)
        print(f"  wrote {args.json}")


def main() -> None:
    p = argparse.ArgumentParser(prog="sitevision")
    sub = p.add_subparsers(dest="cmd", required=True)

    for name, fn in (("demo", cmd_demo), ("bench", cmd_bench)):
        sp = sub.add_parser(name)
        sp.add_argument("--backend", default="stub", choices=["stub", "hf"])
        sp.add_argument("--frames", type=int, default=24,
                        help="number of synthetic frames")
        sp.add_argument("--video", help="path to a real video file (needs opencv)")
        sp.add_argument("--every-n", type=int, default=8,
                        help="sample every Nth frame of --video")
        if name == "bench":
            sp.add_argument("--json", help="write benchmark result to this path")
        sp.set_defaults(func=fn)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
