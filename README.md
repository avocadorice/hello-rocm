# SiteVision — Egocentric VLM Inference on AMD ROCm

> Run a vision-language model over **first-person job-site video** and emit
> structured scene understanding — activity, tools, PPE/safety compliance,
> hazards — on **AMD ROCm**, with a throughput + cost benchmark vs NVIDIA.

This is a learning/portfolio project built on top of the
[`hello-rocm`](./goal.md) curriculum. It exists to answer one concrete
question: *what does it actually take to serve VLM inference over egocentric
video on AMD hardware, and how does the cost math compare to CUDA?*

## Why this project

Egocentric (head/chest-mounted) video understanding at scale is bottlenecked by
**inference cost**, not model quality. A fleet of wearables generating millions
of hours of footage turns every cent-per-frame into a real line item. NVIDIA is
the default; AMD ROCm is the cost lever almost nobody has pipeline-tested
end-to-end.

SiteVision is a minimal but honest end-to-end pipeline:

```
first-person video ──▶ frame sampler ──▶ VLM ──▶ structured scene record
                                        (ROCm)    { activity, tools,
                                                    ppe_compliant, hazards }
                                          │
                                          └──▶ benchmark: frames/s, tokens/s,
                                               p50/p99 latency, $ / 1k frames
                                               (AMD MIxxx vs NVIDIA Hxxx)
```

## Design goals

- **ROCm-first, portable to run anywhere.** The device layer detects ROCm /
  CUDA / Apple MPS / CPU and runs the same code path. You can develop on a Mac
  and ship the *identical* pipeline to an AMD box via `Dockerfile.rocm`.
- **Demoable with zero downloads.** A deterministic `stub` VLM backend lets the
  whole pipeline + benchmark run instantly, so the plumbing is provable even
  before weights or a GPU are in the picture.
- **Real backend when you want numbers.** `--backend hf` loads a small open VLM
  (Moondream2 / Qwen2-VL) through 🤗 Transformers.
- **Cost is a first-class output**, not an afterthought — because that's the
  whole point of looking at AMD.

## Quickstart (runs on a Mac, no GPU, no model download)

```bash
pip install -r requirements.txt
# generate synthetic first-person frames + run the stub pipeline
python -m sitevision.cli demo --frames 24
# benchmark throughput + cost
python -m sitevision.cli bench --frames 200 --backend stub
```

## Real ROCm run

```bash
docker build -f Dockerfile.rocm -t sitevision:rocm .
docker run --rm --device=/dev/kfd --device=/dev/dri \
  --security-opt seccomp=unconfined --group-add video \
  sitevision:rocm python -m sitevision.cli bench --frames 200 --backend hf
```

## Status

v0 — pipeline, device abstraction, stub backend, benchmark + cost model, ROCm
container. See [`goal.md`](./goal.md) for the broader ROCm learning roadmap.
