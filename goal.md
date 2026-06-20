# Learning Goal: hello-rocm

**Source:** https://github.com/datawhalechina/hello-rocm/blob/master/README_en.md

---

## What is hello-rocm?

A community-driven, open-source project that makes AMD's ROCm GPU computing platform accessible for AI and LLM development. It fills the gap left by the CUDA-centric ecosystem by providing structured, hands-on learning materials for deploying, fine-tuning, and optimizing models on AMD GPUs.

---

## Key milestone

ROCm 7.10.0 (December 2025) decoupled the runtime from the OS, enabling:
- The same ROCm APIs on both **Linux and Windows**
- `pip install` ROCm packages directly into virtual environments — just like CUDA

---

## Learning roadmap (5 sections)

| # | Section | Topics |
|---|---------|--------|
| 1 | **Environment Setup** | ROCm installation and configuration |
| 2 | **Deployment** | Running LLMs: vLLM, LM Studio, Ollama, llama.cpp |
| 3 | **Fine-tuning** | Model optimization on AMD hardware |
| 4 | **Infrastructure** | GPU programming and operator optimization |
| 5 | **References & Showcases** | Official resources, community projects |

---

## Target audience

- Developers who own AMD GPUs
- Those seeking structured ROCm learning paths
- Cost-conscious AI practitioners (AMD vs. NVIDIA pricing)
- Hands-on learners who prefer practical guides over theory

---

## Study checklist

- [ ] Understand ROCm vs CUDA: what differs, what is compatible
- [ ] Set up a ROCm environment (Section 1)
- [ ] Deploy at least one LLM locally using vLLM or Ollama (Section 2)
- [ ] Run a fine-tuning experiment on AMD hardware (Section 3)
- [ ] Explore GPU programming basics in ROCm (Section 4)
- [ ] Review community showcases for real-world use cases (Section 5)
