# CooLM — Performance Benchmark Report

**Date:** 2026-11-05 · **Companion documents:** [Comprehensive Audit](CooLM-Comprehensive-Audit-2026-11-05.md) · [Findings JSON](CooLM-Findings-2026-11-05.json) · Raw data: `audit-work/wave3-perf/` (env.json, pt_gpu.json, pt_cpu.json, ort.json, bottleneck.json, benchmark_summary.json)

## 1. Environment (verified, not assumed)

| Item | Value |
|---|---|
| OS / Python | Windows 11, Python 3.12.10 |
| GPU | NVIDIA RTX 4060 Laptop, 8,188 MiB, cc 8.9, 24 SMs, driver 616.64 |
| CPU / RAM | Intel i7-12650H (10C/16T), 15.7 GB usable |
| torch / CUDA | 2.13.0+cu126 (CUDA 12.6, cuDNN 91002); default num_threads = 10 |
| onnxruntime | 1.26.0 — **CPUExecutionProvider only** (no CUDA EP; GPU ORT not measurable) |
| Protocol | 3 warm-up iterations, median of ≥5 runs; `torch.cuda.synchronize` boundaries; separate load/prefill/decode phases |

## 2. Benchmark tables

| Engine | Load (s) | Prefill T=15 (ms) | TTFT (ms) | Decode tok/s | ms/token | e2e typical chat() (ms) | Peak memory |
|---|---|---|---|---|---|---|---|
| PyTorch GPU fp32 | 1.07 (tok 0.006 + load 0.37 + state 0.57 + to-cuda) | 5.2 (T=10/30/64: 5.15/5.43/5.16 — flat) | 6.51 | 138 (166 sustained in-loop) | 7.23 | 75–80 (38.5 short-answer prompt) | 327 MB VRAM peak alloc (311 MiB nvidia-smi); 1,753 MB peak RSS |
| PyTorch CPU | 0.94 | prefill T=64 = 39.7 | 20–25 | 55–66 | 15.1–18.4 (thread-dependent) | — | ~1.3 GB RSS |
| ORT CPU (no KV cache) | 0.49 (session create) | 13.9 (context ~12) | 13.9 | 70 (13 tok / 184 ms) | ~14.2 | 184 (full loop to EOS) | 239 MB RSS (7.3× lighter than PyTorch) |

GPU speedup vs CPU decode: **2.3× only** (host-bound, see §4).

## 3. ONNX no-KV-cache recompute cost curve

Cost of producing the next token at context length L (one full `session.run`), vs PyTorch CPU cached decode (15.4 ms/token):

| Context L | ORT re-run (ms) | PT CPU uncached fwd (ms) | Penalty vs cached PT |
|---|---|---|---|
| 16 | 18.4 | 15.2 | 1.19× |
| 32 | 14.5 | 24.5 | 0.94× |
| 48 | 17.7 | 24.5 | 1.14× |
| 63 | 26.9 | 34.6 | 1.74× |

The O(T) full-recompute design is **not** a meaningful problem at 42M params — per-op dispatch overhead dominates (FIND-058). Published model.onnx cost is identical to the fresh export (L32: 14.9 vs 14.5 ms).

## 4. Bottleneck analysis (measured, not estimated)

- **GPU decode is host-bound:** torch profiler over 20 steps → **~170 CUDA kernel launches/step; 6.8 ms self-CPU vs 1.24 ms self-CUDA per step** → ~82% of decode wall time is Python/dispatch; GPU utilization 23–27%. Forward-only 4.63 ms vs forward+topk+softmax+multinomial 4.80 ms (sampling adds 3.6%).
- Decode step time is flat across context (5.0/4.9/4.7/5.1 ms at ctx 16/32/48/63) — no cache-growth penalty.
- **`torch.cat` KV growth (model.py:30-31) costs 0.015–0.033 ms/step (~0.4%)** — quantitatively irrelevant; not an optimization target.
- CPU path is also per-op-overhead dominated (forward-only ≈ full step; thread count changes decode by <15%).
- **ORT threading:** default(all) 17.8/178 ms (L48 re-run/e2e); 8 threads 25.8/213; 4 threads 23.6/232; **16 threads 38.9/375 (2.1× slower — oversubscription on hybrid CPUs)**; 1 thread 64.1/464. Default is optimal; never set intra_op to logical-core count on this CPU (FIND-059).

## 5. Training-throughput context (documented metric reconciliation)

- The documented **6784 tok/s** counts only masked response tokens (train.py:151-156): mean 14.01 active tokens per 64-token row ≈ 1.7% of processed positions.
- Arithmetic check: 8,968,042 documented tokens ÷ 6784 = 1,321.9 s = **22.03 min** — matches the README's "22.0m (1322s)" exactly; internally consistent and credible.
- Clean-loop equivalent measured on this GPU: 81.4 ms/step at batch 64 (AMP, fwd+bwd+AdamW) → **11,792 active-token tok/s** (49,527 all-position tok/s). The documented figure is plausible for the shipped loop (which adds dataloader/eval overhead).
- Do **not** compare 6784 tok/s to inference decode tok/s (138 GPU / 70 ORT) — different metrics.

## 6. Recommended optimizations (evidence-ranked)

1. **Attack host-side launch overhead, not math:** `torch.compile` the decode step or CUDA-graph it (170 launches/step, 82% host time — several-x headroom without touching weights). Expected gain: decode toward GPU-work-bound ~1.5–2 ms/token (to be measured after implementation; not claimed here as achieved).
2. **Batch decode if serving >1 user** (single-user latency is already fine at 75–80 ms e2e).
3. **Use ORT for constrained targets** — 7.3× lighter host memory (239 MB vs 1,753 MB), 184 ms e2e; for lowest latency use PyTorch GPU + item 1.
4. **Do not** optimize the KV-cache `cat` or ONNX recompute (measured 0.4% and 0.94–1.74×); **do not** hand-tune ORT threads.
5. Server: request-size cap + threading to prevent one slow request from freezing the UI (FIND-016/018) — correctness first, then performance.

## 7. Resource summary

| Scenario | Host RAM peak | VRAM peak | Notes |
|---|---|---|---|
| PyTorch GPU load+generate | 1,753 MB RSS (torch+CUDA DLLs, 475 MB checkpoint) | 311–327 MB | fits 8 GB comfortably |
| PyTorch CPU | ~1.3 GB | — | 2.3× slower decode |
| ORT CPU session+generate | 239 MB | — | right engine for lightweight targets |
