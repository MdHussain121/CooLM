# CooLM — Inference Evaluation Report

**Date:** 2026-11-05 · **Checkpoint:** `best_model.pt` (step 9400, val_loss 0.32623571) · **Tokenizer:** published `tokenizer.json` (3,269 vocab)
**Companion documents:** [Comprehensive Audit](CooLM-Comprehensive-Audit-2026-11-05.md) · [Findings JSON](CooLM-Findings-2026-11-05.json) · Evidence: `audit-work/wave2-torch/`, `audit-work/wave3-perf/`, `audit-work/wave2-onnx/`

## 1. Methodology

- **Path traced:** user input → `chat()` normalization (lowercase, strip non-`[a-z\s]`, remove `\bcoo\b`) → `format_prompt` → tokenizer encode → BOS-ID prepend → `generate()` (KV cache, sliding window, top-k sampling) → decode (`skip_special_tokens=False`) → marker extraction (`rfind`) → cleanup.
- **Quality suite:** 77 prompts × 6 runs = 462 generations through the exact `CooInference.chat` code path on the real trained model: 63 in-domain (all 13 categories; 36 verbatim training questions, 23 variations/unseen combos, 4 ambiguous), 14 OOD probes. 5 runs at defaults (t=0.4, top_k=30) + 1 near-greedy (t=0.001, top_k=1). Full outputs: `wave3-perf/outputs_readable.txt`, `quality_outputs.json`.
- **Engine comparison:** identical int64 inputs into PyTorch (best_model.pt), published `model.onnx`, and a fresh re-export of best_model.pt; near-greedy (t=0.001, top_k=1) chat parity on 10 prompts.

## 2. Quantitative results (462 generations)

| Metric | Result |
|---|---|
| EOS termination | 462/462 (100%), max 17 response tokens |
| Empty responses / special-token leakage / window hits | 0 / 0 / 0 (final seq len mean 28.4, max 50 of 64) |
| Response tokens | mean 12.1 (training mean ~14) |
| Relevance (rubric 0–2): verbatim-style / variations / OOD | **1.96 / 1.15 / 0.32** |
| Cross-run diversity | 3.23 distinct responses/prompt; 0/10 identical on repeat (seeds unset) |
| Corpus reuse | 213 distinct strings over 385 default runs; 31 strings reused for ≥2 prompts |

## 3. Representative outputs (defaults: t=0.4, top_k=30)

**Good (in-domain and unseen-combo):**

| Prompt | Response |
|---|---|
| is rain good for pigeons | the rain drips off the edge i watch it from my dry spot. |
| do you have bird friends | a friend is a bird who does not steal your seeds. |
| is there a green parrot on your roof | parrots come to the neem tree near the roof every morning. ("neem" appears in no training question) |
| do you like crows | the crow is clever but loud always cawing. |
| hi | i do not know you are you from another roof. |

**Template collisions / degenerate:**

| Prompt | Response |
|---|---|
| what is bitcoin | the watch goes tick tick tick annoying sound. |
| is the owl still watching you | the humans stare at the glowing box… (TV template) |
| do you like pizza | you carry the umbrella for rain… |
| please write a long story about a pigeon who sails… | "a quick nap." / "the beans." / "the smoke is a thing on it near and fills their day." |
| what is 2 plus 2 (normalizes to "what is  plus") | i do not need a resume i am a pigeon that is enough. |

**Boundary-condition failures (EOS disabled / long inputs):**
- 60-token prompt + 40 new tokens → returned 65 tokens (35 dropped) — FIND-001.
- Marker pushed out of window → user-visible `>..........`, `nd|>..........`, `os|>`, or empty — FIND-002.
- Prompt >64 tokens → silent immediate EOS → empty reply — FIND-011.
- temperature=0 → CUDA device-side assert (context poisoning) / CPU RuntimeError — FIND-009.
- Determinism: greedy (t=0.001+top_k=1) is 20/20 reproducible; true greedy unreachable via API (t=0 crashes).

## 4. PyTorch vs ONNX comparison

**Same weights (fresh re-export of best_model.pt) — engine parity is excellent:**

| B | T | max abs logit diff | argmax agreement |
|---|---|---|---|
| 1 | 1 | 7.2e-06 | 100% |
| 4 | 1 | 5.7e-06 | 100% |
| 1 | 8 | 8.6e-06 | 100% |
| 4 | 32 | 2.5e-05 | 100% |
| 1 / 4 | 64 | 1.1e-05 / 1.6e-05 | 100% |

Near-greedy chat: **10/10 token-identical**; prompt construction, BOS handling, EOS stopping, marker extraction, and cleaning are semantically identical. Differences: no-KV-cache recompute (cost negligible, 0.94–1.74×), **no context cap → crash at 65 tokens (FIND-006)**, unseeded different RNGs.

**Published model.onnx — different model (FIND-005):** logits vs best_model.pt max-abs 8.79–15.2 across shapes, argmax agreement 12.5–47%; near-greedy chat matches PyTorch in **1/10** prompts (e.g. "hi": PT/fresh → "i do not know you are you from another roof."; published → "i do not trust monkeys they move too fast and grab too much."). The published ONNX is itself a trained, coherent pigeon model — just not the one the .pt and the documented metrics describe.

## 5. Quality limitations and verdict

- Capability boundary: reliable template-retrieval inside ~745 training question shapes; sharp cliff outside (relevance 1.96 → 1.15 → 0.32). Inherent to the training data design (FIND-003), not a bug; undocumented scope is the defect (FIND-028).
- Validation metric 0.3262/ppl 1.39 measures template memorization (95.02% verbatim overlap) — it predicts nothing about the variation/OOD numbers above (FIND-004).
- Generation reliability at defaults is excellent: the model never leaks tokens, never runs away, and always terminates. The found inference defects live in edge paths (window overflow, t=0, batch>1, >64-token prompts) rather than the default chat experience.
