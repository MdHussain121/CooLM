# CooLM — Comprehensive Technical Audit

**Repository:** https://github.com/MdHussain121/CooLM (`main` @ `a717d3a67d61bc5668d9fdd4f666b09ff45f4c62`, last commit 2026-06-14)
**Audit date:** 2026-11-05
**Mode:** Read-only investigation and reporting. No source modifications. No fixes applied.

---

## 1. Executive Summary

**Verdict: PARTIAL PASS.**

CooLM's core — the Transformer architecture, the training math, and the model artifact itself — is genuinely correct and verified to an unusually high standard for a hobby-scale project. The parameter count is exactly as documented (41,476,480, verified three independent ways), weight tying is real, causal masking is provably exact, the KV-cache path is numerically correct, and the published checkpoint is authentic (step 9400, val loss 0.3262357… matching the documented 0.3262, byte-level file accounting exact). The dataset pipeline is fully reproducible: retraining the tokenizer on a regenerated 500K dataset reproduces the published `tokenizer.json` byte-for-byte.

Around that correct core, however, the audit found **8 High-severity defects (all confirmed by reproduction, several independently by multiple investigators)**, no Critical findings, plus 20 Medium, 20 Low, and 13 Informational findings:

1. **The published `model.onnx` is a different trained model than the published `best_model.pt`** (69/70 weight tensors differ; logit max-abs diff 8.79, argmax agreement 12.5%; a fresh re-export matches the .pt exactly). The server path users actually run serves a different personality than the PyTorch checkpoint.
2. **The production ONNX chat path has no context-length cap** — any message longer than ~45 tokens (≈250 chars) crashes ONNX Runtime and the server returns 500 with a misleading UI error.
3. **Both Windows launchers (`chat.bat`, `train_coolm.bat`) `cd` to a hardcoded nonexistent directory** — the documented entry points are dead on any fresh clone.
4. **The HF downloader writes directly to the final cache path with no size/hash verification** — any interrupted download permanently poisons the cache; the server then starts "healthy" with every chat returning 500.
5. **The documented headline metric (val loss 0.3262 / ppl 1.39) is a template-memorization figure**: 95.02% of validation outputs appear verbatim in training (100% template overlap), and the dataset itself has only ~752 distinct response templates (89.7% of outputs are duplicates at 500K).
6. **`generate()` silently truncates its return value** to the last ~65 tokens once generation crosses the 64-token window, and the response-extraction fallback then emits special-token literal fragments.
7. **`weights_only=False` pickle loading** of an internet-distributed 498MB checkpoint — an unnecessary code-execution surface (the safe path was verified to work).
8. **Zero tests, zero CI, no LICENSE file** despite MIT claims in three places.

Measured quality: the model is a competent, reliable template-retrieval chatbot inside its training distribution (100% EOS termination, zero leakage across 462 generations; relevance 1.96/2 on verbatim-style questions) with a sharp cliff outside it (0.32/2 out-of-domain). Measured performance: GPU decode is host-bound (138 tok/s, GPU util 23–27%); the ONNX engine's no-cache design penalty is negligible (0.94–1.74×) at this scale; the documented 6784 tok/s training metric is credible but counts only response tokens.

**Key risks:** artifact integrity (FIND-005), first-run experience (FIND-006/007/008), and misleading evaluation claims (FIND-004). **Key strengths:** verified-correct model math, authentic checkpoint, proven dataset reproducibility, honest engine parity when weights match, and an XSS-safe, mobile-functional UI.

---

## 2. Repository and Architecture Overview

**Actual file inventory (21 tracked files, ~4,375 lines):**

```
coolm/            config.py (29)   model.py (195)   prepare_data.py (113)
                  train.py (231)   inference.py (73)
generate_data.py  (2,221 — the synthetic conversation generator)
tools/            model_export.py (57)   onnx_chat.py (91)   server.py (184)
                  export_hf.py (108)     check_lengths.py (28)
docs/index.html   (753 — the entire browser UI)
chat.bat (23)     train_coolm.bat (52)   requirements.txt (4)   README.md (190)
assets/           coolm.png, chat_preview.png
```

**GitHub state:** default branch `main` (only branch), no tags/releases, no issues, no PRs, no CI workflows, no `.github/`, no tests, no LICENSE file. The local clone was clean and up to date with origin; the audit ran on a dedicated branch `audit/coolm-comprehensive-20261105` created from `main`.

**Hugging Face state:** `MdHussain121/coolm-42M` is public (best_model.pt 497.9MB, model.onnx 174.4MB, tokenizer.json 210.6KB, README, assets). The dataset repo `MdHussain121/coolm-500k` referenced by the model card is **not publicly accessible** (API returns 401) — see blocked items.

**Execution paths:**
- *Training:* `generate_data.py` → `coolm/prepare_data.py` (BPE tokenizer + [N,64] token/mask tensors) → `coolm/train.py` (95/5 split, AdamW, cosine LR, AMP, early stopping).
- *Inference (PyTorch):* `coolm/inference.py` → `model.generate()` with KV cache.
- *Inference (product):* `tools/server.py` (downloads/caches ONNX + tokenizer, serves `docs/`) → `tools/onnx_chat.py` (full re-run per token, no cache) → `docs/index.html` (fetch UI).

**Obsidian vault context:** `C:\Users\moham\Documents\Obsidian Vault` contains no CooLM/pigeon-related notes (searched "coolm", "pigeon", "Coo"). No prior project history to incorporate; vault untouched.

---

## 3. Model Architecture Audit

All items verified by code reading plus focused numerical experiments (Wave 1 architecture agent, re-confirmed by Wave 2 inference agent; scripts under `audit-work/wave1-arch/`, `wave2-torch/`).

| Documented spec | Measured | Verdict |
|---|---|---|
| Parameters 41,476,480 | 41,476,480 (hand + code; non-embedding 39,343,360) | **exact match** |
| 8 layers / 640 hidden / 10 heads / FFN 2560 GeLU | 8 / 640 / 10 (head_dim 64) / 2560, pre-LN | match |
| Vocab 3,269 | embedding rows exactly 3269, 0 padding waste; published tokenizer exactly 3269 entries, IDs contiguous 0–3268 | match |
| Max seq 64, learned positions, weight tying | verified; tying = same Parameter object, preserved through `load_state_dict` | match |

**Correctness experiments (all passed):**
- **Causality:** perturbing token t leaves logits at positions < t unchanged (fp64 delta 2.6e-15). No future-token leakage.
- **Padding:** trailing pads cannot influence real positions (causal structure); pads contribute zero loss (mask −100). Cost is compute only: ~49% wasted positions per sequence.
- **KV cache:** cached-vs-full logits max abs diff 1.9e-6 (fp32 GEMM-shape noise; fp64 clean); causal-bias slicing `bias[T_total−T : T_total, :T_total]` proven correct for full forward, chunked prefill, and single-step decode (low index T_total−T = T_prev ≥ 0 always).
- **Sliding-window rebuild** (model.py:160–172): re-prefill math verified 45/45 greedy steps against a fresh-window oracle. Position re-anchoring (window tokens get positions 0..62 again) is the only choice consistent with training; undocumented (FIND-051).
- **Weight-tying gradients** flow through both paths; optimizer covers all 41,476,480 params exactly once (32 decay / 36 no-decay groups; tied LM head lands in no-decay — FIND-052).
- **Initialization** normal(0, 0.02) for Linear/Embedding, no depth scaling — negligible at 8 layers (FIND-053).

**Defects found in this subsystem:** the generation API edge cases and the sliding-window return truncation (FIND-001, FIND-009, FIND-029, FIND-030, FIND-035, FIND-038). None affect the training path or the default inference path.

---

## 4. Synthetic Dataset Audit

The generator (`generate_data.py`, 2,221 lines) is deterministic (seed 42) and **provably reproducible**: regenerating 500K samples and retraining the BPE tokenizer reproduces the published `tokenizer.json` byte-for-byte (same 3,269-token vocab map, same 3,007 merges in the same order) — the strongest possible evidence that the published dataset is exactly this code's output.

**Measured at full 500K scale (coordinator-verified):**

| Metric | Value |
|---|---|
| Produced / requested | 500,000 / 500,000 (no drops; 0 retry-cap events; max retry depth 15) |
| Unique inputs | 500,000 (100%) — inputs are prefix(31) × base(745) × suffix(31) template compositions |
| Unique outputs | **51,691 (10.3%)** — ~752 response templates + 32 fluff sentences; top output ×1,019 |
| Unique (q,a) template pairs | 19,019 (avg 26.3 samples/pair) |
| Categories | 5.56%–9.90% per category (target 7.69%) — collision retry redistributes (FIND-014) |
| Schema | 100% conformant `{input, output, category}`, all lowercase, 0 empty/malformed |
| Tokenized lengths | max 46 tokens of 64 — **truncation never fires** on shipped data (FIND-037) |
| Loss-masked (response) tokens | mean 14.01/row (21.9% of 64) |

**Findings:** the "100% unique 500k dataset" (README:176) is true only for inputs (FIND-003, High); "balanced evenly across 13 categories" (README:172) breaks at scale (FIND-014, Medium); uniqueness retry can in principle silently emit duplicates (FIND-036, Low); word-substitution produces mildly garbled questions (FIND-047, Low); README's example first line is stale (FIND-050, Info).

**Validation contamination (FIND-004, High):** `train.py` splits 95/5 (seed 42) after the tokenizer has seen everything. Replicating the exact split on the reproduced 500K: **95.02%** of val outputs appear verbatim in train (94.1% of val loss-active tokens), 100% template overlap, 99.48% (q,a)-pair overlap. The documented val loss therefore measures template retrieval, not generalization. Only the 0%-exact-input-overlap axis is clean.

---

## 5. Tokenizer and Preprocessing Audit

Published `tokenizer.json`: 3,269 vocab entries (6 special + 256 byte alphabet + 3,007 merges), special tokens `pad=0 bos=1 eos=2 unk=3 im_start=4 im_end=5` — exactly matching `ModelConfig` (pad 0, bos 1, eos 2). ByteLevel pre-tokenizer (`add_prefix_space=False`), ByteLevel decoder, no normalizer (lowercasing lives in the data generator and inference — consistent). Round-trip encode→decode exact for empty strings, whitespace runs, newlines/CRLF, tabs, 300-char words, mixed case, punctuation, digits, and accented/CJK/Arabic/emoji input; `<|unk|>` unreachable (byte-level coverage).

Verified consistent across the pipeline: tokenizer-training string == tokenization string == `inference.py format_prompt` + response; BOS prepended as an ID and EOS appended as an ID identically in `prepare_data.py:73-74` and `inference.py:38`; joint-vs-split encoding identical on 50,000/50,000 samples. Loss-mask alignment to the shifted targets verified correct (`prepare_data.py:77` + `train.py:125-131`; decoded loss-active targets reproduce the assistant response exactly).

Findings: tokenizer trained on 100% of data including the future val split (FIND-049, Info — negligible here); dead truncation path (FIND-037, Low).

---

## 6. Training Pipeline Audit

The training loop is mechanically correct: loss masking and target shift are right; validation loss is token-weighted correctly (`vl.item() × n_active` summed / total active); AMP order (scale→backward→unscale→clip→step→update) is correct; the LR schedule (warmup 200 → cosine to min_lr) is valid for the shipped configuration; `StopIteration` cycling prevents dataset-exhaustion loops. The published checkpoint is authentic and complete: step 9400, val_loss 0.32623571…, full AdamW state, byte-level file accounting exact (497,935,062 B), all weights finite.

**Findings (no Critical):**
- No resume support; optimizer state saved but never read (FIND-013, Medium).
- Reproducibility: only the split is seeded; model init, dropout, and batch order are not — the exact 10K-step run cannot be reproduced (FIND-034, Low).
- GradScaler skip is silent; post-hoc Adam-step-counter forensics on the shipped checkpoint prove exactly 1 skip in 9,401 iterations (~0.011%), invisible in logs (FIND-033, Low).
- Summary hardcodes `max_steps` and early-stop reporting is off-by-one (FIND-031, Low); step-0 eval saves an untrained best-model candidate and the first update is a no-op at lr=0 (FIND-032, Low).
- Metric definition: "6784 tok/s" counts only masked response tokens (mean 14/row ≈ 1.7% of processed positions). Arithmetic is internally consistent (8,968,042 / 6784 = 22.03 min — matches the documented 22 minutes) (FIND-057, Info).
- Memory profile is fine: 512MB host tensors, ~332MB optimizer, fits 8GB VRAM with room to spare (FIND-016 in wave terms — no defect).

**Documented-metric plausibility:** the README's 22 min / 6784 tok-s / 8,968,042 tokens are mutually consistent and credible for an RTX 4060 (clean-loop equivalent measured at 11,792 active-token tok/s). The claims are not fabricated — but see FIND-004 for what the val loss does *not* measure.

---

## 7. Native PyTorch Inference Audit

The end-to-end path (`inference.chat()` → `generate()`) works correctly for the common case and the KV-cache math is exact. Findings:

- **FIND-001 (High):** `generate()` returns only the last ~65 tokens once the sliding window activates — the accumulator `input_ids` is mutated by window truncation (model.py:163) and returned as-is (model.py:195). Proven by exact tensor equality; two agents independently; latent at defaults (EOS always within ≤20 tokens in 60/60 real runs).
- **FIND-002 (High):** the extraction fallback `full_text[len(prompt):]` operates on the wrong offset (decoded text starts with the literal `<|bos|>`) and emits fragments like `>..........`, `os|>` when the marker is truncated away.
- **FIND-009 (Medium):** `temperature=0` crashes (CUDA device-side assert poisons the context; CPU multinomial RuntimeError); negative temperature silently anti-samples. Greedy decoding is unreachable through the public API (t=0.001 + top_k=1 is the only deterministic route).
- **FIND-010 (Medium):** normalization strips hyphens present in training data and removes the word "coo" that training uses pervasively; README:170's "OOD protection" claim is wrong for the coo removal.
- **FIND-011 (Medium):** prompts >64 tokens are silently head-truncated producing empty replies with no warning.
- **FIND-012 (Medium):** `weights_only=False` is unnecessary — the safe path (`add_safe_globals([ModelConfig])` + `weights_only=True`) loads the authentic checkpoint (verified by three agents).
- **Low/Info:** `max_new_tokens=0` still generates 1 token (FIND-029); batch>1 crashes at `.item()` (FIND-030); exception leaks eval mode (FIND-038); rfind takes the last marker (FIND-039); `<|unk|>` not cleaned (FIND-061); CWD-relative defaults make the entry point unrunnable as shipped (FIND-040).

State restoration is correct on the normal path; `@torch.no_grad()` coverage is complete; checkpoint loading is strict and clean with the real artifact.

---

## 8. ONNX Export and Inference Audit

**Export correctness:** `tools/model_export.py` is faithful — a fresh legacy export from `best_model.pt` is structurally identical to the published graph (same 845-node op sequence, attributes, initializer names, opset 14, dynamic batch/seq axes genuinely honored across B∈{1,4} × T∈{1,8,32,64}), differing by exactly 1 byte of producer string. **But the published artifact does not contain the published checkpoint's weights** (FIND-005, High — see Executive Summary). Export on torch ≥ 2.9 fails on fresh installs (missing `onnxscript`, unpinned requirements — FIND-015).

**Engine parity (same weights — fresh re-export):** PyTorch vs ONNX Runtime logits max-abs 7e-6…2.5e-5 with 100% argmax agreement on every tested shape; near-greedy chat token-identical 10/10. Prompt construction, BOS handling, EOS stopping, marker extraction, and cleaning are semantically identical between the engines; the only real differences are the no-cache recompute (measured penalty 0.94–1.74× — negligible at this scale, FIND-058), no context cap (FIND-006, High — crashes), and unseeded different RNGs (FIND-034).

**ONNX runtime findings:** no context-length guard (FIND-006); unseeded numpy sampling (FIND-034); fresh-clone default paths guaranteed missing with cryptic errors (FIND-040).

---

## 9. Model Quality Evaluation

**Methodology** (Wave 3): 77-prompt suite × 6 runs (462 generations) through the exact `CooInference.chat` code path on the real checkpoint — 63 in-domain prompts covering all 13 documented categories (36 verbatim training questions, 23 natural variations/unseen combinations, 4 ambiguous), 14 out-of-domain probes; 5 runs at defaults (t=0.4, top_k=30) + 1 near-greedy. Full outputs and rubric in `audit-work/wave3-perf/`.

**Results:**

| Metric | Result |
|---|---|
| EOS termination / empty / leakage | 462/462 (100%) / 0 / 0 |
| Response length | mean 12.1 tokens (range 4–17; training mean ~14) |
| Relevance — verbatim-style questions | **1.96 / 2** (91.7% fully on-topic answers) |
| Relevance — variations / unseen combos | **1.15 / 2** (29.6% full score) |
| Relevance — out-of-domain | **0.32 / 2** (64% scored zero; never breaks persona, deflects in-character) |
| Cross-run diversity | 3.23 distinct responses/prompt at defaults; 0/10 identical across repeat runs (seeds unset) |
| Template collisions | "what is bitcoin" → clock template; "do you like pizza" → umbrella template |

**Representative outputs:** in-domain answers are fluent and in-persona ("a friend is a bird who does not steal your seeds."; "the rain drips off the edge i watch it from my dry spot."). OOD inputs stay in-persona ("what is the capital of france" → "the sun is setting time to think about sleep soon."). Long/OOD requests produce fragments ("please write a long story about…" → "a quick nap." / "the beans.").

**Assessment:** against its stated objective (a narrow-domain pigeon persona), the model does exactly what it was built to do, reliably and safely — it is a learned template-matching router with fluent filler. It cannot generalize beyond recognizable training question shapes. This is a capability boundary, not a bug (the audit was instructed not to treat inherent small-model limits as defects); the *defect* adjacent to it is the undocumented scope and the misleading validation metric (FIND-004, FIND-028). No evaluation harness of any kind exists in the repo.

---

## 10. Performance and Hardware Analysis

Full tables and raw data: [CooLM-Performance-Benchmark-2026-11-05.md](CooLM-Performance-Benchmark-2026-11-05.md). Hardware verified: RTX 4060 Laptop 8GB (cc 8.9, 24 SMs), i7-12650H (10C/16T), 15.7GB RAM, torch 2.13+cu126, ORT 1.26 CPU-only.

| Engine | Load (s) | TTFT (ms) | Decode | ms/token | e2e typical | Peak mem |
|---|---|---|---|---|---|---|
| PyTorch GPU fp32 | 1.07 | 6.5 | 138 tok/s | 7.23 | 75–80 ms | 327MB VRAM / 1,753MB RSS |
| PyTorch CPU | 0.94 | 20–25 | 55–66 tok/s | 15.1–18.4 | — | ~1.3GB RSS |
| ORT CPU (no cache) | 0.49 | 13.9 | 70 tok/s | ~14.2 | 184 ms | 239MB RSS |

**Key measured conclusions:**
- **GPU decode is host-bound** (FIND-027): 170 kernel launches/step, 6.8ms host vs 1.24ms GPU work, util 23–27%. Prefill is flat ~5ms for T=10–64; step time is flat across context. GPU speedup over CPU is only 2.3× for this reason. Several-x improvement available via torch.compile/CUDA graphs without touching weights.
- **The suspected inefficiencies are not real:** `torch.cat` KV growth costs 0.033ms/step (0.4%); ONNX's full-recompute design costs 0.94–1.74× vs cached PyTorch at context 16–63 — per-op overhead dominates at 42M params (FIND-058).
- **ORT threading:** default is optimal; explicitly setting 16 logical threads is 2.1× *slower* on the hybrid CPU (FIND-059).
- **Training metric:** 6784 tok/s counts response tokens only; clean-loop equivalent 11,792 active-token tok/s (49,527 all-position) — the documented claim is credible (FIND-057).
- **Memory:** ORT is 7.3× lighter host-side than PyTorch — the right engine for constrained targets.

---

## 11. Local Server and Frontend Audit

**Server** (`tools/server.py`, live-tested against the real code on 127.0.0.1 ephemeral setups): binds localhost only (correct); survives malformed/stress input without crashing (invalid JSON, 5MB bodies, 20 rapid + 5 concurrent requests); static file serving works; path traversal blocked (4 vectors). Defects: non-atomic unverified downloader that accepts partial/corrupt caches as healthy (FIND-008, High); no context cap in the ONNX path it serves (FIND-006, High); single-threaded head-of-line blocking (FIND-016, Medium); no request-size cap + negative Content-Length DoS wedge (FIND-018, Medium); import-time side effects (FIND-017, Medium); raw exception text to clients (FIND-019, Medium); fixed port + forced browser auto-open (FIND-042, Low); unpinned mutable download target (FIND-041, Low); docs/ static exposure of the 174MB model (FIND-056, Info).

**Frontend** (`docs/index.html`, 753 lines — full audit + live browser testing with screenshots): functionally sound and **XSS-safe** (all text via `textContent`, verified with injected payloads in both directions); mobile layout works at 390×844 with no overflow; Enter-to-send and focus management correct. Defects: hardcoded "Active Engine" green badge regardless of backend state (FIND-021, Medium — the UI does *not* communicate actual engine state); no in-flight guard/timeout, eternal typing indicator on hang (FIND-022, Medium); accessibility gaps — zero aria-live/roles/labels, reduced-motion ignored, deprecated onkeypress (FIND-023, Medium); WCAG AA contrast failures on the user bubble and Send button (FIND-043, Low); "runs completely locally in your browser using ONNX" is false + external Google Fonts request (FIND-044, Low); autoscroll yank and unbounded DOM growth (FIND-045, Low); silent empty bubble on contract violation (FIND-046, Low); 29px touch targets (FIND-048, Low). Long messages surface FIND-006 as a misleading "start the server using chat.bat" error.

---

## 12. Security and Dependency Audit

- **Pickle loading (FIND-012, Medium):** `torch.load(weights_only=False)` on `coolm/inference.py:18` and `tools/model_export.py:13` against a 498MB checkpoint published on public HF. The safe path works with this exact checkpoint and was verified by three independent agents. HF itself labels the file "pickle, with flagged imports".
- **Download integrity (FIND-008 High + FIND-041 Low):** no checksum, no size verification, no timeout, no revision pinning, spoofed User-Agent. A MITM or interrupted transfer yields an undetectable corrupt model.
- **Local HTTP surface (FIND-018/019/060):** no request-size cap (memory allocated pre-validation), negative Content-Length wedges the single-threaded server, raw exception text (including ORT build paths) returned to clients; no CORS/CSRF posture (negligible at localhost); traversal blocked; 127.0.0.1 binding correct.
- **Dependencies (FIND-015, Medium):** 4 unpinned requirements; torch≥2.9 breaks export (missing `onnxscript`); `huggingface_hub` needed by `export_hf.py` is undeclared. No credentials hardcoded anywhere; `.env` ignored; the `--token` CLI pattern in export_hf exposes secrets in process listings (Low, inside FIND-024).
- **No SECURITY.md, no LICENSE file (FIND-020/025).**

---

## 13. Testing and Regression Coverage

**Existing tests: none.** No test files or directories, no CI (GitHub API confirms zero workflows), no pytest/tox/nox config, no SECURITY.md (FIND-025, Medium).

**Tests executed during the audit** (all under `audit-work/`, non-destructive, repo untouched — count: 60+ focused experiments across 6 agents plus coordinator):

| Area | Tests | Result |
|---|---|---|
| Param count / tying / shapes | hand + code + checkpoint load | PASS (exact) |
| Causality / pad-invariance | perturbation + fp64 | PASS (exact) |
| Cache-vs-full logit equality | 3 modes | PASS (≤1.9e-6) |
| Sliding-window rebuild vs oracle | 45 greedy steps | PASS (45/45) |
| generate() edge cases (t=0, neg t, top_k, batch, max_new 0, EOS-first, window) | boundary suite | 6 defects found (FIND-001/009/029/030 + 2) |
| Checkpoint forensics | byte accounting, Adam counters, config equality | PASS (authentic) |
| Dataset regeneration 500K + tokenizer retraining | full scale | PASS (byte-identical tokenizer); 3 defects found |
| Train/val overlap replication | exact split replication | CONFIRMED 95.02% |
| ONNX parity (same weights) | logits B×T matrix + near-greedy chat | PASS (100% argmax; 10/10) |
| Published-ONNX vs PT | logits | **FAIL — different models** |
| onnx_chat long input | live crash tests | **FAIL — RuntimeException** |
| Server contract + stress | 20+ live requests | 3 defects (FIND-006/018/019) |
| Downloader fault injection | truncated/interrupted transfers | **FAIL — partial accepted** |
| UI integration + XSS + a11y probes | live browser + static | 7 defects (FIND-021/022/023/043/044/045/046) |
| README claim table | 21 claims | 5 FALSE, 6 MISLEADING (§ claim table in Wave 3 report / FIND-024-026 etc.) |

**Blocked:** HF dataset artifact diff (repo inaccessible — mitigated by tokenizer byte-identity); ORT GPU benchmarks (no CUDA EP in onnxruntime 1.26); full retraining (out of scope).

---

## 14. Consolidated Findings Register

All 61 deduplicated findings with IDs, severities, evidence, and recommendations: **[CooLM-Findings-2026-11-05.json](CooLM-Findings-2026-11-05.json)** (machine-readable). Summary:

| Severity | Count | Notable |
|---|---|---|
| Critical | 0 | — |
| High | 8 | FIND-001…008 (generation truncation, extraction garbage, dataset duplication, val contamination, ONNX/PT artifact mismatch, ONNX context crash, dead launchers, downloader) |
| Medium | 20 | temperature=0 crash, normalization mismatch, long-prompt silence, pickle surface, no resume, category imbalance, deps unpinned, single-threaded server, import side effects, request-size DoS, exception leak, no LICENSE, UI status lie, a11y, export_hf, zero tests, missing export step, host-bound GPU, generalization cliff |
| Low | 20 | API edge cases, off-by-one reporting, step-0 save, scaler skip, reproducibility, retry cap, dead truncation, paths, port/browser, contrast, meta claims, scroll, empty bubble, substitutions, touch targets |
| Info | 13 | tokenizer-on-val, stale README example, window semantics, weight-decay convention, init, NaN edge, RAM streaming, static exposure, metric definition, measured non-issues, ORT threads, CORS, tooling gaps |

Severity was calibrated against the assignment's definitions; no inflation. Every High finding was reproduced by at least one agent and re-verified by the coordinator (source reading and/or independent experiment); FIND-001, FIND-004, FIND-005 were confirmed by two independent agents **plus** coordinator replication.

---

## 15. Prioritized Improvement Roadmap

**P0 — Correctness and reliability (make existing functionality trustworthy)**
1. **Re-publish `model.onnx` from `best_model.pt`** and add a pt/onnx consistency check to `export_hf.py` (fixes FIND-005; effort: minutes; verification: 70/70 identical initializers, 10/10 near-greedy parity).
2. **Cap the ONNX context**: clamp input to the last 64 tokens each step and cap `max_new_tokens` (fixes FIND-006; ~5 lines; verify 300-word message → 200).
3. **Fix the launchers**: `cd /d "%~dp0"`, check errorlevel, skip redundant installs, add the export step to `train_coolm.bat` (fixes FIND-007/026).
4. **Harden the downloader**: temp file + Content-Length check + pinned sha256 + atomic rename + startup cache validation (fixes FIND-008/041).
5. **Fix `generate()` return truncation** with an immutable accumulator (fixes FIND-001/002 together).
6. **Adopt `weights_only=True` + `add_safe_globals([ModelConfig])`** (fixes FIND-012).
7. **Add a LICENSE file** (fixes FIND-020). **Add the minimal test suite + CI** per FIND-025's mapping (prevents regression of all of the above).

**P1 — Model and inference quality**
8. Split at topic level (hold out 1–2 categories) and relabel the val metric as template-retrieval accuracy (FIND-004/028).
9. Enrich response generation (couple response slots to question slots; larger pattern pools) to push distinct outputs beyond ~52K (FIND-003).
10. Align input normalization with training data: keep hyphens, drop the coo removal, collapse whitespace (FIND-010); warn on over-length prompts (FIND-011); guard temperature/top_k/batch contracts (FIND-009/029/030/038).
11. Add an optional seed to `train()` and `generate()` (FIND-034); add resume support or drop the unused optimizer state (FIND-013).

**P2 — Performance and resource efficiency**
12. `torch.compile` (or CUDA-graph) the decode step to attack the 6.8ms host overhead vs 1.24ms GPU work (FIND-027); do **not** bother with the KV-cache `cat` or ONNX recompute — measured irrelevant (FIND-058). Leave ORT threads at default (FIND-059). Threading + request cap for the server (FIND-016/018).

**P3 — Architecture and maintainability**
13. Move server side effects under `__main__`; resolve paths relative to the package root; CLI flags for port/browser (FIND-017/040/042).
14. UI: aria-live/labels/reduced-motion, real status probing, in-flight guard + timeout, contrast fixes (FIND-021/022/023/043/045/046).
15. Document: tok/s metric definition (FIND-057), sliding-window semantics (FIND-051), weight-decay convention (FIND-052), quantified scope limits in the model card (FIND-028), regenerate the stale README example (FIND-050).

---

## 16. Verification and Evidence

- **Environment:** Windows 11, Python 3.12.10, torch 2.13.0+cu126 (CUDA 12.6), onnxruntime 1.26.0 (CPU-only), tokenizers 0.22.2, RTX 4060 Laptop 8GB (driver 616.64), i7-12650H, 15.7GB RAM, 109GB free disk.
- **Artifacts:** `best_model.pt` (sha256 recorded during download; verified authentic), `model.onnx` (published), fresh re-export, `tokenizer.json` — all downloaded to `audit-work/hf/` outside the repo.
- **Subagent evidence:** `audit-work/wave1-data/`, `wave1-arch/`, `wave2-torch/`, `wave2-onnx/`, `wave3-perf/`, `wave3-ui/` (scripts, logs, JSON results, 8 UI screenshots).
- **Coordinator replications:** ONNX/PT logit divergence (maxabs 8.788 / agreement 12.5% vs fresh-export 0.000 / 100%); dataset stats (51,691 unique outputs, 89.66% duplicate rate, top ×1,019, category 5.56–9.90%); val overlap (95.02%); chat.bat/server.py/bat files read directly.
- **Limitations:** dataset artifact itself inaccessible (reproducibility established via tokenizer byte-identity instead); ORT GPU unmeasurable on this machine; ONNX published-artifact provenance (which run produced it) unknown; training metrics accepted as self-reported but cross-checked for internal consistency.

---

## 17. Final Verdict

**PARTIAL PASS.**

Rationale: The model core, training math, and architecture are verified correct and authentic, and the dataset pipeline is provably reproducible — this is a genuinely working small-LM system, and nothing found rises to Critical. However, multiple confirmed High defects remain in the *product layer* that users actually touch: the shipped ONNX artifact is a different model than the shipped checkpoint, the production inference path crashes on ordinary long messages, both documented entry points are dead on fresh clones, the download pipeline can silently poison itself, and the headline evaluation metric is not a generalization measure. The project's educational honesty would also benefit from aligning its claims (uniqueness, balance, license, "runs in browser") with measured reality. All identified issues are fixable within the P0 roadmap without retraining; the P0 set is roughly a day of focused work.

*Report-only diff: the only changes on branch `audit/coolm-comprehensive-20261105` are the files in `audit-reports/`.*
