# CooLM — Post-Audit Remediation Report

**Date:** 2026-11-05
**Repository:** https://github.com/MdHussain121/CooLM
**Branch:** `fix/coolm-comprehensive-remediation-20261105` (created from `audit/coolm-comprehensive-20261105`, HEAD `a717d3a`)
**Mode:** VERIFY → FIX → TEST → REVIEW → REPORT. Six sequential implementation assignments (one active subagent at a time), each reviewed and independently re-verified by the coordinator before the next began. Final regression pass (Assignment 7) performed by the coordinator, not delegated.

---

## 1. Executive Summary

| | |
|---|---|
| Initial verdict (2026-11-05 audit) | PARTIAL PASS — 61 findings (0 critical / 8 high / 20 medium / 20 low / 13 info) |
| **Final verdict** | **PASS WITH OBSERVATIONS** (see §14) |
| Findings FIXED | **50** |
| Findings DEFERRED (with justification) | **3** (FIND-003, FIND-014, FIND-047 — dataset-content changes would break seed-42 byte reproducibility; all three are now honestly documented) |
| Findings INFORMATIONAL / no change required | **8** (FIND-027, 049, 052, 053, 055, 056, 058, 059) |
| Findings unresolved | **0 silent** — every finding has an explicit disposition; the single externally-blocked item (publishing the corrected ONNX, FIND-005) is flagged, with the verified replacement artifact and exact publication steps produced locally |

The verified model core (architecture, checkpoint, training math) is untouched: checkpoint SHA-256 `6db4cb4cf7ff64c160f2f1eb5503558afb9cbcfc0d940f8a0f5c3a9f914fd60e` confirmed unchanged after every assignment. No retraining. No dataset regeneration. No push to GitHub. No upload to Hugging Face.

Regression suite: **122 tests, all passing** (`python -m pytest tests/ -q`, ~48 s), covering generation contracts, artifact parity, downloader fault injection, server contract, checkpoint safety, evaluation splits, UI static checks, resume, and packaging.

---

## 2. High-Severity Remediation (8 findings)

### FIND-005 — Published ONNX ≠ published PyTorch checkpoint — FIXED (code + artifact; publication pending owner action)
- **Fix:** `tools/verify_artifacts.py` (new) verifies checkpoint↔ONNX parity (deterministic seeded inputs, B∈{1,4}×T∈{1,8,32,64}; PASS = 100 % argmax agreement AND max-abs logit diff < 1e-3 — matching weights measure ≤ 2.5e-5, the mismatched published pair measures 8.8–15.2). `tools/model_export.py` exports with the legacy exporter (`dynamo=False`, required on torch ≥ 2.9), writes a provenance sidecar (`<onnx>.provenance.json`: checkpoint SHA-256, step, val_loss, config, opset, timestamps) **only after a successful export**, and deletes the ONNX + sidecar if the self-parity check fails (fail-closed). `tools/export_hf.py` refuses to publish unless the sidecar's checkpoint hash matches the checkpoint being uploaded AND live parity passes.
- **Verified replacement artifact (local, not uploaded):** `audit-work/remediation-artifacts/model.onnx`
  - SHA-256 `ce173d489877e6d30f3f47391ad40a36ce30e7ce4a56ea5a718ebee73ac5fb40`, 174,399,571 bytes
  - Exported from `best_model.pt` (step 9400, sha256 `6db4cb4c…`); provenance sidecar hash chain verified; byte-identical to the audit's independent `fresh_legacy.onnx` re-export
  - Parity vs checkpoint: max-abs diff ≤ 2.19e-05, **argmax agreement 100 % on all 8 shapes**; tokenizer pairing PASS (vocab 3,269 = config 3,269; bos/eos/pad ids match)
  - Negative control: the currently published `model.onnx` (sha256 `f354a06b…`) **fails** the check (max-abs 9.85–15.14, argmax agreement 0–50 %) — the gate provably catches FIND-005
- **Honest interim behavior:** the pinned downloader manifest references the currently-published (mismatched) ONNX with an explicit startup notice, and `PUBLICATION.md` gives exact owner steps: publish the replacement ONNX + sidecar, then update `RELEASE_FILES` (revision + sha256 + size) in `tools/server.py`. **No remote action was taken — publishing requires owner authorization.**

### FIND-006 — ONNX inference crashes on ordinary long messages — FIXED
- Pipeline: `CooONNXInference` reads `max_seq_len` from the provenance sidecar (fallback 64 with a printed note), truncates user text token-wise keeping the most recent tokens with the chat scaffold preserved (same helper as the PyTorch engine), and feeds only the last `max_seq_len` tokens every step (sliding window). The effective input length can never exceed the context for any input length, including mid-generation.
- Server: JSON contract validation (400/413/415), engine errors → generic 500 with server-side logging; no ONNX Runtime exception can reach a client.
- **Evidence:** 1,422-character message through the live server (real engine, isolated staging dir): **HTTP 200**, no crash (audit repro: guaranteed 500). Fed-length recording proves every session feed ≤ 64 tokens. Empty input → `"coo..."`; malformed JSON → 400.

### FIND-007 — Both Windows launchers dead — FIXED
- Both `.bat` files now `cd /d "%~dp0"` (quoted, space-safe), locate `python` with a `py` fallback, install dependencies only when the imports fail, check `errorlevel` at every step, and never reference a developer-specific path.
- `train_coolm.bat` gained **Step 4/4: ONNX export** (FIND-026) so a successful pipeline produces `docs/model.onnx` + `data/tokenizer.json` that chat option 2 needs.
- **Evidence:** fresh working-tree copy (39 files, no artifacts): from an arbitrary CWD **and from a directory path containing spaces**, `chat.bat` shows the menu, resolves the script directory, and exits cleanly on the no-local-model path. Control-flow stub test of `train_coolm.bat` proves all four steps run in order and a step failure produces the FAILED message + non-zero exit. Headless server run (`--port 8123 --no-browser --model local`) served the UI and answered short + 300-word chat POSTs with fluent responses.

### FIND-008 — Download pipeline can poison its cache — FIXED
- `tools/server.py` `download_file()`: streams to a temp file **in the destination directory**, verifies HTTP status, Content-Length, expected size, and SHA-256 **while streaming**, and promotes via `os.replace` only after all checks pass; any failure deletes the temp file, classifies the error (incomplete / size mismatch / checksum mismatch / HTTP / network), retries up to 3 attempts with backoff, and never leaves a `.part` file. Startup cache validation classifies MISSING / SIZE_MISMATCH / CHECKSUM_MISMATCH / VALID and re-downloads invalid caches. Manifest pins the HF **revision** `40c37e23…` with audit-verified sha256+size constants (never fabricated) and an honest User-Agent.
- **Evidence:** 10-test fault-injection matrix — clean download, truncated transfer, tampered payload, size mismatch, 404/500, hanging server (bounded timeout), corrupt cache rejected+replaced, retry-after-failure, failed re-download preserves the previous valid cache, manifest pinning — all pass.

### FIND-004 — Validation metric is template memorization — FIXED (methodology + honest relabeling)
- **New tool `tools/evaluate.py`**: replicates the shipped split (seed 42, 95/5), and builds two leakage-controlled splits — **response-template-disjoint** (whole exact-output groups held out) and **topic-disjoint** (whole categories held out) — each with a per-split leakage table (% val outputs verbatim in train, exact-input overlap, template overlap, token counts). Deterministic; JSON report with hashes + explicit limitations.
- **Coordinator evidence (500K regenerated dataset + real checkpoint, 134 s):**

  | Split | Val rows | Outputs verbatim in train | Val loss | PPL |
  |---|---:|---:|---:|---:|
  | shipped (in-distribution template retrieval) | 25,000 | **95.02 %** (independently reproduces the audit figure) | **0.3262** | **1.39** |
  | response-template-disjoint | 25,426 (5.09 %) | **0.00 %** | 0.3309 | 1.39 |
  | topic-disjoint (confused + food held out) | 97,178 (19.44 %) | **0.00 %** | 0.3414 | 1.41 |

  The shipped split reproduces the documented 0.3262/1.39 to four decimals — validating the replication end-to-end — while the honest splits measure strictly higher loss with zero verbatim overlap. The modest gap is itself informative (outputs are dominated by ~752 templates, so next-token prediction stays largely stylistic) and is reported as measured, nothing tuned. Report: `audit-work/remediation-artifacts/evaluation-report.json`.
- **Documentation:** README labels 0.3262/1.39 as in-distribution template-retrieval performance; the HF model-card metric names now say "template-retrieval, in-distribution" and the card gained a quantified Limitations section. Dataset, tokenizer, and checkpoint are untouched.

### FIND-001 — generate() silently truncates its return — FIXED
- Immutable `full` accumulator (returned) separated from the bounded `window` (consumed by the model); the audit-verified rebuild math (re-prefill 63 + step at position 63, positions re-anchored) is preserved exactly and now documented (FIND-051).
- **Evidence on the real checkpoint (EOS disabled, audit repro shape):** prompt 10 + max_new 45 → returns **55 tokens** (the bug returned a 65-cap), prefix `torch.equal` to the input; prompt 10 + 200 → 210 tokens, prefix intact across multiple window rebuilds. Contract tests: 26+45 → 71; 40+100 → 140; 30+200 → 230.

### FIND-002 — Extraction fallback emits special-token fragments — FIXED
- Response extraction is now by **token index** (`output_ids[0, prompt_len:]`, decoded with `skip_special_tokens=True` plus literal-strip defense-in-depth incl. `<|unk|>`); the character-offset `rfind` path is gone from both engines (also resolves FIND-039). Fragment-producing inputs (`>..........`, `os|>`) are structurally impossible.

---

## 3. Inference Corrections (task §6)

| Item | Disposition | Evidence |
|---|---|---|
| temperature=0 crash (FIND-009) | FIXED — t=0 → greedy argmax (documented convention); t<0 or non-finite → `ValueError`; sub-1e-6 divisors floored so fp32 saturation can never NaN the softmax (coordinator addition) | greedy determinism tests; 1e-300 regression test; real-checkpoint greedy identical across runs |
| Batch generation (FIND-030) | FIXED — per-sequence EOS state; finished rows append pad; uniform shapes; early stop when all rows finish | batch 2/4 shape+prefix tests; forced-EOS per-row test ([eos, pad×11] vs continuing row) |
| max_new_tokens ≤ 0 (FIND-029) | FIXED — returns input unchanged (PyTorch) / empty response, documented (ONNX) | `torch.equal` tests |
| Context handling (FIND-011) | FIXED — chat() truncates user text token-wise (tail kept), scaffold preserved, `UserWarning` emitted; model never head-truncates the scaffold | 300-token-prompt tests (warning + prompt ≤ max_seq_len); real checkpoint: 192-token user text → warning + relevant tail answer |
| Model state (FIND-038) | FIXED — try/finally restores training mode on any exception | monkeypatched-raise test |
| forward() cache assert (FIND-035) | FIXED — reviewed deviation: two asserts (`cached_len + T ≤ max`, `start_pos == cached_len`) because the audit's literal `cached+T+start` formula double-counts positions and would reject the verified rebuild step | audit's overflow repro raises immediately; all verified call sites pass |
| Input validation | FIXED — TypeError/ValueError with named arguments before any state change, both engines | type-validation tests |
| Real-checkpoint sanity | fluent pigeon-persona answers on all prompts tried; greedy deterministic; ONNX engine now produces the same personality as PyTorch (audit: 1/10 match on the old published ONNX) | e.g. "do you have bird friends" → "a friend is a bird who does not steal your seeds." (exact audit reference) |

## 4. ONNX and PyTorch Artifact Consistency
Covered in §2/FIND-005. Provenance chain re-verified at final regression: sidecar `checkpoint_sha256` == `best_model.pt` sha256 (`6db4cb4c…`), step 9400, opset 14, torch 2.13.0+cu126; replacement ONNX sha256 `ce173d48…`; parity PASS on all shapes; published artifact remains the documented negative control. Server-side pairing is guaranteed by the pinned revision + hashes.

## 5. Downloader and Launcher Corrections
Covered in §2/FIND-008 and §2/FIND-007 (+FIND-026 export step, FIND-041 revision pinning, honest UA). Server additionally: `ThreadingHTTPServer` + inference lock (FIND-016), import-safe module (FIND-017, subprocess test proves no I/O on import), 1 MiB body cap → 413, negative/non-integer Content-Length → 400 without wedging (FIND-018), generic 500 bodies with server-side logs (FIND-019), `--port/--host/--no-browser/--model` CLI + clean bind-failure exit (FIND-042), Content-Type → 415 (FIND-060), `GET /health` for the UI status badge.

## 6. Evaluation and Documentation Corrections
- `tools/evaluate.py` + numbers in §2/FIND-004; README now states tok/s counts masked response tokens only (FIND-057) and carries a "Model Scope & Evaluation" section with the measured relevance cliff 1.96 → 1.15 → 0.32 (FIND-028).
- README claim corrections (audit claim table): "runs completely locally in your browser" → server-side inference (FIND-044); "balanced evenly across 13 categories" → measured 5.5 %–9.9 % at 500K (FIND-014); "guarantees exactly 100 % uniqueness" → empirical 100-collision cap, inputs-only (FIND-003/036); "OOD protection" normalization claim → truthful description post-FIND-010; stale dataset example → real seed-42 first line (FIND-050); export_hf "Push dataset and model" → model artifacts only (FIND-024).
- Model card: honest metric labels + quantified Limitations (verbatim overlap, ~752 templates / 51,691 distinct outputs / 89.7 % duplicates, relevance cliff, artifact-verification pointer).

## 7. Checkpoint Safety (FIND-012)
- New single authoritative loader `coolm/checkpoint.py`: `weights_only=True` ALWAYS + explicit `add_safe_globals([ModelConfig])` allowlist; **no unsafe fallback anywhere** (repo-wide static test enforces it); typed error hierarchy (NotFound / Format / Compatibility incl. strict key/shape mismatches and NaN/Inf weights, with the causal-mask `-inf` buffers exempt); missing-config checkpoints fall back to defaults with a loud warning (self-correcting via strict load). Every consumer routes through it (inference, export, verification); `coolm/train.py` tensor loads were already safe.
- **Evidence:** real 498 MB checkpoint loads via the safe path (step 9400) and chats end-to-end; untrusted-pickle `__reduce__` payload is rejected **without executing** (marker-file regression test); structurally corrupted archives raise `CheckpointFormatError`. Documented defense boundary: byte flips confined to tensor payload leave a structurally valid archive with finite garbage weights — that tamper class is the sha256-pinning layer's job (FIND-008/FIND-005), and is disclosed here rather than hidden.

## 8. Medium and Low Finding Remediation
Complete disposition of all 61 IDs (incl. the High set above): see the machine-readable register **`CooLM-Remediation-Status-2026-11-05.json`**. Highlights beyond §2–§7:
- **Training loop:** resume support (`--resume`, `checkpoints/last_model.pt` with optimizer/scaler/step/patience state — FIND-013); summary prints actual completed steps (FIND-031); step-0 eval/save eliminated (FIND-032); GradScaler skips counted and logged (FIND-033); optional `--seed` (FIND-034); zero-active-token batch guard (FIND-054). Resume smoke test: 2-step run → resume → continues at step 2, loss continuous, first `best_model.pt` only after a real eval.
- **Data tooling:** over-length sequences now fail loudly in `prepare_data.py` (FIND-037); retry-cap telemetry in `generate_data.py` (FIND-036) — generation output proven byte-identical across runs.
- **Publishing:** `export_hf.py` requires explicit `repo_id`, token only via `HF_TOKEN` env (no `--token` CLI), per-file upload error handling with summary exit code (FIND-024 remainder).
- **Paths:** defaults resolved against the repo root in inference/onnx_chat/server/check_lengths/train (FIND-040); missing files are named in errors.
- **Seeds:** `seed=None` trailing kwarg on `generate()`/`chat()` in both engines; seeded generation proven deterministic (FIND-034).
- **Packaging:** requirements.txt range-pinned with verified-environment comment + `onnx` + `huggingface_hub` (FIND-015); MIT `LICENSE` added (FIND-020); `.github/workflows/tests.yml` CI (CPU torch; activates on owner push) (FIND-025).
- **Frontend:** live engine status via `/health` probe with honest 3-state badge (FIND-021); in-flight guard + 120 s AbortController + "still generating…" hint (FIND-022); aria-live/roles/labels/`<main>`/`<form>`/reduced-motion/IME-safe keydown (FIND-023); contrast fixes measured at 5.93:1–8.31:1 (FIND-043); truthful local-inference copy + Google Fonts removed (FIND-044); near-bottom autoscroll + 300-bubble cap (FIND-045); response-contract validation bubbles (FIND-046); 44 px touch targets + hidden-tab physics suspension + data-URI favicon (FIND-048); >1,800-char context hint (FIND-006 early feedback). XSS-safety (`textContent`) preserved; all checks live-verified in Chromium via Playwright (24/24 + 5/5, zero page errors) plus static tests.

## 9. Cross-Platform and Configuration
- Windows: launchers, server CLI, downloader, and UI verified live on Windows 11 (this machine).
- Linux/macOS: all entry points are `python -m` commands with repo-root-relative defaults (CWD-independent); the CI workflow runs the suite on `ubuntu-latest` (written, not executed here — activates on owner push). The `.bat` launchers are Windows by design and remain the documented Windows entry points.
- Remaining limitation: real macOS/Linux interactive runs were not executed in this environment (documented, not hidden).

## 10. Code Quality and Comments
Misleading comments removed or corrected (normalization claim, exporter default, uniqueness guarantee); non-obvious behavior documented concisely (window re-anchoring, temperature convention, fail-closed sidecar, allowlist rationale, pinned-manifest warning, contrast ratios as CSS comments); public contracts documented in docstrings (`generate`, `chat` both engines, loader). No machine-specific or personal details in source. Comment density kept low per repo style.

## 11. Regression Test Results
`python -m pytest tests/ -q` → **122 passed, 0 failed** (~48 s), plus live coordinator verification in §14 of the regression report. Areas: generation contracts (20+), artifact parity/gates (8), downloader fault injection (10), server contract (16), checkpoint safety (14), evaluation splits (12), remaining fixes incl. resume/UI/packaging (34), ONNX context cap (8).

## 12. Remaining Limitations
1. **FIND-005 publication:** the verified replacement ONNX + provenance sidecar exist locally with exact publication instructions; uploading to HF requires owner authorization and was NOT performed. Until then, the HF "HF version" server path serves the old artifact behind an explicit warning.
2. Payload-level (non-structural) checkpoint tamper is detectable only via the sha256-pinning layer, not the loader — by design, disclosed.
3. Dataset-content findings (FIND-003/014/047) deferred to preserve seed-42 byte reproducibility; they are now documented honestly wherever they were previously misstated.
4. `SECURITY.md` not created (outside the assignment's explicit file list); CI is written but unrun until the owner pushes; real Linux/macOS interactive sessions not executed here.
5. A full retrain with a template-disjoint validation split would be needed to *train against* the generalization gap — explicitly out of scope (no retraining authorized); `tools/evaluate.py` measures it instead.

## 13. Git Summary
- Branch `fix/coolm-comprehensive-remediation-20261105` off the audit branch; **nothing pushed**; **no remote changes**; the 4 staged audit reports + `.gitignore` remain staged and untouched.
- Commit attempts are **blocked by the Mimosa commit gate** (same SAST false-positive set as during the audit — e.g. SSRF/path-traversal heuristics on the hardcoded HF URL and local CLI paths, plus new heuristic hits on this remediation's own `model_export.py` sidecar write and `tmp_path` test usage). Per assignment rules the gate was NOT bypassed; all verified fixes are preserved in the working tree (staged/untracked) for review. Suggested commit split when the gate permits: (1) `fix(FIND-005)…` artifacts/export/verify; (2) `fix(FIND-001/002/006/009-011/029/030/035/038/039)…` inference; (3) `fix(FIND-007/008/016-019/041/042/060)…` launchers+server; (4) `fix(FIND-012)…` checkpoint loader; (5) `fix(FIND-004/024/028/044/050/057)…` evaluation+docs; (6) `fix(FIND-013/015/020/021-023/025/031-037/040/043/045/046/048/054/061)…` remaining.
- Modified files: `coolm/model.py`, `coolm/inference.py`, `coolm/config.py` (untouched — confirmed), `coolm/train.py`, `coolm/prepare_data.py`, `tools/server.py`, `tools/onnx_chat.py`, `tools/model_export.py`, `tools/export_hf.py`, `tools/check_lengths.py`, `chat.bat`, `train_coolm.bat`, `README.md`, `requirements.txt`, `docs/index.html`, `generate_data.py` (telemetry only). New: `coolm/checkpoint.py`, `tools/verify_artifacts.py`, `tools/evaluate.py`, `tests/` (9 files + conftest), `LICENSE`, `.github/workflows/tests.yml`. `coolm/config.py` was never modified (correction: it is NOT in the changed set).

## 14. Final Verdict
**PASS WITH OBSERVATIONS.** All eight High findings are remediated with independently re-verified evidence; 50 of 61 findings are fixed and covered by 122 passing regression tests; the model core is byte-untouched. The observations keep this short of a clean PASS: (1) the corrected ONNX must still be published to Hugging Face by the owner (local artifact + instructions ready; remote action not authorized), (2) three dataset-content findings are deferred with justification (dataset byte-reproducibility preserved), and (3) CI/SECURITY.md and non-Windows interactive verification remain owner-side follow-ups.
