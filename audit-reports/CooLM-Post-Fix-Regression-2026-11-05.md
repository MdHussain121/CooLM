# CooLM — Post-Fix Regression Verification

**Date:** 2026-11-05
**Performed by:** coordinator (Assignment 7 — explicitly not delegated), after reviewing each implementation assignment independently.
**Environment:** Windows 11, Python 3.12.10, torch 2.13.0+cu126, onnxruntime 1.26.0 (CPU), tokenizers 0.22.2. Real artifacts from `audit-work/hf/` (best_model.pt sha256 `6db4cb4c…fd60e`, step 9400; published mismatched model.onnx; tokenizer.json).

**Suite:** `python -m pytest tests/ -q` → **122 passed, 0 failed** (~48 s). Re-run by the coordinator after every assignment; final run after Assignment 6.

---

## 1. Independent re-verification of the High findings (coordinator-executed, live)

| # | Check | Result | Evidence |
|---|---|---|---|
| 1 | **FIND-001** — generate() return preservation across window transitions, real checkpoint, EOS disabled | **PASS** | audit repro shape (prompt 10 + max_new 45): returned **55** (bug returned 65-cap), prefix `torch.equal` to input; 10+200 → 210, prefix intact across multiple rebuilds |
| 2 | **FIND-002** — no special-token fragments possible | **PASS** | extraction is token-index only (code + tests); real-chat outputs clean |
| 3 | **FIND-005** — replacement ONNX parity | **PASS** | `tools.verify_artifacts`: max-abs ≤ 2.19e-05, argmax 100 % on all 8 shapes; negative control (published ONNX) → FAIL as designed |
| 4 | **FIND-005** — provenance chain | **PASS** | sidecar `checkpoint_sha256` == best_model.pt sha256; step 9400; opset 14; torch 2.13.0+cu126; ONNX sha256 `ce173d48…` matches SHA256SUMS |
| 5 | **FIND-006** — long-message server test, real engine, isolated staging dir | **PASS** | 1,422-char POST → **HTTP 200** (audit: guaranteed 500); `/health` → `{"status":"ok","engine":true,"max_seq_len":64}`; empty → `"coo..."`; bad JSON → 400; UI 200 |
| 6 | **FIND-007** — launcher from fresh clone, arbitrary CWD, path with spaces | **PASS** | `chat.bat` from `…\a7 dir with spaces\` → menu, clean local-model error path, no `cd` error, no traceback |
| 7 | **FIND-008** — downloader fault injection | **PASS** | 10/10 scenarios (truncated / tampered / size / 404 / 500 / hang / corrupt-cache / retry / valid-cache-preserved / manifest pinning) |
| 8 | **FIND-012** — safe loading of the real checkpoint + tamper behavior | **PASS** | loads via `weights_only=True`+allowlist (step 9400); untrusted `__reduce__` payload rejected without execution (suite); structurally-corrupt archive rejected (suite). **Disclosed boundary:** payload-region byte flips keep the archive valid with finite garbage weights — detected by the sha256-pinning layer, not the loader (defense in depth, documented in the remediation report §7) |
| 9 | **FIND-004** — metric replication + honest splits | **PASS** | shipped split reproduces **0.3262 / 1.39** exactly; template-disjoint 0.3309 and topic-disjoint 0.3414 with 0 % verbatim overlap; leakage table reproduces the audit's 95.02 % |
| 10 | Model core untouched | **PASS** | best_model.pt sha256 `6db4cb4c…` verified identical after every assignment; dataset/tokenizer never modified; no retraining |

## 2. Previously failing integration tests → now passing
- Published-ONNX vs PT logits: was **FAIL** (maxabs 8.79, argmax 12.5 %) → replacement artifact **PASS**; the check that fails the old artifact is itself regression-tested.
- ONNX long input: was **RuntimeException → HTTP 500** → HTTP 200 at 1,422 chars; fed lengths ≤ 64 asserted by tests (RecordingSession).
- Both Windows launchers: were dead on fresh clones → verified live from fresh clone + spaces dir; pipeline control flow verified with stubs (4 steps, failure propagation).
- Downloader partial-accept: was `True` on a 300 KB truncation → all partial/corrupt transfers now rejected, destination never created.
- `t=0`, `max_new_tokens=0`, batch>1, >64-token prompts, window-crossing returns, extraction offsets: all covered by `tests/test_generation.py` (21 tests) and `tests/test_onnx_context_cap.py` (8).

## 3. Original parity experiment (audit reproduction)
Fresh export of `best_model.pt` (through the new exporter path) is byte-identical to the audit's independent `fresh_legacy.onnx`; PT↔ONNX parity on identical seeded inputs: max-abs ≤ 2.5e-05, 100 % argmax across B∈{1,4}×T∈{1,8,32,64}; near-greedy chat parity confirmed by the ONNX engine reproducing the PyTorch personality (e.g. "do you have bird friends" → "a friend is a bird who does not steal your seeds." — exact audit reference output).

## 4. Long-message handling (end-to-end matrix)
Empty / short / at-context / 1,422-char / repeated requests — all verified live (HTTP 200 with fluent or empty-by-model-choice responses, no 500s); multibyte input passes through `normalize_user_text` (documented stripping); repeated long requests serialize under the inference lock without errors.

## 5. Metric and model-card accuracy
README headline metrics now labeled in-distribution template-retrieval with pointer to `tools.evaluate`; model card metric names + Limitations quantified; the "runs in browser", "100 % unique", "balanced evenly", and "OOD protection" claims are corrected. Verified by grep review of README + export_hf YAML and by A5 tests.

## 6. Full findings register
All 61 original IDs carry an explicit disposition in `CooLM-Remediation-Status-2026-11-05.json`: 50 FIXED, 3 DEFERRED WITH JUSTIFICATION, 8 INFORMATIONAL/no-change, 0 silent.

## 7. Cross-platform status
Windows verified live (launchers, server CLI, UI in Chromium via Playwright, downloader fault matrix). Linux coverage: repo-root path resolution + `python -m` entry points + CI workflow on `ubuntu-latest` (written, not executed — owner-side push required). macOS: same code paths, not executed here.

## 8. Verdict
**No regressions introduced by the fixes that could be detected.** The final suite is green (122/122), every High finding is re-verified live against the real artifacts, the audited strengths (parameter count, tying, causality, checkpoint authenticity, dataset reproducibility) are untouched by construction (core files unmodified except the audited-defect sites), and the two intentionally-open items (HF publication of the replacement ONNX; CI activation) require only owner actions that were explicitly out of scope.
