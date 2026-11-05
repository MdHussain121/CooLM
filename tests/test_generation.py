"""Remediation tests for inference correctness (PyTorch path).

Covers audit findings FIND-001, FIND-009, FIND-029, FIND-030, FIND-035,
FIND-038, FIND-051 (coolm/model.py) and FIND-002/010/011/039/061
(coolm/inference.py). All tests run on CPU against a tiny in-fixture model —
never the real 498MB checkpoint.
"""
import inspect
import os
import sys

import pytest
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from coolm.inference import normalize_user_text, strip_special_token_literals
from coolm.model import CooLM

VOCAB = 128  # GEN_TINY_CFG.vocab_size (see tests/conftest.py)


def _prompt(batch, length, seed):
    g = torch.Generator().manual_seed(seed)
    # ids >= 6: avoid the special tokens so nothing accidentally means EOS
    return torch.randint(6, VOCAB, (batch, length), generator=g)


# ---------------------------------------------------------------------------
# FIND-001: generate() must return prompt + generated, never window-truncated
# ---------------------------------------------------------------------------

def test_find001_full_sequence_returned_across_window(tiny_model_factory):
    model = tiny_model_factory(seed=0, eos_token_id=-1)  # -1 is never sampled

    # prompt 40 (> max_seq_len 32) + 100 new tokens, crossing the window
    prompt = _prompt(1, 40, seed=101)
    out = model.generate(prompt, max_new_tokens=100)
    assert out.shape == (1, 140)
    assert torch.equal(out[:, :40], prompt)

    # prompt 30 + 200 new tokens -> multiple window rebuilds
    prompt2 = _prompt(1, 30, seed=102)
    out2 = model.generate(prompt2, max_new_tokens=200)
    assert out2.shape == (1, 230)
    assert torch.equal(out2[:, :30], prompt2)

    # exact audit repro shape: prompt 26 + max_new 45 returned 65 before the
    # fix (last 65 tokens of the true sequence); must now return all 71.
    prompt3 = _prompt(1, 26, seed=103)
    out3 = model.generate(prompt3, max_new_tokens=45)
    assert out3.shape == (1, 71)
    assert torch.equal(out3[:, :26], prompt3)


def test_find001_structural_natural_eos(tiny_model_factory):
    model = tiny_model_factory(seed=1)  # natural eos_token_id=2
    prompt = _prompt(1, 10, seed=201)
    out = model.generate(prompt, max_new_tokens=64)
    n_generated = out.size(1) - 10
    assert 1 <= n_generated <= 64
    assert out.shape[1] == 10 + n_generated
    assert torch.equal(out[:, :10], prompt)


def test_find001_structural_eos_on_first_token(tiny_model_factory):
    model = tiny_model_factory(seed=2)
    prompt = _prompt(1, 8, seed=202)
    logits, _, _ = model.forward(prompt)
    first_greedy = int(logits[:, -1, :].argmax(dim=-1))
    model.config.eos_token_id = first_greedy  # force immediate EOS
    out = model.generate(prompt, max_new_tokens=50, temperature=0)
    assert out.shape == (1, 9)  # prompt + exactly the EOS token
    assert torch.equal(out[:, :8], prompt)
    assert int(out[0, -1]) == first_greedy


# ---------------------------------------------------------------------------
# FIND-009: temperature validation and greedy decoding
# ---------------------------------------------------------------------------

def test_find009_temperature_zero_is_greedy_and_deterministic(tiny_model_factory):
    model = tiny_model_factory(seed=3, eos_token_id=-1)
    prompt = _prompt(1, 6, seed=301)
    out1 = model.generate(prompt, max_new_tokens=10, temperature=0)
    out2 = model.generate(prompt, max_new_tokens=10, temperature=0)
    assert out1.shape == (1, 16)
    assert torch.equal(out1, out2)  # greedy: deterministic across calls
    # temperature=0.5 (valid sampling) works and never produces NaN/Inf ids
    out3 = model.generate(prompt, max_new_tokens=10, temperature=0.5)
    assert out3.shape == (1, 16)
    assert torch.isfinite(out3).all()


def test_find009_tiny_temperature_stays_finite(tiny_model_factory):
    # Coordinator regression: a positive temperature below ~1e-6 used to
    # saturate fp32 logits to inf -> NaN softmax -> multinomial crash.
    # The floored divisor must keep it finite (and effectively greedy).
    model = tiny_model_factory(seed=3, eos_token_id=-1)
    prompt = _prompt(1, 6, seed=301)
    out = model.generate(prompt, max_new_tokens=10, temperature=1e-300)
    assert out.shape == (1, 16)
    out2 = model.generate(prompt, max_new_tokens=10, temperature=1e-300)
    assert torch.equal(out, out2)  # floored tiny temperature acts greedy


def test_find009_invalid_temperature_raises(tiny_model_factory):
    model = tiny_model_factory(seed=3)
    prompt = _prompt(1, 4, seed=302)
    with pytest.raises(ValueError):
        model.generate(prompt, max_new_tokens=2, temperature=-1)
    with pytest.raises(ValueError):
        model.generate(prompt, max_new_tokens=2, temperature=float("nan"))
    with pytest.raises(ValueError):
        model.generate(prompt, max_new_tokens=2, temperature=float("inf"))


def test_generate_argument_type_validation(tiny_model_factory):
    model = tiny_model_factory(seed=3)
    prompt = _prompt(1, 4, seed=303)
    with pytest.raises(TypeError):
        model.generate(prompt, max_new_tokens=2.5)
    with pytest.raises(TypeError):
        model.generate(prompt, max_new_tokens=True)  # bool is not an int here
    with pytest.raises(TypeError):
        model.generate(prompt, max_new_tokens=2, temperature="hot")
    with pytest.raises(TypeError):
        model.generate(prompt, max_new_tokens=2, top_k=1.5)
    with pytest.raises(ValueError):
        model.generate(prompt, max_new_tokens=2, top_k=-3)


def test_generate_signature_preserved():
    # coolm/inference.py calls generate() positionally — order must not move.
    # Assignment-6 (FIND-034) appends the optional trailing seed=None kwarg;
    # the original positional order and defaults are still pinned, and seed
    # must stay LAST with a None default (unseeded behavior unchanged).
    sig = inspect.signature(CooLM.generate)
    names = list(sig.parameters)
    assert names[:5] == ["self", "input_ids", "max_new_tokens", "temperature", "top_k"]
    assert names[5:] == ["seed"]
    defaults = [sig.parameters[n].default for n in names[1:5]]
    assert defaults == [inspect.Parameter.empty, 64, 1.0, 40]
    assert sig.parameters["seed"].default is None


def test_top_k_above_vocab_clamped_silently(tiny_model_factory):
    model = tiny_model_factory(seed=4, eos_token_id=-1)
    prompt = _prompt(1, 4, seed=304)
    out = model.generate(prompt, max_new_tokens=3, temperature=1.0, top_k=10_000)
    assert out.shape == (1, 7)  # no crash, no ValueError


# ---------------------------------------------------------------------------
# FIND-029: max_new_tokens <= 0 is a documented no-op
# ---------------------------------------------------------------------------

def test_find029_nonpositive_max_new_tokens_returns_input(tiny_model_factory):
    model = tiny_model_factory(seed=5)
    prompt = _prompt(1, 12, seed=401)
    out0 = model.generate(prompt, max_new_tokens=0)
    assert torch.equal(out0, prompt)
    out_neg = model.generate(prompt, max_new_tokens=-5)
    assert torch.equal(out_neg, prompt)


# ---------------------------------------------------------------------------
# FIND-030: batched generation with per-sequence EOS state
# ---------------------------------------------------------------------------

def test_find030_batch_shape_and_prefix_preserved(tiny_model_factory):
    model = tiny_model_factory(seed=6, eos_token_id=-1)  # never stops early
    prompts = _prompt(2, 9, seed=501)
    out = model.generate(prompts, max_new_tokens=12)
    assert out.shape == (2, 21)  # uniform (B, prompt + generated)
    assert torch.equal(out[:, :9], prompts)

    prompts4 = _prompt(4, 5, seed=502)
    out4 = model.generate(prompts4, max_new_tokens=6)
    assert out4.shape == (4, 11)
    assert torch.equal(out4[:, :5], prompts4)


def test_find030_per_sequence_eos(tiny_model_factory):
    model = tiny_model_factory(seed=7)
    prompts = _prompt(2, 6, seed=503)
    logits, _, _ = model.forward(prompts)
    first_greedy = logits[:, -1, :].argmax(dim=-1)  # (2,)
    row0_eos = int(first_greedy[0])
    assert row0_eos != model.config.pad_token_id  # the test needs them distinct

    # Row 0's very first greedy token becomes EOS: it must finish after one
    # generated token while row 1 keeps generating content.
    model.config.eos_token_id = row0_eos
    out = model.generate(prompts, max_new_tokens=12, temperature=0)

    assert out.shape == (2, 18)  # row 1 runs the full 12 steps
    assert torch.equal(out[:, :6], prompts)

    row0_suffix = out[0, 6:]
    assert int(row0_suffix[0]) == row0_eos  # its final content token is EOS
    assert torch.all(row0_suffix[1:] == model.config.pad_token_id)  # pad after

    row1_suffix = out[1, 6:]
    assert (row1_suffix != model.config.pad_token_id).any()  # real content


# ---------------------------------------------------------------------------
# FIND-038: training state restored even when generate() raises
# ---------------------------------------------------------------------------

def test_find038_training_state_restored_on_failure(tiny_model_factory, monkeypatch):
    model = tiny_model_factory(seed=8)
    model.train()
    prompt = _prompt(1, 4, seed=601)

    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(model, "forward", boom)
    with pytest.raises(RuntimeError, match="boom"):
        model.generate(prompt, max_new_tokens=3)
    assert model.training is True  # finally-branch restored the caller state


def test_find038_training_state_restored_on_success(tiny_model_factory):
    model = tiny_model_factory(seed=8)
    model.train()
    model.generate(_prompt(1, 4, seed=602), max_new_tokens=2)
    assert model.training is True


# ---------------------------------------------------------------------------
# FIND-035: forward() fails early on kv-cache overflow / mis-anchored start
# ---------------------------------------------------------------------------

def test_find035_kv_cache_overflow_assert(tiny_model_factory):
    model = tiny_model_factory(seed=9)
    ids = _prompt(1, 25, seed=701)
    _, _, caches = model.forward(ids, use_cache=True)  # cached_len = 25

    # cached 25 + T 8 + start 0 = 33 > max_seq_len 32: the old assert
    # (T + start_pos <= max) passed this; the new cache assert fires.
    with pytest.raises(AssertionError, match="kv cache overflow"):
        model.forward(ids[:, :8], use_cache=True, kv_caches=caches, start_pos=0)

    # The audit's exact FIND-035 repro shape: full 32-cache + 1 token + start 0
    full_ids = _prompt(1, 32, seed=702)
    _, _, full_caches = model.forward(full_ids, use_cache=True)
    with pytest.raises(AssertionError):
        model.forward(full_ids[:, :1], use_cache=True, kv_caches=full_caches, start_pos=0)

    # Mis-anchored start (cached 20 + T 1 + start 15 = 36 > 32): positions
    # would rewind into the cache — caught by the anchor consistency check.
    ids2 = _prompt(1, 20, seed=703)
    _, _, caches2 = model.forward(ids2, use_cache=True)
    with pytest.raises(AssertionError, match="start_pos"):
        model.forward(ids2[:, :1], use_cache=True, kv_caches=caches2, start_pos=15)

    # The verified incremental path (start_pos == cached_len) must keep working.
    model.forward(ids2[:, :1], use_cache=True, kv_caches=caches2, start_pos=20)


# ---------------------------------------------------------------------------
# FIND-010: shared user-text normalization
# ---------------------------------------------------------------------------

def test_find010_normalize_user_text():
    assert normalize_user_text("HIGH-RISE City!  123") == "high-rise city"
    assert "coo" in normalize_user_text("coo what is rain")
    assert normalize_user_text("!!!123") == ""  # pure symbols -> empty (documented)
    assert normalize_user_text("  a   b\tc\nd ") == "a b c d"


# ---------------------------------------------------------------------------
# FIND-002 / FIND-039 / FIND-061: response = exactly the generated tokens
# ---------------------------------------------------------------------------

def test_find002_response_is_generated_tokens_only(tiny_torch_engine, monkeypatch):
    engine = tiny_torch_engine
    captured = {}
    original_generate = engine.model.generate

    def recording_generate(input_ids, *args, **kwargs):
        out = original_generate(input_ids, *args, **kwargs)
        captured["prompt_len"] = int(input_ids.size(1))
        captured["output"] = out
        return out

    monkeypatch.setattr(engine.model, "generate", recording_generate)
    response = engine.chat("is rain good for pigeons", temperature=0)

    assert isinstance(response, str)
    # No special-token literal fragments of any kind (FIND-002 garbage).
    for bad in ("<|", "os|>", "nd|>", ">...."):
        assert bad not in response

    # The response equals the decode of exactly the generated tokens
    # (token-index extraction, not a character offset).
    generated_ids = captured["output"][0, captured["prompt_len"]:].tolist()
    expected = engine.tokenizer.decode(generated_ids, skip_special_tokens=True)
    assert response == strip_special_token_literals(expected)


def test_chat_rejects_non_string_user_text(tiny_torch_engine):
    with pytest.raises(TypeError):
        tiny_torch_engine.chat(12345)
    with pytest.raises(TypeError):
        tiny_torch_engine.chat(None)


# ---------------------------------------------------------------------------
# FIND-011: oversized user text is tail-truncated with a warning
# ---------------------------------------------------------------------------

def test_find011_long_user_text_truncated_with_warning(tiny_torch_engine, monkeypatch):
    engine = tiny_torch_engine
    captured = {}
    original_generate = engine.model.generate

    def recording_generate(input_ids, *args, **kwargs):
        captured["prompt_len"] = int(input_ids.size(1))
        return original_generate(input_ids, *args, **kwargs)

    monkeypatch.setattr(engine.model, "generate", recording_generate)
    # 300 in-vocab words — far beyond the 32-token context of the tiny model.
    long_text = " ".join(["aa", "ab", "ac", "ad"] * 75)
    with pytest.warns(UserWarning):
        response = engine.chat(long_text)
    assert isinstance(response, str)
    assert captured["prompt_len"] <= engine.model.config.max_seq_len
    assert captured["prompt_len"] >= 2  # scaffold + bos still present


# ---------------------------------------------------------------------------
# FIND-051: the window re-anchoring semantics are documented in the source
# ---------------------------------------------------------------------------

def test_find051_window_reanchoring_documented():
    source = inspect.getsource(CooLM)
    assert "re-anchored" in source  # marker in the rebuild comment/docstring
