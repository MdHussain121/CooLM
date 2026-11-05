"""ONNX-path context-cap tests (audit FIND-006).

Exercises tools/onnx_chat.CooONNXInference against a tiny exported model:
every session feed must stay within the model's context, for any input
length, including mid-generation. All tests run on CPU.
"""
import json
import os
import shutil
import sys
from pathlib import Path

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tools.onnx_chat import CooONNXInference

TINY_MAX_SEQ_LEN = 32  # GEN_TINY_CFG.max_seq_len, exported in the sidecar


class RecordingSession:
    """Proxy that records every fed sequence length while delegating to the
    real ONNX Runtime session."""

    def __init__(self, real):
        self._real = real
        self.fed_lengths = []
        self.run_calls = 0

    def get_inputs(self):
        return self._real.get_inputs()

    def run(self, output_names, input_feed):
        (array,) = input_feed.values()
        self.fed_lengths.append(int(array.shape[1]))
        self.run_calls += 1
        return self._real.run(output_names, input_feed)


def _record(engine):
    engine.session = RecordingSession(engine.session)
    return engine.session


def test_onnx_engine_loads_tiny_export(onnx_tiny_export, infer_tiny_tokenizer):
    engine = CooONNXInference(
        onnx_path=onnx_tiny_export["onnx"], tokenizer_path=infer_tiny_tokenizer
    )
    # max_seq_len comes from the export sidecar (32), not the 64 fallback.
    assert engine.max_seq_len == TINY_MAX_SEQ_LEN


def test_find006_long_input_never_exceeds_context(onnx_engine):
    engine = onnx_engine
    recorder = _record(engine)
    # 300 words — user text far beyond the window; must not raise.
    long_text = " ".join(["aa", "ab", "ac", "ad"] * 75)
    response = engine.chat(long_text)
    assert isinstance(response, str)
    assert recorder.fed_lengths
    assert max(recorder.fed_lengths) <= engine.max_seq_len


def test_max_new_tokens_caps_session_runs(onnx_engine):
    engine = onnx_engine
    recorder = _record(engine)
    response = engine.chat("aa ab ac ad ae", max_new_tokens=5, temperature=0)
    assert isinstance(response, str)
    # One session run per generated token, so at most 5 runs.
    assert 1 <= recorder.run_calls <= 5


def test_onnx_greedy_deterministic_and_invalid_temperature(onnx_engine):
    engine = onnx_engine
    first = engine.chat("aa ab ac", temperature=0)
    second = engine.chat("aa ab ac", temperature=0)
    assert first == second  # greedy is deterministic
    with pytest.raises(ValueError):
        engine.chat("aa", temperature=-1)
    with pytest.raises(ValueError):
        engine.chat("aa", temperature=float("nan"))
    with pytest.raises(TypeError):
        engine.chat("aa", max_new_tokens=2.5)
    with pytest.raises(TypeError):
        engine.chat(12345)


def test_onnx_zero_max_new_tokens_empty_response(onnx_engine):
    engine = onnx_engine
    recorder = _record(engine)
    assert engine.chat("aa ab", max_new_tokens=0) == ""  # documented contract
    assert recorder.run_calls == 0


def test_sidecar_max_seq_len_is_actually_read(onnx_tiny_export, infer_tiny_tokenizer, tmp_path):
    custom = tmp_path / "capped.onnx"
    shutil.copyfile(onnx_tiny_export["onnx"], custom)
    Path(str(custom) + ".provenance.json").write_text(
        json.dumps({"model_config": {"max_seq_len": 24}}), encoding="utf-8"
    )

    engine = CooONNXInference(onnx_path=str(custom), tokenizer_path=infer_tiny_tokenizer)
    assert engine.max_seq_len == 24  # sidecar value, not 64 and not the model's 32

    recorder = _record(engine)
    engine.chat(" ".join(["aa", "ab"] * 60), temperature=0)  # long user text
    assert recorder.fed_lengths
    assert max(recorder.fed_lengths) <= 24


def test_missing_sidecar_falls_back_to_64(onnx_tiny_export, infer_tiny_tokenizer, tmp_path, capsys):
    bare = tmp_path / "bare.onnx"
    shutil.copyfile(onnx_tiny_export["onnx"], bare)  # no sidecar next to it

    engine = CooONNXInference(onnx_path=str(bare), tokenizer_path=infer_tiny_tokenizer)
    assert engine.max_seq_len == 64
    assert "falling back" in capsys.readouterr().out

    # Keep max_new_tokens small: the fallback cap (64) exceeds the tiny
    # model's real context (32), so the prompt plus generation must stay
    # under 32 here.
    recorder = _record(engine)
    engine.chat("aa ab ac", max_new_tokens=5, temperature=0)
    assert recorder.fed_lengths
    assert max(recorder.fed_lengths) <= 32


def test_onnx_user_text_truncation_warns(onnx_engine):
    engine = onnx_engine
    recorder = _record(engine)
    with pytest.warns(UserWarning):
        engine.chat(" ".join(["aa", "ab"] * 150), temperature=0)  # 300 words
    assert recorder.fed_lengths
    assert max(recorder.fed_lengths) <= engine.max_seq_len
