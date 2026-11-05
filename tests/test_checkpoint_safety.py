"""FIND-012 regression tests: checkpoint security and loading reliability.

Every load in the suite goes through coolm.checkpoint (weights_only=True +
an explicit ModelConfig allowlist, never an unsafe-pickle fallback).
"""
import copy
import glob
import os
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from coolm.checkpoint import (
    CheckpointCompatibilityError,
    CheckpointError,
    CheckpointFormatError,
    CheckpointNotFoundError,
    load_checkpoint,
    load_trained_model,
)
from coolm.config import ModelConfig
from coolm.model import CooLM

# Built dynamically so the static repo scan below never trips on this file.
_UNSAFE_FLAG = "weights_only" + "=False"
_LOAD_CALL = "torch." + "load("

REAL_CHECKPOINT_PATH = os.environ.get(
    "COOLM_REAL_CHECKPOINT",
    r"C:\Users\moham\Music\coolm\audit-work\hf\best_model.pt",
)


@pytest.fixture(scope="module")
def tiny_state(infer_tiny_checkpoint):
    """State dict of the conftest tiny checkpoint (shared, read-only)."""
    return load_checkpoint(infer_tiny_checkpoint)["model_state"]


def test_roundtrip_weights_and_greedy_output(infer_tiny_checkpoint):
    ckpt0 = load_checkpoint(infer_tiny_checkpoint)

    ref = CooLM(ckpt0["model_config"])
    ref.load_state_dict(ckpt0["model_state"], strict=True)
    ref.eval()

    model, cfg, ckpt = load_trained_model(infer_tiny_checkpoint)
    model.eval()

    assert cfg is ckpt["model_config"]
    for name, tensor in ckpt0["model_state"].items():
        assert torch.equal(model.state_dict()[name], tensor), name

    input_ids = torch.tensor([[1, 10, 20, 30]], dtype=torch.long)
    ref_out = ref.generate(input_ids, 8, temperature=0, top_k=30)
    loaded_out = model.generate(input_ids, 8, temperature=0, top_k=30)
    assert torch.equal(ref_out, loaded_out)


@pytest.mark.skipif(
    not os.path.isfile(REAL_CHECKPOINT_PATH),
    reason="real published checkpoint not present on this machine",
)
def test_real_published_checkpoint_loads():
    ckpt = load_checkpoint(REAL_CHECKPOINT_PATH)
    assert ckpt["step"] == 9400
    cfg = ckpt["model_config"]
    assert isinstance(cfg, ModelConfig)
    assert cfg.vocab_size == 3269


def test_missing_file_raises_not_found(tmp_path):
    path = str(tmp_path / "absent.pt")
    with pytest.raises(CheckpointNotFoundError, match="absent"):
        load_checkpoint(path)


def test_unsupported_format_raises_format_error(tmp_path):
    path = tmp_path / "garbage.pt"
    path.write_bytes(b"not-a-checkpoint")
    with pytest.raises(CheckpointFormatError) as excinfo:
        load_checkpoint(str(path))
    # The underlying reason is preserved, not swallowed.
    assert "could not load checkpoint" in str(excinfo.value)


def test_corrupted_zip_raises_checkpoint_error(infer_tiny_checkpoint, tmp_path):
    with open(infer_tiny_checkpoint, "rb") as f:
        data = bytearray(f.read())
    # Corrupt the zip central directory (end of file): a guaranteed structural
    # break. Flipping bytes in the tensor-payload region mid-file would leave
    # a loadable archive with garbage-but-finite weights — that class of
    # corruption is caught by checksum pinning (FIND-008) and the parity gate
    # (FIND-005), not by deserialization.
    for i in range(len(data) - 64, len(data)):
        data[i] ^= 0xFF
    path = tmp_path / "corrupted.pt"
    path.write_bytes(bytes(data))
    with pytest.raises(CheckpointError) as excinfo:
        load_checkpoint(str(path))
    assert isinstance(excinfo.value, (CheckpointFormatError, CheckpointCompatibilityError))


def test_missing_model_state_key_raises_compatibility_error(tmp_path):
    path = tmp_path / "no_state.pt"
    torch.save({"step": 5, "val_loss": 1.0}, path)
    with pytest.raises(CheckpointCompatibilityError, match="model_state"):
        load_checkpoint(str(path))


def test_missing_model_config_assumes_defaults_with_warning(tmp_path, tiny_state):
    path = tmp_path / "no_cfg.pt"
    torch.save({"model_state": tiny_state}, path)
    with pytest.warns(UserWarning, match="model_config"):
        ckpt = load_checkpoint(str(path))
    assert isinstance(ckpt["model_config"], ModelConfig)
    assert ckpt["model_config"] == ModelConfig()


def test_missing_model_config_still_validates_strictly(tmp_path, tiny_state):
    # Defaults (vocab 3269, d_model 640, ...) cannot fit the tiny state dict;
    # the fallback is self-correcting: strict loading must reject it.
    path = tmp_path / "no_cfg_mismatch.pt"
    torch.save({"model_state": dict(tiny_state)}, path)
    with pytest.warns(UserWarning):
        with pytest.raises(CheckpointCompatibilityError, match="(?i)size mismatch"):
            load_trained_model(str(path))


def test_unexpected_keys_raise_compatibility_error(tmp_path, tiny_state):
    state = dict(tiny_state)
    state["extra.unexpected.weight"] = torch.zeros(4)
    path = tmp_path / "unexpected.pt"
    torch.save({"model_state": state, "model_config": ModelConfig(
        vocab_size=97, d_model=32, n_layers=2, n_heads=4, ffn_hidden=64,
        dropout=0.0, max_seq_len=32,
    )}, path)
    with pytest.raises(CheckpointCompatibilityError, match="(?i)unexpected"):
        load_trained_model(str(path))


def test_missing_keys_raise_compatibility_error(tmp_path, tiny_state):
    state = {k: v for k, v in tiny_state.items() if k != "blocks.0.ln1.weight"}
    path = tmp_path / "missing_keys.pt"
    torch.save({"model_state": state, "model_config": ModelConfig(
        vocab_size=97, d_model=32, n_layers=2, n_heads=4, ffn_hidden=64,
        dropout=0.0, max_seq_len=32,
    )}, path)
    with pytest.raises(CheckpointCompatibilityError, match="(?i)missing"):
        load_trained_model(str(path))


def test_invalid_tensor_shape_raises_compatibility_error(tmp_path, tiny_state):
    state = dict(tiny_state)
    state["blocks.0.attn.qkv.weight"] = tiny_state["blocks.0.attn.qkv.weight"].reshape(48, 64)
    path = tmp_path / "bad_shape.pt"
    torch.save({"model_state": state, "model_config": ModelConfig(
        vocab_size=97, d_model=32, n_layers=2, n_heads=4, ffn_hidden=64,
        dropout=0.0, max_seq_len=32,
    )}, path)
    with pytest.raises(CheckpointCompatibilityError, match="(?i)size mismatch"):
        load_trained_model(str(path))


def test_nan_weights_raise_compatibility_error(tmp_path, tiny_state):
    state = dict(tiny_state)
    state["blocks.0.ln1.weight"] = torch.full_like(
        tiny_state["blocks.0.ln1.weight"], float("nan")
    )
    state["blocks.1.ln2.bias"] = torch.full_like(
        tiny_state["blocks.1.ln2.bias"], float("inf")
    )
    path = tmp_path / "nan.pt"
    torch.save({"model_state": state, "model_config": ModelConfig(
        vocab_size=97, d_model=32, n_layers=2, n_heads=4, ffn_hidden=64,
        dropout=0.0, max_seq_len=32,
    )}, path)
    with pytest.raises(CheckpointCompatibilityError, match="(?i)nan|inf"):
        load_trained_model(str(path))


def _evil_write_marker(path):
    """Evil reduce callable: would run at unpickle time under an unsafe loader."""
    Path(path).write_text("pwned", encoding="utf-8")


class _EvilPayload:
    def __init__(self, marker_path):
        self.marker_path = marker_path

    def __reduce__(self):
        return _evil_write_marker, (self.marker_path,)


def test_untrusted_pickle_rejected_without_execution(tmp_path, tiny_state):
    marker = tmp_path / "evil_marker.txt"

    path = tmp_path / "evil.pt"
    torch.save({"model_state": dict(tiny_state), "payload": _EvilPayload(str(marker))}, path)

    with pytest.raises(CheckpointError):
        load_checkpoint(str(path))
    # The reduce callable must never have run.
    assert not marker.exists()


def test_repo_has_no_unsafe_checkpoint_loading():
    targets = []
    for rel in ("coolm", "tools", "tests"):
        targets.extend(glob.glob(os.path.join(REPO_ROOT, rel, "*.py")))
    targets.extend(glob.glob(os.path.join(REPO_ROOT, "*.py")))
    assert targets, "repo layout changed: no source files scanned"

    for path in sorted(targets):
        with open(path, encoding="utf-8", errors="ignore") as f:
            source = f.read()
        assert _UNSAFE_FLAG not in source, f"unsafe pickle fallback in {path}"
        for lineno, line in enumerate(source.splitlines(), start=1):
            if _LOAD_CALL in line:
                assert "weights_only=True" in line, (
                    f"{path}:{lineno}: torch.load without weights_only=True: "
                    f"{line.strip()}"
                )
