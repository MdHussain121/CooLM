import dataclasses
import json
import os
import shutil
import subprocess
import sys

import pytest
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from coolm.config import ModelConfig
from coolm.model import CooLM
import tools.export_hf as export_hf
from tools import model_export
import tools.verify_artifacts as va

# Tiny config: the suite never touches the 498MB real checkpoint.
TINY_CFG = ModelConfig(
    vocab_size=97,
    d_model=32,
    n_layers=2,
    n_heads=4,
    ffn_hidden=64,
    dropout=0.0,
    max_seq_len=32,
)


def _tiny_state_dict():
    torch.manual_seed(0)
    return CooLM(TINY_CFG).state_dict()


@pytest.fixture(scope="session")
def tiny_checkpoint(tmp_path_factory):
    path = tmp_path_factory.mktemp("artifacts") / "tiny.pt"
    torch.save(
        {"step": 5, "model_state": _tiny_state_dict(), "model_config": TINY_CFG, "val_loss": 1.23},
        path,
    )
    return str(path)


@pytest.fixture(scope="session")
def tiny_export(tiny_checkpoint, tmp_path_factory):
    out_path = tmp_path_factory.mktemp("export") / "tiny.onnx"
    rc = model_export.export_onnx(checkpoint_path=tiny_checkpoint, output_path=str(out_path))
    assert rc == 0
    return {
        "checkpoint": tiny_checkpoint,
        "onnx": str(out_path),
        "sidecar": str(out_path) + ".provenance.json",
    }


@pytest.fixture(scope="session")
def perturbed_checkpoint(tiny_checkpoint, tmp_path_factory):
    torch.serialization.add_safe_globals([ModelConfig])
    ckpt = torch.load(tiny_checkpoint, map_location="cpu", weights_only=True)
    ckpt["model_state"]["blocks.0.ln1.weight"] = ckpt["model_state"]["blocks.0.ln1.weight"] + 1.0
    path = tmp_path_factory.mktemp("perturbed") / "tiny_perturbed.pt"
    torch.save(ckpt, path)
    return str(path)


def _write_tokenizer(path, vocab_size=97, pad_id=0, bos_id=1, eos_id=2):
    from tokenizers import Tokenizer, models

    id_to_token = {pad_id: "<|pad|>", bos_id: "<|bos|>", eos_id: "<|eos|>"}
    vocab = {}
    next_id = 0
    while len(vocab) < vocab_size:
        if next_id in id_to_token:
            vocab[id_to_token[next_id]] = next_id
        else:
            vocab[f"tok{next_id}"] = next_id
        next_id += 1
    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<|pad|>"))
    tokenizer.save(str(path))
    return str(path)


def test_export_parity_passes(tiny_export):
    results = va.verify_checkpoint_onnx_parity(tiny_export["checkpoint"], tiny_export["onnx"])
    assert results["passed"], results["failures"]
    # seq lengths capped at max_seq_len=32: {1, 8, 32} x batches {1, 4}
    assert len(results["shapes"]) == 6
    assert results["step"] == 5
    assert results["config"] == dataclasses.asdict(TINY_CFG)


def test_parity_detects_mismatch(tiny_export, perturbed_checkpoint):
    results = va.verify_checkpoint_onnx_parity(perturbed_checkpoint, tiny_export["onnx"])
    assert not results["passed"]
    assert results["failures"]


def test_cli_exit_codes(tiny_checkpoint, tiny_export, perturbed_checkpoint):
    assert va.main(["--checkpoint", tiny_checkpoint, "--onnx", tiny_export["onnx"]]) == 0
    assert va.main(["--checkpoint", perturbed_checkpoint, "--onnx", tiny_export["onnx"]]) == 1


def test_module_cli_subprocess_pass(tiny_checkpoint, tiny_export, tmp_path):
    tokenizer = _write_tokenizer(tmp_path / "tokenizer.json")
    proc = subprocess.run(
        [
            sys.executable, "-m", "tools.verify_artifacts",
            "--checkpoint", tiny_checkpoint,
            "--onnx", tiny_export["onnx"],
            "--tokenizer", tokenizer,
        ],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "OVERALL: PASS" in proc.stdout
    assert "TOKENIZER PAIRING: PASS" in proc.stdout


def test_module_cli_subprocess_fail(perturbed_checkpoint, tiny_export):
    proc = subprocess.run(
        [
            sys.executable, "-m", "tools.verify_artifacts",
            "--checkpoint", perturbed_checkpoint,
            "--onnx", tiny_export["onnx"],
        ],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode != 0
    assert "OVERALL: FAIL" in proc.stdout


def test_provenance_sidecar_written_and_content(tiny_export):
    assert os.path.exists(tiny_export["sidecar"])
    with open(tiny_export["sidecar"], encoding="utf-8") as f:
        provenance = json.load(f)
    assert provenance["checkpoint_sha256"] == va.sha256_file(tiny_export["checkpoint"])
    assert provenance["checkpoint_step"] == 5
    assert provenance["checkpoint_val_loss"] == 1.23
    assert provenance["model_config"] == dataclasses.asdict(TINY_CFG)
    assert provenance["opset"] == 14
    assert provenance["input_names"] == ["input_ids"]
    assert provenance["output_names"] == ["logits"]
    assert "export_timestamp" in provenance
    assert "torch_version" in provenance


def test_export_hf_gate_blocks_inconsistent(tiny_export, perturbed_checkpoint, tmp_path):
    # Matching pair -> allowed.
    allowed, reasons = export_hf.check_publish_consistency(
        checkpoint_path=tiny_export["checkpoint"],
        onnx_path=tiny_export["onnx"],
        tokenizer_path=None,
    )
    assert allowed, reasons

    # Mismatched checkpoint -> refused.
    allowed, reasons = export_hf.check_publish_consistency(
        checkpoint_path=perturbed_checkpoint,
        onnx_path=tiny_export["onnx"],
        tokenizer_path=None,
    )
    assert not allowed
    assert any("mismatch" in r for r in reasons)

    # ONNX present but sidecar missing -> refused (fail-closed).
    orphan_onnx = str(tmp_path / "orphan.onnx")
    shutil.copyfile(tiny_export["onnx"], orphan_onnx)
    allowed, reasons = export_hf.check_publish_consistency(
        checkpoint_path=tiny_export["checkpoint"],
        onnx_path=orphan_onnx,
        tokenizer_path=None,
    )
    assert not allowed
    assert any("sidecar missing" in r for r in reasons)

    # ONNX present but checkpoint missing -> refused.
    allowed, reasons = export_hf.check_publish_consistency(
        checkpoint_path=str(tmp_path / "missing.pt"),
        onnx_path=tiny_export["onnx"],
        tokenizer_path=None,
    )
    assert not allowed
    assert any("checkpoint not found" in r for r in reasons)

    # No ONNX at all -> gate skips, publication not blocked by it.
    allowed, reasons = export_hf.check_publish_consistency(
        checkpoint_path=tiny_export["checkpoint"],
        onnx_path=str(tmp_path / "absent.onnx"),
        tokenizer_path=None,
    )
    assert allowed


def test_tokenizer_pairing(tmp_path):
    good = _write_tokenizer(tmp_path / "good.json")
    results = va.verify_tokenizer_pairing(good, TINY_CFG)
    assert results["passed"], results["failures"]
    assert results["vocab_size"] == TINY_CFG.vocab_size
    assert results["special_tokens"]["bos"]["id"] == TINY_CFG.bos_token_id
    assert results["special_tokens"]["eos"]["id"] == TINY_CFG.eos_token_id
    assert results["special_tokens"]["pad"]["id"] == TINY_CFG.pad_token_id

    # bos/pad ids swapped -> must fail.
    bad = _write_tokenizer(tmp_path / "bad.json", pad_id=1, bos_id=0)
    results = va.verify_tokenizer_pairing(bad, TINY_CFG)
    assert not results["passed"]
    assert results["failures"]

    # Wrong vocab size -> must fail.
    small = _write_tokenizer(tmp_path / "small.json", vocab_size=50)
    results = va.verify_tokenizer_pairing(small, TINY_CFG)
    assert not results["passed"]
    assert any("vocab size" in f for f in results["failures"])
