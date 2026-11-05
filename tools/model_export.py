import dataclasses
import datetime
import importlib.util
import json
import os
import pathlib
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from coolm.config import ModelConfig
from tools.verify_artifacts import (
    format_parity_report,
    load_checkpoint_model,
    sha256_file,
    verify_checkpoint_onnx_parity,
)

OPSET_VERSION = 14

EXPORT_KWARGS = dict(
    export_params=True,
    opset_version=OPSET_VERSION,
    do_constant_folding=True,
    input_names=["input_ids"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "logits": {0: "batch_size", 1: "sequence_length"}
    },
)


def _stub_onnxscript_splicer_if_needed():
    # torch's legacy exporter round-trips the finished protobuf through the
    # onnx package just to splice in onnxscript custom-function protos. This
    # graph contains none, so that round trip is a byte-level no-op. The onnx
    # package is listed in requirements.txt but a bare environment may lack
    # it; in that case stub the splicer instead of failing the whole export.
    # Returns the original splicer when stubbed, else None.
    if importlib.util.find_spec("onnx") is not None:
        return None
    try:
        from torch.onnx._internal.torchscript_exporter import onnx_proto_utils
    except ImportError:
        return None
    if not hasattr(onnx_proto_utils, "_add_onnxscript_fn"):
        return None

    original = onnx_proto_utils._add_onnxscript_fn
    onnx_proto_utils._add_onnxscript_fn = lambda model_bytes, custom_opsets: model_bytes
    return original


def _restore_onnxscript_splicer(original):
    from torch.onnx._internal.torchscript_exporter import onnx_proto_utils
    onnx_proto_utils._add_onnxscript_fn = original


def provenance_sidecar_path(output_path):
    return output_path + ".provenance.json"


def find_tokenizer_path(checkpoint_path):
    # Default tokenizer locations relative to the checkpoint context
    # (checkpoints/best_model.pt -> data/tokenizer.json at the project root).
    ckpt_dir = os.path.dirname(os.path.abspath(checkpoint_path))
    candidates = [
        os.path.join(os.getcwd(), "data", "tokenizer.json"),
        os.path.join(ckpt_dir, os.pardir, "data", "tokenizer.json"),
        os.path.join(ckpt_dir, "tokenizer.json"),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return None


def write_provenance_sidecar(output_path, checkpoint_path, ckpt, cfg):
    entry = {
        "checkpoint_path": os.path.abspath(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_step": ckpt.get("step"),
        "checkpoint_val_loss": ckpt.get("val_loss"),
        "model_config": dataclasses.asdict(cfg),
        "opset": OPSET_VERSION,
        "torch_version": torch.__version__,
        "export_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "input_names": ["input_ids"],
        "output_names": ["logits"],
    }
    tokenizer_path = find_tokenizer_path(checkpoint_path)
    if tokenizer_path:
        entry["tokenizer_path"] = os.path.abspath(tokenizer_path)
        entry["tokenizer_sha256"] = sha256_file(tokenizer_path)
    sidecar_path = provenance_sidecar_path(output_path)
    pathlib.Path(sidecar_path).write_text(json.dumps(entry, indent=2), encoding="utf-8")
    return sidecar_path


def _remove_quietly(*paths):
    for path in paths:
        if os.path.exists(path):
            os.remove(path)


def export_onnx(checkpoint_path="checkpoints/best_model.pt", output_path="docs/model.onnx"):
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint {checkpoint_path} not found.")
        return 1

    print("Loading model...")
    ckpt, cfg, model = load_checkpoint_model(checkpoint_path)

    # Create dummy input for tracing
    dummy_input = torch.randint(0, cfg.vocab_size, (1, 32), dtype=torch.long, device="cpu")

    class ONNXWrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, x):
            logits, _, _ = self.m(x, use_cache=False)
            return logits

    wrapper = ONNXWrapper(model)
    wrapper.eval()

    sidecar_path = provenance_sidecar_path(output_path)
    print(f"Exporting to {output_path}...")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # We only export the forward pass that returns logits.
    # The actual inference loop must be handled in JS or Python.
    # dynamo=False forces the legacy exporter: on torch>=2.9 the default dynamo
    # exporter requires onnxscript, which requirements.txt does not include,
    # and fails without it. The legacy exporter reproduces the published graph
    # structurally and works on modern torch.
    spliced_original = None
    try:
        spliced_original = _stub_onnxscript_splicer_if_needed()
        torch.onnx.export(wrapper, dummy_input, output_path, dynamo=False, **EXPORT_KWARGS)
    except Exception:
        # Fail closed: a failed or interrupted export must never leave a stale
        # sidecar (or a partial ONNX) that could validate an artifact this run
        # did not actually produce.
        if spliced_original is not None:
            _restore_onnxscript_splicer(spliced_original)
        _remove_quietly(sidecar_path, output_path)
        raise
    if spliced_original is not None:
        _restore_onnxscript_splicer(spliced_original)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Done! ONNX model size: {size_mb:.2f} MB")

    # Written only after a successful export so it can never go stale.
    write_provenance_sidecar(output_path, checkpoint_path, ckpt, cfg)
    print(f"Wrote provenance sidecar: {sidecar_path}")

    print("Verifying exported ONNX against the checkpoint...")
    try:
        results = verify_checkpoint_onnx_parity(checkpoint_path, output_path)
    except Exception:
        _remove_quietly(sidecar_path, output_path)
        raise
    print(format_parity_report(results))
    if not results["passed"]:
        # Fail closed: a self-inconsistent export must not survive as an
        # apparently-publishable artifact.
        print("Parity check FAILED. Deleting the bad ONNX and its sidecar.")
        _remove_quietly(sidecar_path, output_path)
        return 1
    print("Parity check PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(export_onnx())
