import argparse
import dataclasses
import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from coolm.checkpoint import load_trained_model
from coolm.config import ModelConfig

# fp32 engine noise between PyTorch and ORT on identical weights measured
# <=2.5e-5 during the audit, while mismatched weights diverge by orders of
# magnitude (8.79). 1e-3 separates the two cleanly.
MAX_ABS_DIFF_THRESHOLD = 1e-3
PARITY_SEED = 1234
BATCH_SIZES = (1, 4)
SEQ_LENGTHS = (1, 8, 32, 64)


def sha256_file(path, chunk_size=1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def load_checkpoint_model(checkpoint_path):
    """Backward-compatible wrapper: delegate to the authoritative safe loader
    in coolm.checkpoint. Returns (ckpt, cfg, model) with the model in eval()
    state, as callers historically expect."""
    model, cfg, ckpt = load_trained_model(checkpoint_path)
    model.eval()
    return ckpt, cfg, model


def parity_shapes(cfg, max_tokens=None):
    cap = cfg.max_seq_len
    if max_tokens is not None:
        cap = min(cap, max_tokens)
    return [(b, t) for b in BATCH_SIZES for t in SEQ_LENGTHS if t <= cap]


def run_parity(model, onnx_path, cfg, shapes, seed=PARITY_SEED):
    import onnxruntime as ort

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    rng = np.random.default_rng(seed)
    results = []
    for batch, seq_len in shapes:
        x = rng.integers(0, cfg.vocab_size, size=(batch, seq_len)).astype(np.int64)
        with torch.no_grad():
            pt_logits, _, _ = model(torch.from_numpy(x), use_cache=False)
        pt_logits = pt_logits.numpy()
        ort_logits = session.run(None, {input_name: x})[0]
        diff = np.abs(pt_logits.astype(np.float64) - ort_logits.astype(np.float64))
        results.append({
            "batch": batch,
            "seq_len": seq_len,
            "max_abs_diff": float(diff.max()),
            "argmax_agreement": float((pt_logits.argmax(-1) == ort_logits.argmax(-1)).mean()),
        })
    return results


def verify_checkpoint_onnx_parity(checkpoint_path, onnx_path, max_tokens=None, seed=PARITY_SEED):
    ckpt, cfg, model = load_checkpoint_model(checkpoint_path)
    shapes = parity_shapes(cfg, max_tokens)
    shape_results = run_parity(model, onnx_path, cfg, shapes, seed=seed)
    failures = []
    for r in shape_results:
        if r["argmax_agreement"] != 1.0:
            failures.append(
                f"B={r['batch']} T={r['seq_len']}: argmax agreement {r['argmax_agreement']:.4f} != 1.0"
            )
        if r["max_abs_diff"] >= MAX_ABS_DIFF_THRESHOLD:
            failures.append(
                f"B={r['batch']} T={r['seq_len']}: max abs logit diff {r['max_abs_diff']:.6g} "
                f">= threshold {MAX_ABS_DIFF_THRESHOLD:g}"
            )
    return {
        "checkpoint_path": os.path.abspath(checkpoint_path),
        "onnx_path": os.path.abspath(onnx_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "onnx_sha256": sha256_file(onnx_path),
        "step": ckpt.get("step"),
        "val_loss": ckpt.get("val_loss"),
        "config": dataclasses.asdict(cfg),
        "shapes": shape_results,
        "max_abs_diff_threshold": MAX_ABS_DIFF_THRESHOLD,
        "seed": seed,
        "passed": not failures,
        "failures": failures,
    }


def verify_tokenizer_pairing(tokenizer_path, cfg):
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    vocab_size = tokenizer.get_vocab_size(with_added_tokens=True)
    expected_specials = {
        "bos": ("<|bos|>", cfg.bos_token_id),
        "eos": ("<|eos|>", cfg.eos_token_id),
        "pad": ("<|pad|>", cfg.pad_token_id),
    }
    special_tokens = {}
    failures = []
    if vocab_size != cfg.vocab_size:
        failures.append(
            f"tokenizer vocab size {vocab_size} != config.vocab_size {cfg.vocab_size}"
        )
    for name, (token, expected_id) in expected_specials.items():
        token_id = tokenizer.token_to_id(token)
        special_tokens[name] = {"token": token, "id": token_id, "expected_id": expected_id}
        if token_id != expected_id:
            failures.append(
                f"tokenizer {token} id {token_id} != config {name}_token_id {expected_id}"
            )
    return {
        "tokenizer_path": os.path.abspath(tokenizer_path),
        "vocab_size": vocab_size,
        "expected_vocab_size": cfg.vocab_size,
        "special_tokens": special_tokens,
        "passed": not failures,
        "failures": failures,
    }


def format_parity_report(results):
    lines = [
        f"checkpoint: {results['checkpoint_path']}",
        f"  sha256: {results['checkpoint_sha256']}",
        f"onnx: {results['onnx_path']}",
        f"  sha256: {results['onnx_sha256']}",
        f"step: {results['step']}  val_loss: {results['val_loss']}",
        f"config: {results['config']}",
        f"pass criteria: argmax agreement 1.0 and max_abs_diff < "
        f"{results['max_abs_diff_threshold']:g} on every shape (seed {results['seed']})",
        f"{'batch':>5} {'seq_len':>7} {'max_abs_diff':>13} {'argmax_agree':>13}",
    ]
    for r in results["shapes"]:
        lines.append(
            f"{r['batch']:>5} {r['seq_len']:>7} {r['max_abs_diff']:>13.6g} {r['argmax_agreement']:>13.4f}"
        )
    for failure in results["failures"]:
        lines.append(f"FAIL: {failure}")
    lines.append("PARITY: " + ("PASS" if results["passed"] else "FAIL"))
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Verify that an ONNX export matches a PyTorch checkpoint before publication."
    )
    parser.add_argument("--checkpoint", required=True, help="Path to the .pt checkpoint")
    parser.add_argument("--onnx", required=True, help="Path to the .onnx export")
    parser.add_argument(
        "--tokenizer", default=None,
        help="Optional tokenizer.json to pair against the checkpoint config"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=None,
        help="Cap the sequence lengths exercised (further capped by config.max_seq_len)"
    )
    args = parser.parse_args(argv)

    if not os.path.exists(args.checkpoint):
        print(f"Error: checkpoint not found: {args.checkpoint}")
        return 2
    if not os.path.exists(args.onnx):
        print(f"Error: onnx not found: {args.onnx}")
        return 2

    print("Loading checkpoint and running parity matrix...")
    try:
        parity = verify_checkpoint_onnx_parity(args.checkpoint, args.onnx, max_tokens=args.max_tokens)
    except Exception as e:
        print(f"Error: verification could not run: {type(e).__name__}: {e}")
        return 2
    print(format_parity_report(parity))

    tokenizer_ok = True
    if args.tokenizer:
        if not os.path.exists(args.tokenizer):
            print(f"Error: tokenizer not found: {args.tokenizer}")
            return 2
        pairing = verify_tokenizer_pairing(args.tokenizer, ModelConfig(**parity["config"]))
        print(f"tokenizer: {pairing['tokenizer_path']}")
        print(f"  vocab_size: {pairing['vocab_size']} (expected {pairing['expected_vocab_size']})")
        for info in pairing["special_tokens"].values():
            print(f"  {info['token']}: id {info['id']} (expected {info['expected_id']})")
        for failure in pairing["failures"]:
            print(f"FAIL: {failure}")
        print("TOKENIZER PAIRING: " + ("PASS" if pairing["passed"] else "FAIL"))
        tokenizer_ok = pairing["passed"]

    if parity["passed"] and tokenizer_ok:
        print("OVERALL: PASS")
        return 0
    print("OVERALL: FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
