import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coolm.config import ModelConfig
from tools.verify_artifacts import (
    sha256_file,
    verify_checkpoint_onnx_parity,
    verify_tokenizer_pairing,
)

CHECKPOINT_PATH = "checkpoints/best_model.pt"
ONNX_PATH = "docs/model.onnx"
TOKENIZER_PATH = os.path.join("data", "tokenizer.json")

# FIND-004: the val metrics above are real measured values of the shipped
# (template-contaminated) split; the names and this section say so explicitly.
LIMITATIONS_MD = """

## Limitations

- **Template-contaminated validation split:** the shipped random 95/5 validation split has 95.02% of its outputs appearing verbatim in the training set (independent audit, 2026-11-05). The validation loss/perplexity metrics above measure template retrieval, **not** open-domain generalization.
- **Template-only response diversity:** the 500K dataset draws its responses from ~752 distinct response templates (51,691 distinct output strings across 500,000 rows, ~89.7% duplicate-output rate); inputs are 100% unique.
- **Relevance cliff outside the training distribution:** measured relevance is 1.96/2 on verbatim-style training questions, 1.15/2 on paraphrased variations, and 0.32/2 on out-of-domain prompts (audit 2026-11-05, 462 generations).
- **Artifact consistency:** verify that model.onnx and best_model.pt in this repo come from the same training run; see tools/verify_artifacts.py.
- For reproducible template-disjoint / topic-disjoint evaluation with per-split leakage statistics, run `python -m tools.evaluate`.
"""


def provenance_sidecar_path(onnx_path):
    return onnx_path + ".provenance.json"


def check_publish_consistency(checkpoint_path=CHECKPOINT_PATH, onnx_path=ONNX_PATH,
                              tokenizer_path=TOKENIZER_PATH):
    """Artifact-consistency gate run before publishing (FIND-005).

    Refuses publication unless the ONNX is proven to come from the exact
    checkpoint being uploaded. Purely local: no network, no huggingface_hub
    import, safe to call from tests. Returns (allowed, reasons).
    """
    if not os.path.exists(onnx_path):
        # Not publishing an ONNX, so there is nothing to cross-check.
        return True, ["no onnx artifact present, consistency gate skipped"]
    if not os.path.exists(checkpoint_path):
        return False, [f"checkpoint not found: {checkpoint_path}"]

    sidecar_path = provenance_sidecar_path(onnx_path)
    if not os.path.exists(sidecar_path):
        return False, [
            f"provenance sidecar missing: {sidecar_path}",
            "re-export with tools/model_export.py before publishing model.onnx",
        ]

    with open(sidecar_path, "r", encoding="utf-8") as f:
        provenance = json.load(f)

    checkpoint_sha256 = sha256_file(checkpoint_path)
    if provenance.get("checkpoint_sha256") != checkpoint_sha256:
        return False, [
            "checkpoint/ONNX provenance mismatch:",
            f"  sidecar checkpoint_sha256: {provenance.get('checkpoint_sha256')}",
            f"  actual {checkpoint_path}:  {checkpoint_sha256}",
            "model.onnx was not exported from the checkpoint being uploaded",
        ]

    parity = verify_checkpoint_onnx_parity(checkpoint_path, onnx_path)
    if not parity["passed"]:
        return False, (
            ["live checkpoint/ONNX parity check failed:"]
            + ["  " + failure for failure in parity["failures"]]
        )

    reasons = [
        f"provenance ok (checkpoint {checkpoint_sha256[:12]}...)",
        f"live parity check passed ({len(parity['shapes'])} shapes, "
        f"max_abs_diff {max(r['max_abs_diff'] for r in parity['shapes']):.3g})",
    ]

    if tokenizer_path is not None and os.path.exists(tokenizer_path):
        pairing = verify_tokenizer_pairing(tokenizer_path, ModelConfig(**parity["config"]))
        if not pairing["passed"]:
            return False, (
                ["tokenizer does not pair with the checkpoint:"]
                + ["  " + failure for failure in pairing["failures"]]
            )
        reasons.append("tokenizer pairing passed")

    return True, reasons


def _require_huggingface_hub():
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print("Missing dependency: the 'huggingface_hub' package is required by tools/export_hf.py")
        print("Install it with: pip install huggingface_hub")
        raise SystemExit(1)
    return HfApi, create_repo


def push_to_hub(repo_id, token=None):
    """Publish the gated artifacts to ``repo_id``.

    FIND-024 remainders:
    - ``repo_id`` is REQUIRED (no placeholder default) — a missing repo id is
      an error before anything is uploaded.
    - The token comes ONLY from the HF_TOKEN environment variable (secrets in
      argv are visible in process listings).
    - Every file upload is individually guarded; a single failure is reported
      in the summary and makes the process exit non-zero instead of
      half-silently pushing a partial release.
    """
    if not repo_id or not str(repo_id).strip():
        print("Error: a HuggingFace repo id is required (e.g. --repo_id MdHussain121/coolm-42M).")
        print("The placeholder default was removed — pass the real target explicitly.")
        return 1
    if not token:
        token = os.environ.get("HF_TOKEN")
    if not token:
        print("Error: no HuggingFace token found.")
        print("Set the HF_TOKEN environment variable (tokens are deliberately not")
        print("accepted as a CLI argument so secrets never appear in process listings).")
        return 1

    HfApi, create_repo = _require_huggingface_hub()

    # Gate BEFORE any upload: publishing a checkpoint without a matching ONNX
    # (or vice versa) is exactly how the mismatched artifacts shipped.
    allowed, reasons = check_publish_consistency()
    for reason in reasons:
        print(reason)
    if not allowed:
        print("Aborting upload: artifact consistency gate refused publication.")
        return 1
    print("Artifact consistency gate passed.")

    api = HfApi(token=token)

    print(f"Creating repository {repo_id}...")
    try:
        create_repo(repo_id, exist_ok=True, token=token)
    except Exception as e:
        print(f"Error creating repo: {e}")
        return 1

    failures = []

    def upload(path_or_fileobj, path_in_repo, description):
        try:
            api.upload_file(
                path_or_fileobj=path_or_fileobj,
                path_in_repo=path_in_repo,
                repo_id=repo_id,
            )
        except Exception as e:
            failures.append(path_in_repo)
            print(f"  ERROR uploading {description} ({path_in_repo}): {type(e).__name__}: {e}")

    print("Uploading model weights...")
    if os.path.exists(CHECKPOINT_PATH):
        print(f"  sha256 {CHECKPOINT_PATH}: {sha256_file(CHECKPOINT_PATH)}")
        upload(CHECKPOINT_PATH, "best_model.pt", "model weights")
    else:
        print(f"  skipped (not found: {CHECKPOINT_PATH})")

    print("Uploading tokenizer...")
    if os.path.exists(TOKENIZER_PATH):
        print(f"  sha256 {TOKENIZER_PATH}: {sha256_file(TOKENIZER_PATH)}")
        upload(TOKENIZER_PATH, "tokenizer.json", "tokenizer")
    else:
        print(f"  skipped (not found: {TOKENIZER_PATH})")

    print("Uploading ONNX model...")
    if os.path.exists(ONNX_PATH):
        print(f"  sha256 {ONNX_PATH}: {sha256_file(ONNX_PATH)}")
        upload(ONNX_PATH, "model.onnx", "ONNX model")
    else:
        print(f"  skipped (not found: {ONNX_PATH})")

    print("Uploading README.md (Model Card)...")
    if os.path.exists("README.md"):
        with open("README.md", "r", encoding="utf-8") as f:
            readme_content = f.read()

        yaml_frontmatter = """---
license: mit
language:
- en
tags:
- text-generation
- pytorch
- onnx
- tiny-lm
- custom-architecture
- pigeon
- rooftop-pigeon
model-index:
- name: CooLM-42M
  results:
  - task:
      type: text-generation
      name: Text Generation
    dataset:
      name: coolm-500k
      type: MdHussain121/coolm-500k
    metrics:
    - type: perplexity
      value: 1.39
      name: Validation Perplexity (template-retrieval, in-distribution)
    - type: loss
      value: 0.3262
      name: Validation Loss (template-retrieval, in-distribution)
---

"""
        combined_content = yaml_frontmatter + readme_content + LIMITATIONS_MD
        upload(io.BytesIO(combined_content.encode("utf-8")), "README.md", "model card")
    else:
        print("  skipped (not found: README.md)")

    print("Uploading assets...")
    if os.path.exists("assets"):
        for f in sorted(os.listdir("assets")):
            file_path = os.path.join("assets", f)
            if os.path.isfile(file_path):
                print(f"Uploading asset: {f}...")
                upload(file_path, f"assets/{f}", f"asset {f}")

    if failures:
        print("\nUpload FAILED for the following files (nothing half-silent):")
        for path_in_repo in failures:
            print(f"  - {path_in_repo}")
        print(f"{len(failures)} upload(s) failed; the release is incomplete. Re-run after fixing.")
        return 1

    print("Done! Model pushed to HuggingFace.")
    return 0

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Publish the CooLM artifacts to Hugging Face (gated by tools/verify_artifacts checks)."
    )
    parser.add_argument("--repo_id", type=str, required=True,
                        help="HuggingFace repo ID (required, e.g. MdHussain121/coolm-42M)")
    # FIND-024: no --token flag — secrets in argv leak via process listings;
    # the token is read from the HF_TOKEN environment variable only.
    args = parser.parse_args()
    sys.exit(push_to_hub(repo_id=args.repo_id))
