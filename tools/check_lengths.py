"""Check sample length distribution (non-padding content).

Small standalone diagnostic (FIND-061): length percentiles, the share of
rows ending with <|eos|>, and an explicit flag for rows at/over pad_len.
"""
import os

import torch

# FIND-040: default data paths resolve against the repo root, not the CWD.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# <|eos|> token id (matches coolm.config.ModelConfig / the trained tokenizer).
EOS_ID = 2


def main():
    data_path = os.path.join(_REPO_ROOT, "data", "train_data.pt")
    mask_path = os.path.join(_REPO_ROOT, "data", "train_mask.pt")
    for path in (data_path, mask_path):
        if not os.path.exists(path):
            raise SystemExit(f"error: required data file not found: {path}")

    data = torch.load(data_path, map_location="cpu", weights_only=True)
    mask = torch.load(mask_path, map_location="cpu", weights_only=True)

    # Count non-padding tokens per sample
    content_lengths = (data != 0).sum(dim=1).tolist()
    output_lengths = mask.sum(dim=1).tolist()

    content_lengths.sort()
    output_lengths.sort()
    n = len(content_lengths)
    pad_len = data.size(1)

    print(f"samples: {n}")
    print(f"padded length: {pad_len}")
    print(f"\n--- Content tokens (non-padding) ---")
    print(f"min: {content_lengths[0]}, max: {content_lengths[-1]}")
    print(f"mean: {sum(content_lengths)/n:.1f}")
    for p in [50, 75, 90, 95, 99, 100]:
        idx = min(int(n * p / 100), n - 1)
        print(f"  {p}th percentile: {content_lengths[idx]}")

    print(f"\n--- Output tokens (mask=1, loss-active) ---")
    print(f"min: {output_lengths[0]}, max: {output_lengths[-1]}")
    print(f"mean: {sum(output_lengths)/n:.1f}")
    for p in [50, 75, 90, 95, 99, 100]:
        idx = min(int(n * p / 100), n - 1)
        print(f"  {p}th percentile: {output_lengths[idx]}")

    # FIND-061: explicit over-length flag — rows whose non-padding content
    # fills (or would exceed) the padded window, i.e. candidates for the
    # disabled truncation path.
    over_length = sum(1 for length in content_lengths if length >= pad_len)
    print(f"\n--- Flags ---")
    print(f"rows at/over pad_len ({pad_len}): {over_length} ({over_length/n*100:.4f}%)")

    # FIND-061: share of rows whose last loss-active token is <|eos|>
    # (termination learned by the model depends on this).
    has_output = mask.sum(dim=1) > 0
    rows_with_output = int(has_output.sum())
    last_active_idx = mask.size(1) - 1 - mask.flip(1).argmax(dim=1)
    last_tokens = data[torch.arange(data.size(0)), last_active_idx]
    eos_endings = int((last_tokens[has_output] == EOS_ID).sum())
    denom = max(rows_with_output, 1)
    print(f"rows ending with <|eos|>: {eos_endings}/{rows_with_output} "
          f"({eos_endings/denom*100:.2f}% of rows with loss-active output)")


if __name__ == "__main__":
    main()
