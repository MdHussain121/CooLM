"""Template-disjoint, leakage-measuring evaluation for CooLM (FIND-004).

The documented val loss 0.3262 (ppl 1.39) is a template-memorization figure:
the shipped random 95/5 split puts ~95% of validation outputs verbatim in
train. This tool evaluates a checkpoint on three splits and prints a leakage
table for every one:

- shipped_split: the exact training split (seeded shuffle, 95/5, as in
  coolm/train.py), honestly labeled as in-distribution template retrieval;
- response_template_disjoint: whole exact-output groups assigned to val
  (0% output verbatim overlap by construction);
- topic_disjoint: entire categories held out of train.

Loss math replicates train.py exactly: token-weighted
sum(loss * active) / sum(active) over the masked response tokens of
[bos] + prompt + response + [eos] rows built like coolm/prepare_data.py.
"""

import argparse
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch
from tokenizers import Tokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coolm.checkpoint import load_trained_model
from tools.verify_artifacts import sha256_file

SHIPPED_SEED = 42
SHIPPED_TRAIN_FRACTION = 0.95
TEMPLATE_VAL_FRACTION = 0.05
DEFAULT_TOPIC_HOLDOUT = 2
BATCH_SIZE = 64
DEFAULT_PAD_LEN = 64

SHIPPED_LABEL = (
    "template-retrieval (in-distribution); outputs largely verbatim-repeated in train"
)
TEMPLATE_DISJOINT_LABEL = (
    "response-template-disjoint; every val output string is absent from train "
    "by construction"
)
TOPIC_DISJOINT_LABEL = (
    "topic-disjoint; held-out categories appear only in val"
)

LIMITATIONS = [
    "The shipped random 95/5 validation split is template-contaminated: the "
    "2026-11-05 audit measured 95.02% of validation outputs appearing verbatim "
    "in the training set (100% response-template overlap, 99.48% "
    "(input, output)-pair overlap). Its loss/perplexity measures template "
    "retrieval, not open-domain generalization.",
    "The 500K dataset draws responses from ~752 distinct response templates: "
    "51,691 distinct output strings across 500,000 rows (~89.7% duplicate-output "
    "rate); inputs are 100% unique.",
    "Measured relevance drops outside the training distribution: 1.96/2 on "
    "verbatim-style training questions, 1.15/2 on paraphrased variations, "
    "0.32/2 on out-of-domain prompts (audit 2026-11-05, 462 generations).",
    "Artifact consistency: verify that model.onnx and best_model.pt in this "
    "repo come from the same training run; see tools/verify_artifacts.py.",
]


def load_jsonl_rows(data_path):
    """Stream a jsonl dataset, validating the {"input","output","category"}
    all-strings schema. Malformed lines are skipped and counted.

    Returns (rows, malformed_count) with rows as (input, output, category).
    """
    rows = []
    malformed = 0
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            if not isinstance(obj, dict):
                malformed += 1
                continue
            inp = obj.get("input")
            out = obj.get("output")
            cat = obj.get("category")
            if not (isinstance(inp, str) and isinstance(out, str) and isinstance(cat, str)):
                malformed += 1
                continue
            rows.append((inp, out, cat))
    return rows, malformed


def build_shipped_split(n_rows, seed=SHIPPED_SEED):
    """Replicate the training split exactly (coolm/train.py): seeded shuffle
    then 95/5. Val is the tail of the permutation. Returns a set of indices."""
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n_rows, generator=generator)
    n_train = int(SHIPPED_TRAIN_FRACTION * n_rows)
    return set(indices[n_train:].tolist())


def build_template_disjoint_split(rows, seed=SHIPPED_SEED):
    """Assign whole exact-output groups to val until ~5% of rows are val.

    Every row sharing one exact output string goes entirely to val or
    entirely to train, so 0% output verbatim overlap is guaranteed.
    Returns a set of indices.
    """
    groups = defaultdict(list)
    for i, (_, output, _) in enumerate(rows):
        groups[output].append(i)
    group_lists = list(groups.values())
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(group_lists), generator=generator).tolist()
    target = max(1, round(TEMPLATE_VAL_FRACTION * len(rows)))
    val = set()
    val_count = 0
    for g in order:
        if val_count >= target:
            break
        val.update(group_lists[g])
        val_count += len(group_lists[g])
    return val


def build_topic_disjoint_split(rows, holdout_categories=None):
    """Hold out entire categories from train (default: the K largest by row
    count, name tie-break for determinism). Returns (val_index_set,
    held_out_names, category_counts)."""
    counts = Counter(cat for _, _, cat in rows)
    if holdout_categories:
        held_out = [c.strip() for c in holdout_categories.split(",") if c.strip()]
        missing = [c for c in held_out if c not in counts]
        if missing:
            raise SystemExit(
                f"error: --holdout-categories not present in the dataset: {missing}; "
                f"available: {sorted(counts)}"
            )
        if len(held_out) == len(counts):
            raise SystemExit(
                "error: --holdout-categories lists every category; "
                "nothing would remain for train"
            )
    else:
        ranked = sorted(counts, key=lambda c: (-counts[c], c))
        held_out = ranked[:min(DEFAULT_TOPIC_HOLDOUT, len(ranked) - 1)]
    held = set(held_out)
    val = {i for i, (_, _, cat) in enumerate(rows) if cat in held}
    return val, held_out, dict(counts)


def leakage_stats(train_rows, val_rows):
    """Leakage table for one split: exact-string membership of every val
    row's output/input in the train side, plus output-template overlap."""
    train_inputs = {r[0] for r in train_rows}
    train_outputs = {r[1] for r in train_rows}
    n_val = len(val_rows)
    output_hits = sum(1 for r in val_rows if r[1] in train_outputs)
    input_hits = sum(1 for r in val_rows if r[0] in train_inputs)
    val_templates = {r[1] for r in val_rows}
    templates_in_train = sum(1 for t in val_templates if t in train_outputs)
    return {
        "val_rows": n_val,
        "train_rows": len(train_rows),
        "val_output_in_train_count": output_hits,
        "val_output_in_train_pct": 100.0 * output_hits / n_val if n_val else 0.0,
        "val_input_in_train_count": input_hits,
        "val_input_in_train_pct": 100.0 * input_hits / n_val if n_val else 0.0,
        "val_distinct_output_templates": len(val_templates),
        "val_output_templates_in_train": templates_in_train,
        "val_output_template_overlap_pct": (
            100.0 * templates_in_train / len(val_templates) if val_templates else 0.0
        ),
    }


def resolve_special_ids(tokenizer, cfg):
    """Resolve pad/bos/eos ids from the tokenizer, falling back to the
    checkpoint config values when a token literal is absent."""
    ids = {}
    for name, token in (("pad", "<|pad|>"), ("bos", "<|bos|>"), ("eos", "<|eos|>")):
        tid = tokenizer.token_to_id(token)
        if tid is None:
            if cfg is None or not hasattr(cfg, f"{name}_token_id"):
                raise SystemExit(
                    f"error: tokenizer has no '{token}' token and no checkpoint "
                    f"config to fall back on"
                )
            tid = getattr(cfg, f"{name}_token_id")
        ids[name] = tid
    return ids


def tokenize_row(tokenizer, row, pad_len, ids):
    """Build one [bos] + prompt + response + [eos] row exactly like
    coolm/prepare_data.py: mask 1 only on the response (+ <|im_end|> + eos),
    head-truncate to pad_len, pad with mask 0."""
    input_text = f"<|im_start|>user\n{row[0]}<|im_end|>\n<|im_start|>assistant\n"
    output_text = f"{row[1]}<|im_end|>"
    input_ids = [ids["bos"]] + tokenizer.encode(input_text).ids
    output_ids = tokenizer.encode(output_text).ids + [ids["eos"]]
    full_ids = input_ids + output_ids
    mask = [0] * len(input_ids) + [1] * len(output_ids)
    if len(full_ids) > pad_len:
        full_ids = full_ids[:pad_len]
        mask = mask[:pad_len]
    if len(full_ids) < pad_len:
        n = pad_len - len(full_ids)
        full_ids = full_ids + [ids["pad"]] * n
        mask = mask + [0] * n
    return (
        torch.tensor(full_ids, dtype=torch.long),
        torch.tensor(mask, dtype=torch.long),
    )


def compute_split_loss(model, tokenized, device, batch_size=BATCH_SIZE):
    """Token-weighted loss exactly like train.py's eval loop:
    sum(loss * active) / sum(active). Returns (loss, active_tokens)."""
    total_loss = 0.0
    total_active = 0
    with torch.no_grad():
        for start in range(0, len(tokenized), batch_size):
            chunk = tokenized[start:start + batch_size]
            padded = torch.stack([t for t, _ in chunk]).to(device)
            mask = torch.stack([m for _, m in chunk]).to(device)
            x = padded[:, :-1].contiguous()
            targets = padded[:, 1:].contiguous()
            target_mask = mask[:, 1:].contiguous()
            targets[target_mask == 0] = -100
            n_active = (targets != -100).sum().item()
            if n_active == 0:
                continue
            with torch.amp.autocast(device_type=device, enabled=(device == "cuda")):
                _, loss, _ = model(x, targets=targets)
            total_loss += loss.item() * n_active
            total_active += n_active
    return total_loss / max(total_active, 1), total_active


def _subsample_val(val_rows, max_samples, seed):
    generator = torch.Generator().manual_seed(seed)
    picks = torch.randperm(len(val_rows), generator=generator)[:max_samples]
    return [val_rows[i] for i in sorted(picks.tolist())]


def evaluate_dataset(data_path, tokenizer_path, checkpoint_path=None,
                     max_samples=None, seed=SHIPPED_SEED, holdout_categories=None):
    """Build the three splits, measure leakage on each, compute val loss when
    a checkpoint is given. Returns the full report dict."""
    t_start = time.time()

    rows, malformed = load_jsonl_rows(data_path)
    if not rows:
        raise SystemExit(f"error: no valid rows in dataset: {data_path}")
    print(f"dataset: {data_path}")
    print(f"  valid rows: {len(rows):,}  malformed skipped: {malformed:,}")

    distinct_inputs = len({r[0] for r in rows})
    distinct_outputs = len({r[1] for r in rows})
    category_counts = Counter(r[2] for r in rows)
    dataset_stats = {
        "path": os.path.abspath(data_path),
        "sha256": sha256_file(data_path),
        "rows": len(rows),
        "malformed_rows_skipped": malformed,
        "distinct_inputs": distinct_inputs,
        "distinct_outputs": distinct_outputs,
        "duplicate_output_pct": 100.0 * (1.0 - distinct_outputs / len(rows)),
        "categories": {
            cat: category_counts[cat]
            for cat in sorted(category_counts, key=lambda c: -category_counts[c])
        },
    }

    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = None
    cfg = None
    ckpt_stats = None
    if checkpoint_path:
        model, cfg, ckpt = load_trained_model(checkpoint_path, device=device)
        model.eval()
        ckpt_stats = {
            "path": os.path.abspath(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
            "step": ckpt.get("step"),
            "train_val_loss": ckpt.get("val_loss"),
        }
        print(f"checkpoint: {checkpoint_path} (step {ckpt.get('step')}, "
              f"train-time val loss {ckpt.get('val_loss')})")
    else:
        print("note: no --checkpoint given; skipping loss computation "
              "(split/leakage statistics only)")
    print(f"device: {device}")

    pad_len = cfg.max_seq_len if cfg is not None else DEFAULT_PAD_LEN
    ids = resolve_special_ids(tokenizer, cfg)

    shipped_val = build_shipped_split(len(rows), seed)
    template_val = build_template_disjoint_split(rows, seed)
    topic_val, held_out, _ = build_topic_disjoint_split(rows, holdout_categories)

    split_specs = [
        ("shipped_split", SHIPPED_LABEL, shipped_val),
        ("response_template_disjoint", TEMPLATE_DISJOINT_LABEL, template_val),
        ("topic_disjoint", TOPIC_DISJOINT_LABEL, topic_val),
    ]

    split_reports = []
    for name, label, val_idx in split_specs:
        train_rows = [rows[i] for i in range(len(rows)) if i not in val_idx]
        val_rows = [rows[i] for i in sorted(val_idx)]
        if max_samples and len(val_rows) > max_samples:
            val_rows = _subsample_val(val_rows, max_samples, seed)

        split_report = {
            "name": name,
            "label": label,
            "n_train": len(train_rows),
            "n_val": len(val_rows),
            "val_row_pct": 100.0 * len(val_rows) / len(rows),
            "leakage": {},
            "loss": None,
            "ppl": None,
        }
        if name == "response_template_disjoint":
            split_report["n_distinct_output_groups"] = len({r[1] for r in rows})
        if name == "topic_disjoint":
            split_report["held_out_categories"] = held_out
            split_report["held_out_category_counts"] = {
                c: category_counts[c] for c in held_out
            }
            split_report["train_category_counts"] = {
                c: n for c, n in dataset_stats["categories"].items()
                if c not in set(held_out)
            }

        leak = leakage_stats(train_rows, val_rows)
        tokenized = [tokenize_row(tokenizer, r, pad_len, ids) for r in val_rows]
        leak["val_active_tokens"] = sum(int(m.sum().item()) for _, m in tokenized)
        split_report["leakage"] = leak

        if model is not None:
            loss, _ = compute_split_loss(model, tokenized, device)
            split_report["loss"] = loss
            split_report["ppl"] = math.exp(loss)
        split_reports.append(split_report)
        print(f"  evaluated split: {name} "
              f"(train {len(train_rows):,} / val {len(val_rows):,})")

    return {
        "tool": "tools/evaluate.py",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "device": device,
        "pad_len": pad_len,
        "batch_size": BATCH_SIZE,
        "runtime_seconds": round(time.time() - t_start, 2),
        "dataset": dataset_stats,
        "tokenizer": {
            "path": os.path.abspath(tokenizer_path),
            "sha256": sha256_file(tokenizer_path),
        },
        "checkpoint": ckpt_stats,
        "splits": split_reports,
        "limitations": LIMITATIONS,
    }


def print_report(report):
    """Per-split leakage tables plus the final summary table."""
    ds = report["dataset"]
    print(f"\ndataset: {ds['path']}")
    print(f"  sha256: {ds['sha256']}")
    print(f"  rows: {ds['rows']:,}  distinct inputs: {ds['distinct_inputs']:,}  "
          f"distinct outputs: {ds['distinct_outputs']:,}  "
          f"duplicate-output rate: {ds['duplicate_output_pct']:.2f}%")

    for s in report["splits"]:
        leak = s["leakage"]
        print("\n" + "-" * 78)
        print(f"split: {s['name']}")
        print(f"  label: {s['label']}")
        print(f"  rows: train {s['n_train']:,} | val {s['n_val']:,} "
              f"({s['val_row_pct']:.2f}% of dataset)")
        if "held_out_categories" in s:
            print(f"  held-out categories: "
                  f"{', '.join(s['held_out_categories'])}")
            for cat, count in s["held_out_category_counts"].items():
                print(f"    {cat}: {count:,} rows")
        print("  leakage (val vs train):")
        print(f"    val outputs verbatim in train      : "
              f"{leak['val_output_in_train_pct']:7.2f}%  "
              f"({leak['val_output_in_train_count']:,} of {leak['val_rows']:,} rows)")
        print(f"    val inputs verbatim in train       : "
              f"{leak['val_input_in_train_pct']:7.2f}%  "
              f"({leak['val_input_in_train_count']:,} of {leak['val_rows']:,} rows)")
        print(f"    distinct val output templates      : "
              f"{leak['val_distinct_output_templates']:,}")
        print(f"    val output templates found in train: "
              f"{leak['val_output_templates_in_train']:,} "
              f"({leak['val_output_template_overlap_pct']:.2f}% overlap)")
        print(f"    val active tokens                  : "
              f"{leak['val_active_tokens']:,}")
        if s["loss"] is not None:
            print(f"  val loss: {s['loss']:.4f} (ppl {s['ppl']:.2f})")
        else:
            print("  val loss: not computed (no --checkpoint given)")

    print("\n" + "=" * 78)
    print(f"summary (seed {report['seed']}, batch {report['batch_size']}, "
          f"pad_len {report['pad_len']}, device {report['device']})")
    print("=" * 78)
    print(f"{'split':<30} {'val rows':>9} {'out-ovlp%':>10} {'tmpl-ovlp%':>11} "
          f"{'loss':>9} {'ppl':>7}")
    for s in report["splits"]:
        leak = s["leakage"]
        loss_str = f"{s['loss']:.4f}" if s["loss"] is not None else "--"
        ppl_str = f"{s['ppl']:.2f}" if s["ppl"] is not None else "--"
        print(f"{s['name']:<30} {leak['val_rows']:>9,} "
              f"{leak['val_output_in_train_pct']:>10.2f} "
              f"{leak['val_output_template_overlap_pct']:>11.2f} "
              f"{loss_str:>9} {ppl_str:>7}")
    print("=" * 78)
    print("interpretation:")
    print("  shipped_split is in-distribution template retrieval; its loss is")
    print("  NOT a generalization metric (see 'limitations' in the JSON report).")
    print("  response_template_disjoint / topic_disjoint measure generalization")
    print("  to unseen response templates / topics.")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Template-disjoint, leakage-measuring evaluation (FIND-004)."
    )
    parser.add_argument("--data", required=True,
                        help="Path to the dataset .jsonl (input/output/category)")
    parser.add_argument("--tokenizer", required=True,
                        help="Path to tokenizer.json")
    parser.add_argument("--checkpoint", default=None,
                        help="Optional .pt checkpoint to compute val loss with")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Cap the number of val rows evaluated per split "
                             "(deterministic seeded subsample, for smoke runs)")
    parser.add_argument("--seed", type=int, default=SHIPPED_SEED,
                        help="Split seed (default 42, matches training)")
    parser.add_argument("--holdout-categories", default=None,
                        help="Comma-separated category names for the "
                             "topic-disjoint split (default: the 2 largest)")
    parser.add_argument("--out", default=None,
                        help="Write the JSON report to this path "
                             "(default: print to stdout)")
    args = parser.parse_args(argv)

    if not os.path.isfile(args.data):
        print(f"error: dataset not found: {args.data}", file=sys.stderr)
        print("produce it with:", file=sys.stderr)
        print("  python generate_data.py 500000", file=sys.stderr)
        print("  python -m coolm.prepare_data pigeon_data.jsonl", file=sys.stderr)
        return 2
    if not os.path.isfile(args.tokenizer):
        print(f"error: tokenizer not found: {args.tokenizer}", file=sys.stderr)
        print("produce it with: python -m coolm.prepare_data pigeon_data.jsonl",
              file=sys.stderr)
        return 2
    if args.checkpoint and not os.path.isfile(args.checkpoint):
        print(f"error: checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 2

    report = evaluate_dataset(
        data_path=args.data,
        tokenizer_path=args.tokenizer,
        checkpoint_path=args.checkpoint,
        max_samples=args.max_samples,
        seed=args.seed,
        holdout_categories=args.holdout_categories,
    )
    print_report(report)

    if args.out:
        out_path = os.path.abspath(args.out)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        Path(out_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nreport written to {out_path}")
    else:
        print("\n" + json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
