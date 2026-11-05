"""FIND-004 regression tests: template-disjoint, leakage-measuring evaluation.

Everything runs on small synthetic datasets plus the conftest tiny
checkpoint/tokenizer. The real 500K dataset and the 498MB production
checkpoint are explicitly NOT part of pytest — the coordinator-evidence run
is executed separately via `python -m tools.evaluate`.
"""
import json
import math
import os
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tools import evaluate as ev

CATEGORIES = ("alpha", "beta", "gamma")
_OUTPUT_WORDS = ["aaa", "bbb", "ccc", "ddd", "eee", "fff",
                 "ggg", "hhh", "iii", "jjj", "kkk", "lll"]


def make_synthetic_rows():
    """12 distinct outputs (4 per category), each appearing 3 or 4 times.

    Output j appears 3 times when j is even, 4 when odd -> 42 rows, 14 per
    category, every input unique.
    """
    rows = []
    n = 0
    for j, word in enumerate(_OUTPUT_WORDS):
        for _ in range(3 if j % 2 == 0 else 4):
            rows.append({
                "input": f"unique prompt number {n}",
                "output": f"{word} reply",
                "category": CATEGORIES[j // 4],
            })
            n += 1
    return rows


def write_jsonl(path, rows):
    path = Path(path)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    return str(path)


def rows_as_tuples(rows):
    """The split/leakage functions consume (input, output, category) tuples,
    exactly what tools.evaluate.load_jsonl_rows produces from disk."""
    return [(r["input"], r["output"], r["category"]) for r in rows]


def write_test_tokenizer(path):
    """Small WordLevel tokenizer covering the synthetic vocabulary."""
    from tokenizers import Tokenizer, models, pre_tokenizers

    specials = ["<|pad|>", "<|bos|>", "<|eos|>", "<|unk|>",
                "<|im_start|>", "<|im_end|>"]
    words = _OUTPUT_WORDS + ["reply", "unique", "prompt", "number"]
    vocab = {tok: i for i, tok in enumerate(specials)}
    vocab.update({word: i for i, word in enumerate(words, start=len(specials))})
    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<|unk|>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.add_special_tokens(specials)
    tokenizer.save(str(path))
    return str(path)


@pytest.fixture()
def synthetic_dataset(tmp_path):
    rows = make_synthetic_rows()
    data_path = write_jsonl(tmp_path / "synthetic.jsonl", rows)
    tok_path = write_test_tokenizer(tmp_path / "tokenizer.json")
    return data_path, tok_path, rows


def test_template_disjoint_split_zero_overlap_and_row_share(synthetic_dataset):
    data_path, tok_path, rows = synthetic_dataset
    report = ev.evaluate_dataset(data_path=data_path, tokenizer_path=tok_path)
    td = {s["name"]: s for s in report["splits"]}["response_template_disjoint"]

    leak = td["leakage"]
    assert leak["val_output_in_train_count"] == 0
    assert leak["val_output_in_train_pct"] == 0.0
    assert leak["val_output_templates_in_train"] == 0
    assert leak["val_output_template_overlap_pct"] == 0.0

    # Whole-group granularity: the closest achievable val share above the 5%
    # target on this dataset is one output-group (3 or 4 of 42 rows).
    assert 0.0 < td["val_row_pct"] <= 10.0
    assert abs(td["val_row_pct"] - 5.0) <= 5.0
    assert td["n_val"] in (3, 4)
    assert td["leakage"]["val_rows"] == td["n_val"]


def test_template_disjoint_split_deterministic():
    rows = rows_as_tuples(make_synthetic_rows())
    assert ev.build_template_disjoint_split(rows, seed=42) == \
        ev.build_template_disjoint_split(rows, seed=42)


def test_topic_disjoint_split_holds_out_whole_categories(synthetic_dataset):
    _, _, rows = synthetic_dataset
    rows = rows_as_tuples(rows)

    val, held_out, counts = ev.build_topic_disjoint_split(rows, None)
    # All counts tie at 14; the name tie-break keeps this deterministic.
    assert held_out == ["alpha", "beta"]
    train_cats = {rows[i][2] for i in range(len(rows)) if i not in val}
    val_cats = {rows[i][2] for i in val}
    assert train_cats == {"gamma"}
    assert val_cats == {"alpha", "beta"}

    val2, held_out2, _ = ev.build_topic_disjoint_split(rows, "gamma")
    assert held_out2 == ["gamma"]
    assert {rows[i][2] for i in val2} == {"gamma"}
    assert all(rows[i][2] != "gamma" for i in range(len(rows)) if i not in val2)

    with pytest.raises(SystemExit, match="not present"):
        ev.build_topic_disjoint_split(rows, "nonexistent")


def test_shipped_split_seed42_deterministic_and_matches_train_py():
    v1 = ev.build_shipped_split(1000, seed=42)
    v2 = ev.build_shipped_split(1000, seed=42)
    assert v1 == v2
    assert len(v1) == 1000 - int(0.95 * 1000)

    # Independently re-derive the split from coolm/train.py's exact shuffle:
    # seeded permutation, val is the tail after the 95% cut.
    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(1000, generator=generator)
    assert v1 == set(indices[int(0.95 * 1000):].tolist())


def test_leakage_stats_exact_by_construction():
    train_rows = [
        ("q1", "A", "c1"), ("q2", "B", "c1"), ("q3", "C", "c2"), ("qx", "A", "c2"),
    ]
    val_rows = [
        ("q1", "C", "c1"), ("q9", "D", "c1"), ("q8", "E", "c2"), ("q7", "C", "c2"),
    ]
    stats = ev.leakage_stats(train_rows, val_rows)
    assert stats["val_rows"] == 4
    assert stats["train_rows"] == 4
    assert stats["val_output_in_train_count"] == 2  # both "C" rows
    assert stats["val_output_in_train_pct"] == 50.0
    assert stats["val_input_in_train_count"] == 1   # only "q1"
    assert stats["val_input_in_train_pct"] == 25.0
    assert stats["val_distinct_output_templates"] == 3  # C, D, E
    assert stats["val_output_templates_in_train"] == 1
    assert stats["val_output_template_overlap_pct"] == pytest.approx(100.0 / 3)


def test_shipped_split_report_matches_independent_leakage_computation(
        synthetic_dataset):
    data_path, tok_path, rows = synthetic_dataset
    report = ev.evaluate_dataset(data_path=data_path, tokenizer_path=tok_path)
    shipped = {s["name"]: s for s in report["splits"]}["shipped_split"]

    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(len(rows), generator=generator)
    val_idx = set(indices[int(0.95 * len(rows)):].tolist())
    train_outputs = {rows[i]["output"] for i in range(len(rows)) if i not in val_idx}
    expected_hits = sum(1 for i in val_idx if rows[i]["output"] in train_outputs)

    leak = shipped["leakage"]
    assert leak["val_rows"] == len(val_idx)
    assert leak["val_output_in_train_count"] == expected_hits
    assert leak["val_output_in_train_pct"] == pytest.approx(
        100.0 * expected_hits / len(val_idx))
    # Inputs are unique by construction -> zero input leakage on any split.
    assert leak["val_input_in_train_pct"] == 0.0


def test_loss_computation_tiny_checkpoint(infer_tiny_checkpoint,
                                         infer_tiny_tokenizer, tmp_path):
    rows = make_synthetic_rows()
    data_path = write_jsonl(tmp_path / "synthetic.jsonl", rows)
    report = ev.evaluate_dataset(
        data_path=data_path,
        tokenizer_path=infer_tiny_tokenizer,
        checkpoint_path=infer_tiny_checkpoint,
    )

    assert report["checkpoint"] is not None
    assert report["checkpoint"]["path"] == os.path.abspath(infer_tiny_checkpoint)
    assert len(report["checkpoint"]["sha256"]) == 64
    assert len(report["dataset"]["sha256"]) == 64
    assert len(report["tokenizer"]["sha256"]) == 64

    names = {s["name"] for s in report["splits"]}
    assert names == {"shipped_split", "response_template_disjoint", "topic_disjoint"}
    for split in report["splits"]:
        assert split["label"]
        assert math.isfinite(split["loss"])
        assert math.isfinite(split["ppl"])
        assert split["ppl"] >= 1.0
        assert split["leakage"]["val_active_tokens"] > 0
        for key in ("val_output_in_train_pct", "val_input_in_train_pct",
                    "val_distinct_output_templates",
                    "val_output_templates_in_train"):
            assert key in split["leakage"]

    assert isinstance(report["limitations"], list) and report["limitations"]
    assert "template" in report["limitations"][0].lower()


def test_max_samples_caps_val_rows(synthetic_dataset):
    data_path, tok_path, _ = synthetic_dataset
    report = ev.evaluate_dataset(
        data_path=data_path, tokenizer_path=tok_path, max_samples=2)
    for split in report["splits"]:
        assert split["leakage"]["val_rows"] == 2
        assert split["n_val"] == 2


def test_malformed_rows_skipped_and_counted(tmp_path):
    good = [
        {"input": "q one", "output": "a one", "category": "alpha"},
        {"input": "q two", "output": "a two", "category": "beta"},
    ]
    path = tmp_path / "mixed.jsonl"
    path.write_text(
        json.dumps(good[0]) + "\n"
        "not json at all\n"
        "[1, 2, 3]\n"
        '{"input": "q3", "output": "a3"}\n'
        '{"input": 42, "output": "a4", "category": "c"}\n'
        "\n"
        + json.dumps(good[1]) + "\n",
        encoding="utf-8",
    )

    rows, malformed = ev.load_jsonl_rows(str(path))
    assert len(rows) == 2
    assert malformed == 4
    assert rows[0][0] == "q one" and rows[1][0] == "q two"


def test_missing_data_file_error_names_file_and_fix(capsys):
    rc = ev.main(["--data", "definitely_absent.jsonl", "--tokenizer", "tok.json"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "definitely_absent.jsonl" in err
    assert "generate_data.py 500000" in err
    assert "coolm.prepare_data" in err


def test_missing_tokenizer_error_names_file(tmp_path, capsys):
    data_path = write_jsonl(tmp_path / "data.jsonl", make_synthetic_rows())
    rc = ev.main(["--data", data_path,
                  "--tokenizer", str(tmp_path / "absent_tokenizer.json")])
    assert rc == 2
    assert "absent_tokenizer.json" in capsys.readouterr().err


def test_report_written_to_out(synthetic_dataset, tmp_path):
    data_path, tok_path, _ = synthetic_dataset
    out_path = tmp_path / "reports" / "eval.json"
    rc = ev.main(["--data", data_path, "--tokenizer", tok_path,
                  "--out", str(out_path)])
    assert rc == 0
    report = json.loads(out_path.read_text(encoding="utf-8"))
    assert report["dataset"]["sha256"]
    assert report["tokenizer"]["sha256"]
    assert report["checkpoint"] is None  # no checkpoint given
    assert {s["name"] for s in report["splits"]} == {
        "shipped_split", "response_template_disjoint", "topic_disjoint"}
    for split in report["splits"]:
        assert split["loss"] is None  # loss skipped without a checkpoint
        assert split["label"]
        assert split["leakage"]
    assert report["limitations"]
