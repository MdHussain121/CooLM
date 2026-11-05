"""Shared fixtures for the CooLM test suite.

Everything here uses tiny models (vocab ~128, 2 layers, max_seq_len 32)
built in fixtures — the 498MB real checkpoint is never touched by tests.
"""
import copy
import itertools
import os
import sys

import pytest
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from coolm.config import ModelConfig
from coolm.model import CooLM

# Tiny config shared by the generation / inference tests. vocab_size must
# match the tokenizer built by write_inference_tokenizer below.
GEN_TINY_CFG = ModelConfig(
    vocab_size=128,
    d_model=32,
    n_layers=2,
    n_heads=4,
    ffn_hidden=64,
    dropout=0.0,
    max_seq_len=32,
)

SPECIAL_TOKENS = ["<|pad|>", "<|bos|>", "<|eos|>", "<|unk|>", "<|im_start|>", "<|im_end|>"]

# Pure-alpha vocab words: normalize_user_text strips digits, so test user
# text must survive it unchanged.
_WORD_POOL = ["".join(p) for p in itertools.product("abcdefghij", repeat=3)]


def write_inference_tokenizer(path, vocab_size=GEN_TINY_CFG.vocab_size):
    """Write a tiny WordLevel tokenizer.json suitable for CooInference.

    Unlike the artifact-consistency fixture tokenizer, this one:
    - registers the chat special tokens as special added tokens, so
      ``decode(..., skip_special_tokens=True)`` actually skips them;
    - sets a whitespace pre-tokenizer, so multi-word text encodes to one
      token per word (the budget math depends on that);
    - uses ``<|unk|>`` as the unk token.
    """
    from tokenizers import Tokenizer, models, pre_tokenizers

    vocab = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
    words = _WORD_POOL[: vocab_size - len(SPECIAL_TOKENS)]
    vocab.update({word: i for i, word in enumerate(words, start=len(SPECIAL_TOKENS))})
    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<|unk|>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.add_special_tokens(SPECIAL_TOKENS)
    tokenizer.save(str(path))
    return str(path)


@pytest.fixture(scope="session")
def tiny_model_factory():
    """Factory for fresh tiny CooLM models (CPU, eval mode, seeded).

    Each call deep-copies the config, so tests may freely mutate
    ``model.config`` (e.g. eos_token_id) without leaking state.
    """

    def _make(seed=0, **config_overrides):
        cfg = copy.deepcopy(GEN_TINY_CFG)
        for key, value in config_overrides.items():
            setattr(cfg, key, value)
        torch.manual_seed(seed)
        model = CooLM(cfg)
        model.eval()
        return model

    return _make


def _tiny_state_dict():
    torch.manual_seed(0)
    return CooLM(GEN_TINY_CFG).state_dict()


@pytest.fixture(scope="session")
def infer_tiny_checkpoint(tmp_path_factory):
    """Tiny checkpoint loadable by CooInference via the safe loader."""
    path = tmp_path_factory.mktemp("infer_ckpt") / "tiny.pt"
    torch.save(
        {
            "step": 3,
            "model_state": _tiny_state_dict(),
            "model_config": copy.deepcopy(GEN_TINY_CFG),
            "val_loss": 2.5,
        },
        path,
    )
    return str(path)


@pytest.fixture(scope="session")
def infer_tiny_tokenizer(tmp_path_factory):
    path = tmp_path_factory.mktemp("infer_tok") / "tokenizer.json"
    return write_inference_tokenizer(path)


@pytest.fixture(scope="session")
def tiny_torch_engine(infer_tiny_checkpoint, infer_tiny_tokenizer):
    """CooInference over the tiny checkpoint, forced to CPU."""
    from coolm.inference import CooInference

    engine = CooInference(model_path=infer_tiny_checkpoint, tokenizer_path=infer_tiny_tokenizer)
    # Force CPU for determinism regardless of host CUDA availability.
    engine.model = engine.model.to("cpu").eval()
    engine.device = "cpu"
    return engine


@pytest.fixture(scope="session")
def onnx_tiny_export(infer_tiny_checkpoint, tmp_path_factory):
    """Tiny ONNX export (Assignment-1 export code) with its sidecar."""
    from tools import model_export

    out_path = tmp_path_factory.mktemp("onnx_export") / "tiny.onnx"
    rc = model_export.export_onnx(checkpoint_path=infer_tiny_checkpoint, output_path=str(out_path))
    assert rc == 0
    return {"onnx": str(out_path), "sidecar": str(out_path) + ".provenance.json"}


@pytest.fixture()
def onnx_engine(onnx_tiny_export, infer_tiny_tokenizer):
    """Function-scoped CooONNXInference (tests may swap engine.session)."""
    from tools.onnx_chat import CooONNXInference

    return CooONNXInference(
        onnx_path=onnx_tiny_export["onnx"], tokenizer_path=infer_tiny_tokenizer
    )
