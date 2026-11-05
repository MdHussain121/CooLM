"""The single authoritative checkpoint loader for CooLM.

Every consumer (coolm.inference, tools.verify_artifacts, tools.model_export)
loads checkpoints through this module so the security-relevant settings live
in exactly one place: torch.load always runs with weights_only=True against
an explicit allowlist of trusted types. There is deliberately no
unsafe-pickle fallback anywhere — a checkpoint that cannot be loaded safely
is an error, never a warning.
"""
import os
import warnings

import torch

from coolm.config import ModelConfig
from coolm.model import CooLM


class CheckpointError(Exception):
    """Base class for all checkpoint loading failures."""


class CheckpointNotFoundError(CheckpointError):
    """The checkpoint file does not exist at the given path."""


class CheckpointFormatError(CheckpointError):
    """The file exists but cannot be deserialized (bad magic, corrupted
    archive, unsupported format, or a pickle referencing non-allowlisted
    globals)."""


class CheckpointCompatibilityError(CheckpointError):
    """The checkpoint deserialized but does not fit the architecture
    (missing/wrong-dict keys, key or shape mismatch against CooLM, or
    non-finite weights)."""


# The checkpoint pickles a ModelConfig dataclass instance under
# "model_config". weights_only=True rejects every pickle global it does not
# recognize, so exactly this class is allow-listed via add_safe_globals —
# nothing else is trusted, and there is deliberately NO unsafe-pickle
# fallback (an internet-distributed checkpoint must never execute arbitrary
# pickle code).
SAFE_CLASSES = [ModelConfig]


def load_checkpoint(checkpoint_path, map_location="cpu"):
    """Safely load a CooLM checkpoint dict from ``checkpoint_path``.

    Returns the raw checkpoint dict with ``ckpt["model_config"]`` guaranteed
    to be a ModelConfig instance (defaults are substituted — with a warning —
    when the checkpoint omits it).

    Raises:
        CheckpointNotFoundError: the file does not exist.
        CheckpointFormatError: torch could not deserialize the file.
        CheckpointCompatibilityError: the dict does not look like a CooLM
            checkpoint (no ``model_state`` key, or a non-ModelConfig
            ``model_config``).
    """
    checkpoint_path = os.fspath(checkpoint_path)
    if not os.path.isfile(checkpoint_path):
        raise CheckpointNotFoundError(f"checkpoint not found: {checkpoint_path}")

    torch.serialization.add_safe_globals(SAFE_CLASSES)
    try:
        ckpt = torch.load(checkpoint_path, map_location=map_location, weights_only=True)
    except Exception as e:
        raise CheckpointFormatError(
            f"could not load checkpoint {checkpoint_path}: {type(e).__name__}: {e}"
        ) from e

    if not isinstance(ckpt, dict) or "model_state" not in ckpt:
        raise CheckpointCompatibilityError(
            f"checkpoint {checkpoint_path} has no 'model_state' key "
            f"(top-level object: {type(ckpt).__name__}"
            + (
                f", keys: {sorted(ckpt.keys())}"
                if isinstance(ckpt, dict)
                else ""
            )
            + ")"
        )

    if "model_config" in ckpt:
        cfg = ckpt["model_config"]
        if not isinstance(cfg, ModelConfig):
            raise CheckpointCompatibilityError(
                f"checkpoint {checkpoint_path}: model_config must be a "
                f"ModelConfig instance, got {type(cfg).__name__}"
            )
    else:
        warnings.warn(
            f"checkpoint {checkpoint_path} contains no model_config; "
            f"assuming ModelConfig() defaults (strict state-dict loading "
            f"will reject any architecture mismatch)",
            UserWarning,
            stacklevel=2,
        )
        ckpt["model_config"] = ModelConfig()
    return ckpt


def _validate_state_dict_finite(state_dict, checkpoint_path):
    """Reject state dicts containing NaN/Inf weights (corrupt artifacts).

    The registered causal-mask buffers (``*.attn.bias``) are exempt: by
    construction they hold -inf above the diagonal, which is correct
    attention masking, not corruption.
    """
    for name, tensor in state_dict.items():
        if not torch.is_tensor(tensor) or name.endswith(".attn.bias"):
            continue
        if not torch.isfinite(tensor).all():
            raise CheckpointCompatibilityError(
                f"checkpoint {checkpoint_path}: state-dict tensor '{name}' "
                f"contains NaN or Inf values"
            )


def load_trained_model(checkpoint_path, device="cpu"):
    """Load a checkpoint and build the trained CooLM it describes.

    Returns ``(model, cfg, ckpt)`` where ``model`` is on ``device`` in its
    constructed state (callers decide whether to ``eval()``) and ``ckpt`` is
    the raw checkpoint dict (step/val_loss metadata included).

    Raises the :class:`CheckpointError` subclasses from :func:`load_checkpoint`,
    plus CheckpointCompatibilityError when the state dict does not strictly
    match the architecture or contains non-finite weights.
    """
    ckpt = load_checkpoint(checkpoint_path)
    cfg = ckpt["model_config"]
    model = CooLM(cfg)
    try:
        model.load_state_dict(ckpt["model_state"], strict=True)
    except RuntimeError as e:
        # torch's strict-load message carries the exact missing/unexpected
        # keys and every size mismatch; surface it verbatim.
        raise CheckpointCompatibilityError(
            f"checkpoint {checkpoint_path} does not match the "
            f"{type(model).__name__} architecture: {e}"
        ) from e
    _validate_state_dict_finite(ckpt["model_state"], checkpoint_path)
    return model.to(device), cfg, ckpt
