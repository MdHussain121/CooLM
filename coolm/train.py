"""Train CooLM with loss masking."""

import dataclasses
import os
import math
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from coolm.config import ModelConfig, TrainConfig
from coolm.model import CooLM

# FIND-040: default data/checkpoint locations resolve against the repo root,
# not the CWD, so training works from any directory. Callers may override
# both via train()'s keyword arguments (explicit paths always win).
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_DATA_DIR = os.path.join(REPO_ROOT, "data")
DEFAULT_CHECKPOINT_DIR = os.path.join(REPO_ROOT, "checkpoints")


def get_lr(it, config):
    if it < config.warmup_steps:
        return config.learning_rate * it / config.warmup_steps
    ratio = (it - config.warmup_steps) / (config.max_steps - config.warmup_steps)
    return config.min_lr + 0.5 * (config.learning_rate - config.min_lr) * (1 + math.cos(math.pi * ratio))


def configure_optimizer(model, config):
    decay = set()
    no_decay = set()
    for name, param in model.named_parameters():
        if param.ndim < 2 or "bias" in name or "norm" in name or "embedding" in name:
            no_decay.add(name)
        else:
            decay.add(name)

    param_dict = {pn: p for pn, p in model.named_parameters()}
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(decay)], "weight_decay": config.weight_decay},
        {"params": [param_dict[pn] for pn in sorted(no_decay)], "weight_decay": 0.0},
    ]

    return torch.optim.AdamW(optim_groups, lr=config.learning_rate, betas=(0.9, 0.95))


class CooDataset(Dataset):
    def __init__(self, tokens, masks):
        self.tokens = tokens
        self.masks = masks

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        return self.tokens[idx], self.masks[idx]


def train(resume=None, seed=None, data_dir=None, checkpoint_dir=None):
    """Train CooLM.

    Args:
        resume: path to a ``last_model.pt`` resume state. Relative paths are
            resolved against ``checkpoint_dir``. ``None`` (default) starts a
            fresh run — unchanged behavior (FIND-013).
        seed: when set, ``torch.manual_seed`` (+ CUDA seed) is applied before
            model init and the DataLoader shuffle uses a seeded generator —
            a documented reproducibility aid (FIND-034). ``None`` keeps the
            previous unseeded behavior.
        data_dir: directory holding train_data.pt / train_mask.pt (default:
            ``<repo root>/data``, FIND-040).
        checkpoint_dir: directory for best_model.pt / periodic checkpoints /
            last_model.pt (default: ``<repo root>/checkpoints``, FIND-040).

    The training-loop math (schedule, masking, optimizer order) is identical
    for a fresh unseeded run.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"using device: {device}")

    # FIND-034: seed before model init so the init itself is reproducible.
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        print(f"seed: {seed} (model init + dataloader shuffle)")

    model_cfg = ModelConfig()
    train_cfg = TrainConfig()

    model = CooLM(model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params:,}")
    assert n_params < 100_000_000, f"model too large: {n_params:,}"
    print(f"  vocab: {model_cfg.vocab_size}, dim: {model_cfg.d_model}")
    print(f"  layers: {model_cfg.n_layers}, heads: {model_cfg.n_heads}")
    print(f"  ffn: {model_cfg.ffn_hidden}, max_seq: {model_cfg.max_seq_len}")

    data_dir = data_dir if data_dir is not None else DEFAULT_DATA_DIR
    checkpoint_dir = checkpoint_dir if checkpoint_dir is not None else DEFAULT_CHECKPOINT_DIR
    data_path = os.path.join(data_dir, "train_data.pt")
    mask_path = os.path.join(data_dir, "train_mask.pt")
    if not os.path.exists(data_path):
        print(f"error: run prepare_data.py first — {data_path} not found")
        return
    if not os.path.exists(mask_path):
        print(f"error: {mask_path} not found (run prepare_data.py to generate it)")
        return

    tokens = torch.load(data_path, map_location="cpu", weights_only=True)
    masks = torch.load(mask_path, map_location="cpu", weights_only=True)

    # Shuffle dataset defensively before splitting
    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(len(tokens), generator=generator)
    tokens = tokens[indices]
    masks = masks[indices]

    n = int(0.95 * len(tokens))
    train_tokens, val_tokens = tokens[:n], tokens[n:]
    train_masks, val_masks = masks[:n], masks[n:]

    train_loader = DataLoader(
        CooDataset(train_tokens, train_masks), batch_size=train_cfg.batch_size, shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(seed) if seed is not None else None,
    )
    val_loader = DataLoader(
        CooDataset(val_tokens, val_masks), batch_size=train_cfg.batch_size, shuffle=False,
        num_workers=0,
    )

    optimizer = configure_optimizer(model, train_cfg)
    scaler = torch.amp.GradScaler(enabled=(device == "cuda"))

    os.makedirs(checkpoint_dir, exist_ok=True)
    last_model_path = os.path.join(checkpoint_dir, "last_model.pt")
    best_model_path = os.path.join(checkpoint_dir, "best_model.pt")
    best_val_loss = float("inf")
    patience_counter = 0
    max_patience = 5
    step = 0
    total_tokens = 0
    scaler_skips = 0
    zero_active_batches = 0
    stopped_early = False
    iter_data = iter(train_loader)

    # FIND-013: restore the full resume state (model, optimizer, scaler,
    # schedule position, early-stop counters, token throughput). The stored
    # "step" is the number of COMPLETED updates, so training continues at the
    # next unperformed iteration and get_lr(step) picks the schedule up
    # automatically. (best_model.pt keeps its published-artifact format.)
    if resume is not None:
        resume_path = resume if os.path.isabs(resume) else os.path.join(checkpoint_dir, resume)
        if not os.path.exists(resume_path):
            print(f"error: resume checkpoint not found: {resume_path}")
            return
        print(f"resuming from {resume_path}")
        state = torch.load(resume_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        scaler.load_state_dict(state["scaler_state"])
        best_val_loss = state.get("best_val_loss", float("inf"))
        patience_counter = state.get("patience_counter", 0)
        total_tokens = state.get("total_tokens", 0)
        step = state.get("step", 0)
        print(f"  restored step {step} ({step:,} updates completed, {total_tokens:,} tokens seen)")

    print(f"\ndata: {len(train_tokens)} train / {len(val_tokens)} val sequences")
    print(f"batch: {train_cfg.batch_size}, steps: {train_cfg.max_steps}")
    print(f"{'step':>6} | {'lr':>10} | {'loss':>8} | {'val':>8} | {'ppl':>8} | {'tok/s':>8} | {'g norm':>7} | {'eta':>8}")
    print("-" * 80)

    t_start = time.time()
    t_step_start = t_start
    recent_losses = []

    while step < train_cfg.max_steps:
        try:
            batch = next(iter_data)
        except StopIteration:
            iter_data = iter(train_loader)
            batch = next(iter_data)

        padded, mask_batch = batch
        padded = padded.to(device)
        mask_batch = mask_batch.to(device)

        targets = padded[:, 1:].contiguous()
        x = padded[:, :-1].contiguous()
        target_mask = mask_batch[:, 1:].contiguous()

        # ignore input tokens and padding (mask=0) in loss
        targets[target_mask == 0] = -100

        n_active = (targets != -100).sum().item()

        # FIND-054: a batch with zero active (loss) tokens would make
        # cross_entropy return NaN and silently trigger a scaler skip.
        # Skip the update with a counted warning instead.
        if n_active == 0:
            zero_active_batches += 1
            print(f"  -> warning: batch at step {step} has zero active loss tokens; skipping update (count: {zero_active_batches})")
            step += 1
            continue

        # Update LR before step so the correct schedule value is used
        lr = get_lr(step, train_cfg)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        with torch.amp.autocast(device_type=device, enabled=(device == "cuda")):
            _, loss, _ = model(x, targets=targets)

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.gradient_clip).item()

        # FIND-033: detect GradScaler skips — a skipped update backs the
        # scale off (scale_after < scale_before); a growth step raises it.
        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() < scale_before:
            scaler_skips += 1
            print(f"  -> warning: GradScaler skipped the update at step {step} (non-finite gradients); skips: {scaler_skips}")
        optimizer.zero_grad()

        recent_losses.append(loss.item())
        total_tokens += n_active

        if step % train_cfg.log_interval == 0:
            elapsed = time.time() - t_start
            step_time = time.time() - t_step_start
            tok_s = total_tokens / elapsed if elapsed > 0 else 0
            avg_loss = sum(recent_losses[-train_cfg.log_interval:]) / min(len(recent_losses), train_cfg.log_interval)
            steps_left = train_cfg.max_steps - step - 1
            eta = step_time / max(train_cfg.log_interval, 1) * steps_left if steps_left > 0 else 0
            eta_str = f"{eta/60:.0f}m" if eta < 3600 else f"{eta/3600:.1f}h"
            print(f"{step:6d} | {lr:10.2e} | {avg_loss:8.4f} | {'--':>8} | {'--':>8} | {tok_s:8.0f} | {grad_norm:7.2f} | {eta_str:>8}")
            t_step_start = time.time()

        # FIND-032: never evaluate/save at step 0 — the model is untrained
        # then, and a crash before the first real eval must not leave an
        # untrained best_model.pt behind.
        if step > 0 and step % train_cfg.eval_interval == 0:
            model.eval()
            total_val_loss = 0.0
            total_val_tokens = 0
            with torch.no_grad():
                for vb, vm in val_loader:
                    vx = vb.to(device)
                    vm = vm.to(device)
                    vt = vx[:, 1:].contiguous()
                    vx = vx[:, :-1].contiguous()
                    vm = vm[:, 1:].contiguous()
                    vt[vm == 0] = -100
                    n_active_val = (vt != -100).sum().item()
                    with torch.amp.autocast(device_type=device, enabled=(device == "cuda")):
                        _, vl, _ = model(vx, targets=vt)
                    if n_active_val > 0:
                        total_val_loss += vl.item() * n_active_val
                        total_val_tokens += n_active_val
            avg_val = total_val_loss / max(total_val_tokens, 1)
            ppl = math.exp(avg_val)
            elapsed = time.time() - t_start
            tok_s = total_tokens / elapsed if elapsed > 0 else 0
            steps_left = train_cfg.max_steps - step - 1
            avg_step_s = elapsed / max(step + 1, 1)
            eta = avg_step_s * steps_left
            eta_str = f"{eta/60:.0f}m" if eta < 3600 else f"{eta/3600:.1f}h"
            print(f"{step:6d} | {lr:10.2e} | {'--':>8} | {avg_val:8.4f} | {ppl:8.2f} | {tok_s:8.0f} | {grad_norm:7.2f} | {eta_str:>8}")
            model.train()

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                torch.save({
                    "step": step,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_loss": avg_val,
                    "model_config": model_cfg,
                }, best_model_path)
                print(f"  -> saved best (val loss {avg_val:.4f})")
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"  -> patience: {patience_counter}/{max_patience}")
                if patience_counter >= max_patience:
                    # FIND-031: report the completed-update count, not the
                    # 0-based iteration index.
                    stopped_early = True
                    print(f"Early stopping triggered at step {step} ({step + 1} updates completed)!")
                    break

        if step % train_cfg.save_interval == 0 and step > 0:
            torch.save({
                "step": step,
                "model_state": model.state_dict(),
                "model_config": model_cfg,
            }, os.path.join(checkpoint_dir, f"model_step{step}.pt"))
            print(f"  -> saved checkpoint step {step}")

            # FIND-013: full resume state alongside the periodic checkpoint.
            # "step" is the completed-update count (step + 1), so a resume
            # continues at the next unperformed iteration.
            torch.save({
                "step": step + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scaler_state": scaler.state_dict(),
                "best_val_loss": best_val_loss,
                "patience_counter": patience_counter,
                "total_tokens": total_tokens,
                "config": {
                    "model": dataclasses.asdict(model_cfg),
                    "train": dataclasses.asdict(train_cfg),
                },
            }, last_model_path)
            print(f"  -> saved resume state after {step + 1} updates (last_model.pt)")

        step += 1

    total_time = time.time() - t_start
    completed = (step + 1) if stopped_early else step
    print(f"\n{'='*60}")
    print(f"  done. {completed} steps in {total_time/60:.1f}m ({total_time:.0f}s)")
    print(f"  gradscaler skips: {scaler_skips}, zero-active-token batches skipped: {zero_active_batches}")
    print(f"  tokens processed: {total_tokens:,}")
    print(f"  avg throughput: {total_tokens/total_time:.0f} tok/s")
    print(f"  best val loss: {best_val_loss:.4f} (ppl {math.exp(best_val_loss):.2f})")
    print(f"{'='*60}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train CooLM with loss masking.")
    parser.add_argument("--resume", nargs="?",
                        const=os.path.join("checkpoints", "last_model.pt"),
                        default=None,
                        help="resume from a last_model.pt state; a bare --resume uses "
                             "checkpoints/last_model.pt (relative paths resolve against "
                             "the repo checkpoints directory)")
    parser.add_argument("--seed", type=int, default=None,
                        help="seed torch before model init and seed the DataLoader "
                             "shuffle (documented reproducibility aid, FIND-034)")
    args = parser.parse_args()
    train(resume=args.resume, seed=args.seed)
