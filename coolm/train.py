"""Train CooLM with loss masking."""

import os
import math
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from coolm.config import ModelConfig, TrainConfig
from coolm.model import CooLM


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


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"using device: {device}")

    model_cfg = ModelConfig()
    train_cfg = TrainConfig()

    model = CooLM(model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params:,}")
    assert n_params < 100_000_000, f"model too large: {n_params:,}"
    print(f"  vocab: {model_cfg.vocab_size}, dim: {model_cfg.d_model}")
    print(f"  layers: {model_cfg.n_layers}, heads: {model_cfg.n_heads}")
    print(f"  ffn: {model_cfg.ffn_hidden}, max_seq: {model_cfg.max_seq_len}")

    data_path = os.path.join("data", "train_data.pt")
    mask_path = os.path.join("data", "train_mask.pt")
    if not os.path.exists(data_path):
        print(f"error: run prepare_data.py first — {data_path} not found")
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
    )
    val_loader = DataLoader(
        CooDataset(val_tokens, val_masks), batch_size=train_cfg.batch_size, shuffle=False,
        num_workers=0,
    )

    optimizer = configure_optimizer(model, train_cfg)
    scaler = torch.amp.GradScaler(enabled=(device == "cuda"))

    os.makedirs("checkpoints", exist_ok=True)
    best_val_loss = float("inf")
    patience_counter = 0
    max_patience = 5
    step = 0
    iter_data = iter(train_loader)

    print(f"\ndata: {len(train_tokens)} train / {len(val_tokens)} val sequences")
    print(f"batch: {train_cfg.batch_size}, steps: {train_cfg.max_steps}")
    print(f"{'step':>6} | {'lr':>10} | {'loss':>8} | {'val':>8} | {'ppl':>8} | {'tok/s':>8} | {'g norm':>7} | {'eta':>8}")
    print("-" * 80)

    t_start = time.time()
    t_step_start = t_start
    recent_losses = []
    total_tokens = 0

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

        # Update LR before step so the correct schedule value is used
        lr = get_lr(step, train_cfg)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        with torch.amp.autocast(device_type=device, enabled=(device == "cuda")):
            _, loss, _ = model(x, targets=targets)

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.gradient_clip).item()
        scaler.step(optimizer)
        scaler.update()
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

        if step % train_cfg.eval_interval == 0:
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
                }, "checkpoints/best_model.pt")
                print(f"  -> saved best (val loss {avg_val:.4f})")
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"  -> patience: {patience_counter}/{max_patience}")
                if patience_counter >= max_patience:
                    print(f"Early stopping triggered at step {step}!")
                    break

        if step % train_cfg.save_interval == 0 and step > 0:
            torch.save({
                "step": step,
                "model_state": model.state_dict(),
                "model_config": model_cfg,
            }, f"checkpoints/model_step{step}.pt")
            print(f"  -> saved checkpoint step {step}")

        step += 1

    total_time = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  done. {train_cfg.max_steps} steps in {total_time/60:.1f}m ({total_time:.0f}s)")
    print(f"  tokens processed: {total_tokens:,}")
    print(f"  avg throughput: {total_tokens/total_time:.0f} tok/s")
    print(f"  best val loss: {best_val_loss:.4f} (ppl {math.exp(best_val_loss):.2f})")
    print(f"{'='*60}")


if __name__ == "__main__":
    train()
