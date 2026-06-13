import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        assert config.d_model % config.n_heads == 0

        self.qkv = nn.Linear(config.d_model, 3 * config.n_heads * self.head_dim, bias=False)
        self.out = nn.Linear(config.n_heads * self.head_dim, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.register_buffer("bias", torch.triu(
            torch.full((config.max_seq_len, config.max_seq_len), float("-inf")), diagonal=1
        ))

    def forward(self, x, mask=None, use_cache=False, kv_cache=None):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if use_cache:
            if kv_cache is not None:
                k_prev, v_prev = kv_cache
                k = torch.cat([k_prev, k], dim=-2)
                v = torch.cat([v_prev, v], dim=-2)
            new_kv_cache = (k, v)
        else:
            new_kv_cache = None

        T_total = k.size(-2)
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)

        if mask is None:
            causal_bias = self.bias[T_total - T : T_total, :T_total]
            attn = attn + causal_bias
        else:
            attn = attn + mask

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = attn @ v
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.out(out), new_kv_cache


class FFN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.up = nn.Linear(config.d_model, config.ffn_hidden, bias=False)
        self.down = nn.Linear(config.ffn_hidden, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.down(F.gelu(self.up(x))))


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = Attention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.ffn = FFN(config)

    def forward(self, x, mask=None, use_cache=False, kv_cache=None):
        attn_out, new_kv_cache = self.attn(self.ln1(x), mask, use_cache=use_cache, kv_cache=kv_cache)
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x, new_kv_cache


class CooLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )
        self.ln_final = nn.LayerNorm(config.d_model)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.apply(self._init_weights)

        # Weight tying after init so both share the same initialized tensor
        self.lm_head.weight = self.token_embedding.weight

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            nn.init.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, targets=None, use_cache=False, kv_caches=None, start_pos=0):
        B, T = input_ids.shape
        assert T + start_pos <= self.config.max_seq_len

        tok_emb = self.token_embedding(input_ids)
        pos = torch.arange(start_pos, start_pos + T, device=input_ids.device)
        pos_emb = self.pos_embedding(pos)
        x = self.dropout(tok_emb + pos_emb)

        new_kv_caches = []
        for i, block in enumerate(self.blocks):
            block_cache = kv_caches[i] if kv_caches is not None else None
            x, block_new_cache = block(x, use_cache=use_cache, kv_cache=block_cache)
            if use_cache:
                new_kv_caches.append(block_new_cache)

        x = self.ln_final(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,
            )
        return logits, loss, new_kv_caches

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=64, temperature=1.0, top_k=40):
        was_training = self.training
        self.eval()

        if input_ids.size(1) > self.config.max_seq_len:
            input_ids = input_ids[:, -self.config.max_seq_len:]

        curr_len = input_ids.size(1)
        logits, _, kv_caches = self.forward(input_ids, use_cache=True, start_pos=0)

        logits = logits[:, -1, :] / temperature
        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, -1:]] = float("-inf")

        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_id], dim=1)

        if next_id.item() == self.config.eos_token_id:
            if was_training:
                self.train()
            return input_ids

        for _ in range(max_new_tokens - 1):
            if curr_len >= self.config.max_seq_len:
                # Truncate and rebuild cache from scratch
                input_ids = input_ids[:, -self.config.max_seq_len:]
                # Build cache from all but last token, then step the last token
                logits, _, kv_caches = self.forward(
                    input_ids[:, :-1], use_cache=True, start_pos=0
                )
                curr_len = self.config.max_seq_len - 1
                logits, _, kv_caches = self.forward(
                    input_ids[:, -1:], use_cache=True,
                    kv_caches=kv_caches, start_pos=curr_len
                )
            else:
                logits, _, kv_caches = self.forward(
                    next_id, use_cache=True,
                    kv_caches=kv_caches, start_pos=curr_len
                )

            curr_len += 1

            logits = logits[:, -1, :] / temperature
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, -1:]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_id], dim=1)

            if next_id.item() == self.config.eos_token_id:
                break

        if was_training:
            self.train()
        return input_ids
