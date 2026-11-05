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
        assert (
            T + start_pos <= self.config.max_seq_len
        ), (
            f"input length {T} + start_pos {start_pos} exceeds max_seq_len "
            f"{self.config.max_seq_len}"
        )
        if kv_caches is not None:
            # FIND-035: cover the cache state too — an overflowing call used
            # to slip past this assert and die later inside attention with an
            # obscure broadcast error. Positions are absolute: the cache
            # holds positions 0..cached_len-1, so the new tokens must anchor
            # exactly at the cache end (every generate() call site does).
            # Together these two checks reject any call where
            # cached_len + T + start_pos overflows or mis-anchors.
            cached_len = kv_caches[0][0].shape[-2]
            assert (
                cached_len + T <= self.config.max_seq_len
            ), (
                f"kv cache overflow: cached_len {cached_len} + new tokens {T} "
                f"exceeds max_seq_len {self.config.max_seq_len}"
            )
            assert (
                start_pos == cached_len
            ), (
                f"inconsistent start_pos {start_pos} with kv cache of length "
                f"{cached_len}; new tokens must anchor at the cache end "
                f"(start_pos == cached_len)"
            )

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
    def generate(self, input_ids, max_new_tokens=64, temperature=1.0, top_k=40, seed=None):
        """Autoregressive generation. Inference-only API; the audited
        forward()/Attention/FFN math is used unchanged.

        Contract:
        - Returns the FULL sequence ``prompt + generated tokens`` with shape
          ``(B, prompt_len + generated)``. The sliding window below bounds
          what the model *sees*, never what the caller gets back: the return
          value is never truncated (FIND-001).
        - ``max_new_tokens <= 0`` returns ``input_ids`` unchanged (out of
          place; nothing is appended or truncated) (FIND-029).
        - ``temperature == 0`` means greedy decoding (argmax — deterministic,
          the standard convention). ``temperature < 0`` or non-finite raises
          ``ValueError``; non-numeric arguments raise ``TypeError`` (bools
          are rejected everywhere) (FIND-009).
        - ``top_k <= 0`` disables top-k filtering; ``top_k > vocab_size`` is
          clamped silently.
        - Batched EOS (FIND-030): each row stops independently once it
          samples ``config.eos_token_id`` — the EOS itself is that row's last
          content token; afterwards the row appends ``config.pad_token_id``
          so it produces no further content. Generation stops early only
          when ALL rows are finished, so the output shape stays uniform.
        - Sliding window (FIND-051): once the KV window is full it is
          rebuilt from the most recent ``max_seq_len`` tokens with positions
          re-anchored to ``0 .. max_seq_len-1`` (audit-verified rebuild
          math, preserved exactly).
        - ``forward`` raises ``AssertionError`` if the KV window would
          overflow ``max_seq_len`` (FIND-035).
        - ``seed`` (FIND-034): when set (int), the global torch RNG is
          seeded before sampling, making the sampled tokens reproducible;
          ``None`` (default) keeps the previous unseeded behavior. Must be
          an int (bools rejected like the other arguments).
        """
        # Validate before touching any module state: invalid arguments must
        # not even toggle eval mode (pairs with the try/finally below).
        if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int):
            raise TypeError(
                f"max_new_tokens must be an int, got {type(max_new_tokens).__name__}"
            )
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
            raise TypeError(
                f"temperature must be a real number, got {type(temperature).__name__}"
            )
        if not math.isfinite(temperature) or temperature < 0:
            raise ValueError(
                f"temperature must be a finite non-negative number, got {temperature!r}; "
                "use temperature=0 for greedy decoding"
            )
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise TypeError(f"top_k must be an int, got {type(top_k).__name__}")
        if top_k < 0:
            raise ValueError(f"top_k must be >= 0, got {top_k}")
        if seed is not None:
            if isinstance(seed, bool) or not isinstance(seed, int):
                raise TypeError(f"seed must be an int or None, got {type(seed).__name__}")
            # FIND-034: reproducible sampling. Seeds the global torch RNG
            # (documented side effect); greedy decoding ignores it.
            torch.manual_seed(seed)
            if input_ids.is_cuda:
                torch.cuda.manual_seed_all(seed)

        if max_new_tokens <= 0:
            # FIND-029: a no-op request returns the prompt unchanged.
            # generate() never mutates input_ids, so this is out of place.
            return input_ids

        max_len = self.config.max_seq_len
        eos_id = self.config.eos_token_id
        pad_id = self.config.pad_token_id

        was_training = self.training
        try:
            self.eval()

            # FIND-001: `full` is the immutable accumulator that is returned;
            # `window` is the bounded view the model actually consumes. Both
            # only ever grow by concatenation — the window is sliced for the
            # model's benefit, the return value never is.
            full = input_ids
            window = input_ids[:, -max_len:]

            logits, _, kv_caches = self.forward(window, use_cache=True, start_pos=0)
            next_id = self._sample_token(logits[:, -1, :], temperature, top_k)

            # FIND-030: per-row EOS state — no .item(), so any batch works.
            finished = next_id.squeeze(-1) == eos_id  # (B,) bool
            full = torch.cat([full, next_id], dim=1)
            window = torch.cat([window, next_id], dim=1)
            if bool(finished.all()):
                return full

            # Invariant at the top of each step: the KV cache holds
            # window[:, :-1] and curr_len == window.size(1) - 1.
            curr_len = window.size(1) - 1
            for _ in range(max_new_tokens - 1):
                if curr_len >= max_len:
                    # Window rebuild — audit-verified math, kept exactly:
                    # re-prefill on window[:-1] at start_pos=0, then step
                    # window[-1:] at start_pos=max_len-1.
                    #
                    # FIND-051: the window tokens are positionally
                    # RE-ANCHORED to 0..max_len-1 — the model re-reads the
                    # most recent tokens as if they started the sequence.
                    # This bounds what the model sees; `full` above still
                    # carries every token for the return value.
                    window = window[:, -max_len:]
                    logits, _, kv_caches = self.forward(
                        window[:, :-1], use_cache=True, start_pos=0
                    )
                    curr_len = max_len - 1
                    logits, _, kv_caches = self.forward(
                        window[:, -1:], use_cache=True,
                        kv_caches=kv_caches, start_pos=curr_len
                    )
                else:
                    logits, _, kv_caches = self.forward(
                        window[:, -1:], use_cache=True,
                        kv_caches=kv_caches, start_pos=curr_len
                    )

                curr_len += 1

                next_id = self._sample_token(logits[:, -1, :], temperature, top_k)
                eos_hit = next_id.squeeze(-1) == eos_id
                append = next_id.clone()
                # Rows that were already finished append pad (no further
                # content); rows finishing on THIS step keep their sampled
                # EOS as their final content token.
                append[finished] = pad_id
                finished = finished | eos_hit
                full = torch.cat([full, append], dim=1)
                window = torch.cat([window, append], dim=1)
                if bool(finished.all()):
                    break

            return full
        finally:
            # FIND-038: never leak eval mode over the caller's state.
            if was_training:
                self.train()

    def _sample_token(self, logits, temperature, top_k):
        """Sample one next token per row from (B, vocab) logits."""
        if temperature == 0:
            # FIND-009: temperature=0 is greedy decoding by convention; no
            # probability distribution is built, so no NaN/Inf can occur.
            return logits.argmax(dim=-1, keepdim=True)
        # Denominators below ~1e-6 saturate fp32 logits to inf, which would
        # turn the softmax into NaN; floor them to keep the distribution
        # finite (a floored temperature is effectively greedy anyway).
        logits = logits / max(temperature, 1e-6)
        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, -1:]] = float("-inf")
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)
