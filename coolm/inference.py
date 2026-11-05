import os
import re
import warnings

import torch
from tokenizers import Tokenizer

from coolm.checkpoint import load_trained_model

# FIND-040: default artifact paths resolve against the repo root (the parent
# of this module's directory), not the CWD, so entry points work from any
# directory. Explicitly passed paths always win.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _repo_default(*parts):
    return os.path.join(_REPO_ROOT, *parts)


# The chat scaffold with an empty user turn (identical to
# CooInference.format_prompt / the training format). Encoding it gives the
# fixed token cost every user text pays on top of its own tokens.
_EMPTY_USER_SCAFFOLD = "<|im_start|>user\n<|im_end|>\n<|im_start|>assistant\n"

# Special-token literals stripped from decoded responses as
# defense-in-depth. FIND-061: <|unk|> added to the list.
_SPECIAL_TOKEN_LITERALS = (
    "<|im_end|>",
    "<|im_start|>",
    "<|eos|>",
    "<|bos|>",
    "<|pad|>",
    "<|unk|>",
)


def normalize_user_text(text):
    """Normalize free-form user input toward the training-text distribution.

    Lowercase; keep ASCII letters, whitespace and hyphens (training inputs
    contain e.g. 'high-rise'); drop every other character; collapse
    whitespace runs to single spaces; strip.

    FIND-010: the word 'coo' is deliberately NOT removed — training uses it
    pervasively — and hyphens survive. Pure-symbol input normalizes to the
    empty string (documented behavior).
    """
    text = text.lower()
    text = re.sub(r"[^a-z\s-]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def budget_user_tokens(tokenizer, max_seq_len):
    """Return how many user-text tokens fit inside a max_seq_len prompt.

    Accounts for the <|bos|> id that is prepended as a token (1) plus the
    scaffold encoded with an empty user turn.
    """
    scaffold_len = len(tokenizer.encode(_EMPTY_USER_SCAFFOLD).ids)
    return max_seq_len - 1 - scaffold_len


def truncate_user_text(tokenizer, user_text, max_seq_len):
    """Token-wise tail truncation of normalized user text (FIND-011).

    Keeps the most recent tokens (the tail usually carries the question),
    preserves the chat scaffold by construction, and surfaces a UserWarning
    instead of silently degrading the reply. Returns the input string
    unchanged when it already fits the budget.
    """
    budget = budget_user_tokens(tokenizer, max_seq_len)
    ids = tokenizer.encode(user_text).ids
    if len(ids) <= budget:
        return user_text
    warnings.warn(
        f"user text encodes to {len(ids)} tokens but only {budget} fit in the "
        f"{max_seq_len}-token model context; truncating to the most recent "
        f"{budget} tokens",
        UserWarning,
        stacklevel=2,
    )
    if budget <= 0:
        return ""
    # Decode the kept tail back to text. Special tokens (e.g. <|unk|>) are
    # dropped, which can only shrink the re-encoded prompt further below the
    # budget — the reconstructed prompt never exceeds max_seq_len.
    return tokenizer.decode(ids[-budget:], skip_special_tokens=True).strip()


def strip_special_token_literals(text):
    """Defense-in-depth cleanup of special-token literals in decoded text."""
    for literal in _SPECIAL_TOKEN_LITERALS:
        text = text.replace(literal, "")
    return text.strip()


class CooInference:
    def __init__(self, model_path=None, tokenizer_path=None):
        model_path = model_path or _repo_default("checkpoints", "best_model.pt")
        tokenizer_path = tokenizer_path or _repo_default("data", "tokenizer.json")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # FIND-040: name the missing file instead of tokenizers' path-less
        # "The system cannot find the path specified" error.
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"tokenizer not found: {tokenizer_path}")
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.bos_id = self.tokenizer.token_to_id("<|bos|>")
        self.eos_id = self.tokenizer.token_to_id("<|eos|>")

        self.model, _, _ = load_trained_model(model_path, device=self.device)
        self.model.eval()
        print(f"loaded model ({sum(p.numel() for p in self.model.parameters()):,} params)")

    def format_prompt(self, user_text):
        # Return text WITHOUT <|bos|> — it will be prepended as a token ID
        return f"<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n"

    def chat(self, user_text, max_new_tokens=64, temperature=0.4, top_k=30, seed=None):
        """Chat with the model; the response is exactly the generated text.

        Contract:
        - ``user_text`` must be a ``str`` (TypeError otherwise — no silent
          conversion).
        - User text is normalized (:func:`normalize_user_text`), then
          token-wise truncated to the model's context budget with the chat
          scaffold preserved; a ``UserWarning`` is emitted when truncation
          kicks in (FIND-011). It is never head-truncated by the model.
        - The response is the decode of exactly the generated tokens
          (``output_ids[0, prompt_len:]`` with ``skip_special_tokens=True``),
          never a character offset into the decoded full text (FIND-002,
          FIND-039). Special-token literals are additionally stripped
          (FIND-061).
        - Sampling arguments are validated by ``CooLM.generate``
          (``temperature=0`` -> greedy, ``max_new_tokens <= 0`` -> no
          generated tokens -> empty response, etc.).
        - ``seed`` (FIND-034): when set (int), torch is seeded before
          sampling so generation is reproducible; ``None`` (default) keeps
          the previous unseeded behavior. Seeds the global torch RNG.
        """
        if not isinstance(user_text, str):
            raise TypeError(f"user_text must be a str, got {type(user_text).__name__}")

        text = normalize_user_text(user_text)
        text = truncate_user_text(self.tokenizer, text, self.model.config.max_seq_len)

        prompt = self.format_prompt(text)
        encoded = self.tokenizer.encode(prompt)
        # Prepend bos_id as a token ID, matching training data preparation
        input_ids = torch.tensor([[self.bos_id] + encoded.ids], dtype=torch.long, device=self.device)

        output_ids = self.model.generate(input_ids, max_new_tokens, temperature, top_k, seed=seed)

        # FIND-002: extract by TOKEN INDEX — the response is exactly the
        # generated tokens, never a character offset into a decoded string.
        prompt_len = input_ids.size(1)
        generated_ids = output_ids[0, prompt_len:].tolist()
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        response = strip_special_token_literals(response)
        return response


def main():
    import sys
    engine = CooInference()

    print("\ncoo — a rooftop pigeon. type 'exit' to quit.\n")
    while True:
        user = input("you> ").strip()
        if user.lower() in ("exit", "quit"):
            break
        if not user:
            continue
        resp = engine.chat(user)
        print(f"coo> {resp}\n")


if __name__ == "__main__":
    main()
