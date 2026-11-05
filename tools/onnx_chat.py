import json
import os
import sys

import numpy as np
from tokenizers import Tokenizer

# Allow running as a script (python tools/onnx_chat.py) as well as a module:
# the shared inference helpers live in coolm/inference.py.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coolm.inference import (
    normalize_user_text,
    strip_special_token_literals,
    truncate_user_text,
)

def top_k_logits(logits, k):
    if k <= 0:
        return logits
    k = min(k, logits.shape[-1])  # top_k above the vocab filters nothing
    values = np.sort(logits)[-k]
    logits[logits < values] = -float('Inf')
    return logits


# FIND-040: default artifact paths resolve against the repo root (the parent
# of the tools/ directory), not the CWD. Explicit paths always win.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _repo_default(*parts):
    return os.path.join(_REPO_ROOT, *parts)


class CooONNXInference:
    # Context cap used when the provenance sidecar is absent or unreadable
    # (matches the trained ModelConfig.max_seq_len).
    FALLBACK_MAX_SEQ_LEN = 64

    def __init__(self, onnx_path=None, tokenizer_path=None):
        import onnxruntime as ort
        onnx_path = onnx_path or _repo_default("docs", "model.onnx")
        tokenizer_path = tokenizer_path or _repo_default("data", "tokenizer.json")

        # FIND-040: name the missing file instead of tokenizers' path-less
        # "The system cannot find the path specified" error.
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"tokenizer not found: {tokenizer_path}")
        if not os.path.exists(onnx_path):
            print("ONNX model not found. Generating it now...")
            from tools.model_export import export_onnx
            export_onnx(output_path=onnx_path)

        # FIND-006: the supported context comes from the export provenance
        # sidecar (tools/model_export.py), not from a hardcoded guess.
        self.max_seq_len = self._read_max_seq_len(onnx_path)

        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.bos_id = self.tokenizer.token_to_id("<|bos|>")
        self.eos_id = self.tokenizer.token_to_id("<|eos|>")

        print("Loading ONNX model...")
        self.session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        print("ONNX model loaded successfully.")

    def _read_max_seq_len(self, onnx_path):
        """Read max_seq_len from the provenance sidecar next to the ONNX file."""
        sidecar_path = onnx_path + ".provenance.json"
        fallback = self.FALLBACK_MAX_SEQ_LEN
        try:
            with open(sidecar_path, encoding="utf-8") as f:
                provenance = json.load(f)
            value = provenance.get("model_config", {}).get("max_seq_len")
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                return value
            print(
                f"Note: {sidecar_path} has no usable model_config.max_seq_len; "
                f"falling back to max_seq_len={fallback}."
            )
        except (OSError, ValueError) as exc:
            print(
                f"Note: provenance sidecar {sidecar_path} absent or unreadable "
                f"({type(exc).__name__}); falling back to max_seq_len={fallback}."
            )
        return fallback

    def format_prompt(self, user_text):
        return f"<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n"

    def _validate_sampling_args(self, max_new_tokens, temperature, top_k):
        # Mirrors CooLM.generate() validation.
        if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int):
            raise TypeError(
                f"max_new_tokens must be an int, got {type(max_new_tokens).__name__}"
            )
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
            raise TypeError(
                f"temperature must be a real number, got {type(temperature).__name__}"
            )
        if not np.isfinite(temperature) or temperature < 0:
            raise ValueError(
                f"temperature must be a finite non-negative number, got {temperature!r}; "
                "use temperature=0 for greedy decoding"
            )
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise TypeError(f"top_k must be an int, got {type(top_k).__name__}")
        if top_k < 0:
            raise ValueError(f"top_k must be >= 0, got {top_k}")

    def chat(self, user_text, max_new_tokens=64, temperature=0.4, top_k=30, seed=None):
        """Chat via the exported ONNX graph; the response is exactly the
        generated text (same policy as coolm.inference.CooInference.chat).

        Contract:
        - ``user_text`` must be a ``str`` (TypeError otherwise).
        - User text is normalized and token-wise truncated to the context
          budget with the scaffold preserved; a ``UserWarning`` is emitted on
          truncation (FIND-011/FIND-006).
        - Every session feed is capped to the last ``self.max_seq_len``
          tokens (sliding window with re-anchored positions, mirroring the
          PyTorch generate() semantics), so no input length can crash ONNX
          Runtime (FIND-006).
        - ``temperature == 0`` -> greedy (deterministic ``np.argmax``);
          ``temperature < 0`` or non-finite -> ValueError; ``top_k <= 0`` ->
          no filtering; ``top_k > vocab`` -> clamped.
        - ``max_new_tokens <= 0`` generates nothing -> the response is the
          empty string (documented contract, no crash).
        - EOS stops generation (the EOS token is consumed, not returned).
        - ``seed`` (FIND-034): when set (int), the global numpy RNG is
          seeded before sampling so generation is reproducible; ``None``
          (default) keeps the previous unseeded behavior. (The ONNX path
          samples with ``np.random.choice``.)
        """
        if not isinstance(user_text, str):
            raise TypeError(f"user_text must be a str, got {type(user_text).__name__}")
        if seed is not None:
            if isinstance(seed, bool) or not isinstance(seed, int):
                raise TypeError(f"seed must be an int or None, got {type(seed).__name__}")
            # FIND-034: reproducible sampling on the numpy RNG.
            np.random.seed(seed)
        self._validate_sampling_args(max_new_tokens, temperature, top_k)

        if max_new_tokens <= 0:
            # Documented contract: no generated tokens -> empty response.
            return ""

        text = normalize_user_text(user_text)
        # FIND-006/FIND-011: keep scaffold + user text inside the context.
        text = truncate_user_text(self.tokenizer, text, self.max_seq_len)

        prompt = self.format_prompt(text)
        encoded = self.tokenizer.encode(prompt)
        input_ids = [self.bos_id] + encoded.ids
        prompt_len = len(input_ids)

        for _ in range(max_new_tokens):
            # Sliding window: feed only the most recent max_seq_len tokens.
            # The exported graph has no cache, so window tokens get
            # re-anchored positions 0..max_seq_len-1 on every step — the
            # same semantics as the PyTorch generate() rebuild — and the
            # fed length never exceeds the supported context.
            window = input_ids[-self.max_seq_len:]
            inputs = {self.session.get_inputs()[0].name: np.array([window], dtype=np.int64)}
            logits = self.session.run(None, inputs)[0]

            # get last token logits
            next_token_logits = np.asarray(logits[0, -1, :], dtype=np.float64)
            if temperature == 0:
                # Greedy decoding: deterministic, no sampling.
                next_id = int(np.argmax(next_token_logits))
            else:
                # Floor the divisor like CooLM.generate(): sub-1e-6 fp32
                # temperatures would saturate the logits to inf and NaN out
                # the softmax.
                next_token_logits = next_token_logits / max(temperature, 1e-6)
                next_token_logits = top_k_logits(next_token_logits, top_k)

                # softmax
                exp_logits = np.exp(next_token_logits - np.max(next_token_logits))
                probs = exp_logits / np.sum(exp_logits)

                # sample
                next_id = int(np.random.choice(len(probs), p=probs))
            input_ids.append(next_id)

            if next_id == self.eos_id:
                break

        # Response = exactly the generated tokens, extracted by token index
        # (never a character offset into the decoded full text).
        generated_ids = input_ids[prompt_len:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return strip_special_token_literals(response)

def main():
    try:
        import onnxruntime
    except ImportError:
        print("onnxruntime is not installed. Run 'pip install onnxruntime' or use chat_onnx.bat")
        return

    engine = CooONNXInference()

    print("\ncoo — a rooftop pigeon (ONNX Local). type 'exit' to quit.\n")
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
