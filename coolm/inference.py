import os
import json
import torch
from tokenizers import Tokenizer

from coolm.config import ModelConfig
from coolm.model import CooLM


class CooInference:
    def __init__(self, model_path="checkpoints/best_model.pt", tokenizer_path="data/tokenizer.json"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.bos_id = self.tokenizer.token_to_id("<|bos|>")
        self.eos_id = self.tokenizer.token_to_id("<|eos|>")

        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        cfg = ckpt.get("model_config", ModelConfig())
        self.model = CooLM(cfg).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        print(f"loaded model ({sum(p.numel() for p in self.model.parameters()):,} params)")

    def format_prompt(self, user_text):
        # Return text WITHOUT <|bos|> — it will be prepended as a token ID
        return f"<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n"

    def chat(self, user_text, max_new_tokens=64, temperature=0.4, top_k=30):
        import re
        user_text = user_text.lower()
        user_text = re.sub(r'[^a-z\s]', '', user_text)
        user_text = re.sub(r'\bcoo\b', '', user_text).strip()
        
        prompt = self.format_prompt(user_text)
        encoded = self.tokenizer.encode(prompt)
        # Prepend bos_id as a token ID, matching training data preparation
        input_ids = torch.tensor([[self.bos_id] + encoded.ids], dtype=torch.long, device=self.device)

        output_ids = self.model.generate(input_ids, max_new_tokens, temperature, top_k)

        full_text = self.tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=False)

        # extract assistant response (everything after last <|im_start|>assistant)
        marker = "<|im_start|>assistant\n"
        idx = full_text.rfind(marker)
        if idx != -1:
            response = full_text[idx + len(marker):]
        else:
            response = full_text[len(prompt):]

        # clean up
        response = response.replace("<|im_end|>", "").replace("<|im_start|>", "").replace("<|eos|>", "").replace("<|bos|>", "").replace("<|pad|>", "").strip()
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
