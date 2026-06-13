import os
import re
import numpy as np
from tokenizers import Tokenizer

def top_k_logits(logits, k):
    if k == 0:
        return logits
    values = np.sort(logits)[-k]
    logits[logits < values] = -float('Inf')
    return logits

class CooONNXInference:
    def __init__(self, onnx_path="docs/model.onnx", tokenizer_path="data/tokenizer.json"):
        import onnxruntime as ort
        if not os.path.exists(onnx_path):
            print("ONNX model not found. Generating it now...")
            from tools.model_export import export_onnx
            export_onnx(output_path=onnx_path)

        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.bos_id = self.tokenizer.token_to_id("<|bos|>")
        self.eos_id = self.tokenizer.token_to_id("<|eos|>")
        
        print("Loading ONNX model...")
        self.session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        print("ONNX model loaded successfully.")

    def format_prompt(self, user_text):
        return f"<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n"

    def chat(self, user_text, max_new_tokens=64, temperature=0.4, top_k=30):
        user_text = user_text.lower()
        user_text = re.sub(r'[^a-z\s]', '', user_text)
        user_text = re.sub(r'\bcoo\b', '', user_text).strip()
        
        prompt = self.format_prompt(user_text)
        encoded = self.tokenizer.encode(prompt)
        input_ids = [self.bos_id] + encoded.ids
        
        for _ in range(max_new_tokens):
            inputs = {self.session.get_inputs()[0].name: np.array([input_ids], dtype=np.int64)}
            logits = self.session.run(None, inputs)[0]
            
            # get last token logits
            next_token_logits = logits[0, -1, :] / temperature
            next_token_logits = top_k_logits(next_token_logits, top_k)
            
            # softmax
            exp_logits = np.exp(next_token_logits - np.max(next_token_logits))
            probs = exp_logits / np.sum(exp_logits)
            
            # sample
            next_id = np.random.choice(len(probs), p=probs)
            input_ids.append(next_id)
            
            if next_id == self.eos_id:
                break

        full_text = self.tokenizer.decode(input_ids, skip_special_tokens=False)
        marker = "<|im_start|>assistant\n"
        idx = full_text.rfind(marker)
        if idx != -1:
            response = full_text[idx + len(marker):]
        else:
            response = full_text[len(prompt):]

        response = response.replace("<|im_end|>", "").replace("<|im_start|>", "").replace("<|eos|>", "").replace("<|bos|>", "").replace("<|pad|>", "").strip()
        return response

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
