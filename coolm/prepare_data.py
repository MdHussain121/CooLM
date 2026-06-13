"""Prepare data for CooLM — per-sample with loss masks."""

import json
import os
import torch

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors


SPECIAL_TOKENS = [
    "<|pad|>", "<|bos|>", "<|eos|>", "<|unk|>",
    "<|im_start|>", "<|im_end|>",
]


def train_tokenizer(data_path, vocab_size=4096, output_dir="data"):
    os.makedirs(output_dir, exist_ok=True)

    all_text = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            user_text = obj.get("input", "")
            assistant_text = obj.get("output", "")
            sample = f"<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n{assistant_text}<|im_end|>"
            all_text.append(sample)

    raw_path = os.path.join(output_dir, "raw_text.txt")
    with open(raw_path, "w", encoding="utf-8") as f:
        f.write("\n".join(all_text))

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        min_frequency=2,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    tokenizer.train([raw_path], trainer)

    tokenizer_path = os.path.join(output_dir, "tokenizer.json")
    tokenizer.save(tokenizer_path)
    print(f"tokenizer saved to {tokenizer_path}")

    return tokenizer


def tokenize_dataset(data_path, tokenizer, output_dir="data", pad_len=64):
    os.makedirs(output_dir, exist_ok=True)

    bos_id = tokenizer.token_to_id("<|bos|>")
    eos_id = tokenizer.token_to_id("<|eos|>")
    pad_id = tokenizer.token_to_id("<|pad|>")

    all_tokens = []
    all_masks = []
    total_tokens = 0
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            user_text = obj.get("input", "")
            assistant_text = obj.get("output", "")

            input_text = f"<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n"
            output_text = f"{assistant_text}<|im_end|>"

            input_ids = [bos_id] + tokenizer.encode(input_text).ids
            output_ids = tokenizer.encode(output_text).ids + [eos_id]

            full_ids = input_ids + output_ids
            mask = [0] * len(input_ids) + [1] * len(output_ids)

            if len(full_ids) > pad_len:
                full_ids = full_ids[:pad_len]
                mask = mask[:pad_len]

            if len(full_ids) < pad_len:
                n = pad_len - len(full_ids)
                full_ids += [pad_id] * n
                mask += [0] * n

            all_tokens.append(full_ids)
            all_masks.append(mask)
            total_tokens += sum(mask)  # count actual output tokens in this sample

    data_tensor = torch.tensor(all_tokens, dtype=torch.long)
    mask_tensor = torch.tensor(all_masks, dtype=torch.long)

    print(f"sequences: {len(data_tensor):,}")
    print(f"pad_len: {pad_len}")

    data_path_save = os.path.join(output_dir, "train_data.pt")
    mask_path_save = os.path.join(output_dir, "train_mask.pt")
    torch.save(data_tensor, data_path_save)
    torch.save(mask_tensor, mask_path_save)
    print(f"data saved to {data_path_save}")
    print(f"mask saved to {mask_path_save}")

    return data_tensor, mask_tensor


if __name__ == "__main__":
    import sys
    data_path = sys.argv[1] if len(sys.argv) > 1 else "pigeon_data.jsonl"
    tokenizer = train_tokenizer(data_path, vocab_size=3269)
    data, masks = tokenize_dataset(data_path, tokenizer)
    print(f"vocab: {tokenizer.get_vocab_size()}")
