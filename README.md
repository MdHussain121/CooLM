
<p align="center">
  <img src="assets/coolm.png" alt="CooLM" width="400"/>
</p>

<h1 align="center">CooLM</h1>
<p align="center"><em>A ~42M parameter LM that talks like a rooftop pigeon.</em></p>

<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/🤗_Dataset-coolm--500k-blue" alt="Dataset"/></a>&nbsp;
  <a href="https://huggingface.co/MdHussain121/coolm-42M"><img src="https://img.shields.io/badge/🤗_Model-coolm--42M-orange" alt="Model"/></a>&nbsp;
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green" alt="License"/></a>
</p>

---

> **This project exists to show that training your own language model is not magic.**
> No PhD required. No massive GPU cluster. One local machine, a few minutes, and you have a working LLM that you built from scratch — data generation, tokenizer, model architecture, training loop, and inference.
>
> It won't produce a billion-parameter model that writes essays. But it will show you exactly how every piece works — from raw text to trained weights to generated output — so the big models stop feeling like black boxes.

---

![CooLM Chat Preview](assets/chat_preview.png)

---

## What is CooLM?

CooLM is a tiny language model that pretends to be a pigeon named Coo. It speaks in short, lowercase sentences about rooftops, food, weather, and surviving in the city. It doesn't understand human abstractions like money, phones, or politics — and it's not trying to.

It's trained from scratch on 500K synthetic conversations across 13 topics, can be trained locally on a single GPU, and serves chat through a small local Python server (ONNX Runtime) — there is no in-browser runtime; inference always runs server-side.

---

## Architecture

| | |
|---|---|
| **Parameters** | 41,476,480 |
| **Layers** | 8 |
| **Hidden dim** | 640 |
| **Heads** | 10 |
| **FFN** | 2560 (GeLU) |
| **Vocab** | 3,269 (BPE) |
| **Max sequence** | 64 tokens |
| **Norm** | LayerNorm |
| **Position** | Learned embeddings |
| **LM head** | Weight-tied with embeddings |

Vanilla decoder-only transformer. No GQA, no RoPE, no SwiGLU. As simple as it gets.

---

## Training Performance

This was my first time ever building a language model from scratch! The entire training process ran locally on a standard gaming laptop (**NVIDIA RTX 4060 Laptop GPU**) and completed in 22 minutes.

```text
================================================================================
done. 10000 steps in 22.0m (1322s)
tokens processed: 8,968,042
avg throughput: 6784 tok/s
best val loss: 0.3262 (ppl 1.39)
```

Two notes on reading these numbers honestly:

- **tok/s counts masked response tokens only** — roughly 14 active tokens per 64-token row (the prompt and padding are masked out of the loss), so raw tensor throughput is much higher.
- **best val loss 0.3262 (ppl 1.39) is in-distribution template-retrieval performance.** The shipped 95/5 validation split is random, so ~95% of validation outputs appear verbatim in the training set; this number does not measure open-domain generalization. Run `python -m tools.evaluate` for template-disjoint and topic-disjoint numbers with per-split leakage statistics.

---

## Personality

Coo:
- Speaks in short, lowercase sentences with simple punctuation.
- Experiences the world through weather, rooftops, loud city sounds, and food.
- Doesn't understand human abstractions (alarms, jobs, money).
- Is observant, slightly anxious about threats, and always looking for a snack.

**13 topics:** birds, city_sounds, confused, emotions, flight, food, greetings, nesting, perch, rain, sleep, threats, weather.

---

## Quick Start

### Chat locally

You can run the entire pipeline or chat locally using the provided Windows batch files:

- **`train_coolm.bat`**: Installs requirements, generates the 500k dataset, prepares tokenizer/tensors, trains the model, and exports it to ONNX (the full generate → prepare → train → export pipeline).
- **`chat.bat`**: Spins up the optimized local Python web server. It prompts you to choose between the **Hugging Face version** (downloads and caches the pre-trained weights from `MdHussain121/coolm-42M` using a robust, built-in chunked downloader) or your own **local version**. It then automatically opens your browser to the beautiful dark mode UI.

#### Manual Step-by-Step Execution (No `.bat`)

If you prefer not to use the `.bat` files, you can execute the commands manually in your terminal:

1. **Install requirements**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Generate dataset** (500k samples):
   ```bash
   python generate_data.py 500000
   ```
3. **Prepare and tokenize dataset**:
   ```bash
   python -m coolm.prepare_data pigeon_data.jsonl
   ```
4. **Train the model**:
   ```bash
   python -m coolm.train
   ```
5. **Export the trained model to ONNX**:
   ```bash
   python -m tools.model_export
   ```
6. **Run the local chat server**:
   ```bash
   python -m tools.server
   ```
   *The server will prompt you to choose between the Hugging Face version (Option 1) and your own local version (Option 2). Once selected, it automatically opens `http://localhost:8000/index.html` in your browser.*
   *Flags: `--port` (default 8000), `--host` (default 127.0.0.1), `--no-browser` (skip auto-opening), `--model local|hf` (skip the interactive prompt).*

7. **Evaluate honestly** (template-disjoint / topic-disjoint splits with per-split leakage statistics):
   ```bash
   python -m tools.evaluate --data pigeon_data.jsonl --tokenizer data/tokenizer.json --checkpoint checkpoints/best_model.pt --out evaluation-report.json
   ```

---

## Dataset

| | |
|---|---|
| Samples | 500,000 (475K train / 25K val) |
| Format | `{"input": "...", "output": "...", "category": "..."}` |
| Categories | 13 |
| Generation | Synthetic template composition |

Composition: **inputs are 100% unique** (one prompt per row); **outputs come from ~750 response templates** — 51,691 distinct output strings across the 500,000 rows, a ~89.7% duplicate-output rate. The random 95/5 split is therefore template-contaminated (see [Model Scope & Evaluation](#model-scope--evaluation)).

```python
import json
with open("pigeon_data.jsonl", "r") as f:
    print(json.loads(f.readline()))
# {'input': 'i want to know why do cars honk so much maybe', 'output': 'horns honking all day long do the humans not have ears. coo coo.', 'category': 'city_sounds'}
```

---

## Project Structure

```
coolm/
├── config.py               Hyperparameters (model + training)
├── model.py                Vanilla transformer
├── prepare_data.py         Data prep + tokenizer training
├── train.py                Training loop (cosine LR, AMP)
└── inference.py            Chat interface

generate_data.py            Conversation data generator (13 topics)

tools/
├── check_lengths.py        Validates token sequence lengths
├── model_export.py         Export model to ONNX
├── verify_artifacts.py     Checkpoint/ONNX parity + tokenizer pairing gate
├── evaluate.py             Template-disjoint, leakage-measuring evaluation
└── export_hf.py            Push model artifacts (checkpoint, tokenizer, ONNX,
                             model card) to HuggingFace — not the dataset

docs/
└── index.html              Chat web UI (served by tools/server.py; inference
                             runs server-side in Python)

assets/
└── coolm.png               Logo
```

---

## Design Decisions

**Why no system prompt?** Every training sample had the same one. A small model can't conditionally follow instructions — the personality is baked into the weights.

**Input Normalization:** `inference.py` normalizes your prompt toward the training-text distribution: it lowercases, keeps letters, hyphens and whitespace (training inputs contain e.g. 'high-rise' and 'coo'), drops every other character, and collapses whitespace runs. Prompts that would overflow the 64-token context are token-wise truncated to the most recent user text, with a warning. Remaining limitation: heavy punctuation and digits are still stripped, which shifts far-out-of-distribution input away from what the user actually typed.

**Why synthetic data?** A pigeon character with consistent personality needs consistent training data. Template composition with randomized components generates 500K samples from basic templates. Categories receive equal *sampling weights* (1/13 each), but the global input-uniqueness retry skews the realized balance at 500K scale: measured per-category shares range from 5.5% to 9.9% rather than an even 7.7%.

**Vocabulary Optimization:** We optimized the BPE tokenizer to train exactly 3,269 tokens, and set `vocab_size` in config to match this value exactly. This leaves 0 unused embedding parameters, saving over 1 million redundant weights and eliminating parameter bloat.

**Dataset Uniqueness:** Prompt inputs are tracked for uniqueness in memory with a 100-collision retry cap; at 500K samples the cap was never hit (0 cap events, max retry depth 15), so inputs are 100% unique in practice — this is an empirical property, not a hard guarantee. Outputs are NOT unique by design: they are template responses drawn from ~750 response templates.

**Early Stopping:** To prevent overfitting on the synthetic dataset, the training loop tracks validation loss and halts training automatically if validation loss does not improve for 5 consecutive evaluations (patience = 5).

---

## Model Scope & Evaluation

CooLM is a narrow-domain **template-retrieval chatbot**, not an open-domain assistant. Independent evaluation (audit 2026-11-05, 462 generations across 77 prompts):

| Prompt style | Relevance (0–2) |
|---|---|
| Verbatim-style training questions | 1.96 |
| Paraphrased variations | 1.15 |
| Out-of-domain prompts | 0.32 |

The documented val loss 0.3262 (ppl 1.39) comes from a random 95/5 split in which 95.02% of validation outputs appear verbatim in the training set — it measures template retrieval, **not** open-domain generalization. Run `python -m tools.evaluate` for reproducible template-disjoint / topic-disjoint splits with per-split leakage statistics.

---

## Inspiration & Credits

CooLM was heavily inspired by **GuppyLM** (which demonstrated training a 9M parameter model on 60k samples). CooLM scales this up to a 42M parameter model trained on a 500k sample dataset with 100% unique prompts, validation tracking, early stopping, and an interactive dark mode web interface.

---

## License

MIT
