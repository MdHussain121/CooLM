from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 3269
    d_model: int = 640
    n_layers: int = 8
    n_heads: int = 10
    ffn_hidden: int = 2560
    dropout: float = 0.1
    max_seq_len: int = 64
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2


@dataclass
class TrainConfig:
    batch_size: int = 64
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    warmup_steps: int = 200
    max_steps: int = 10000
    eval_interval: int = 200
    save_interval: int = 500
    log_interval: int = 10
    gradient_clip: float = 1.0
