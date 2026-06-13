import os
import torch
from coolm.config import ModelConfig
from coolm.model import CooLM

def export_onnx(checkpoint_path="checkpoints/best_model.pt", output_path="docs/model.onnx"):
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint {checkpoint_path} not found.")
        return

    print("Loading model...")
    device = "cpu"
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt.get("model_config", ModelConfig())
    model = CooLM(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Create dummy input for tracing
    dummy_input = torch.randint(0, cfg.vocab_size, (1, 32), dtype=torch.long, device=device)

    class ONNXWrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, x):
            logits, _, _ = self.m(x, use_cache=False)
            return logits

    wrapper = ONNXWrapper(model)
    wrapper.eval()

    print(f"Exporting to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # We only export the forward pass that returns logits.
    # The actual inference loop must be handled in JS or Python.
    torch.onnx.export(
        wrapper, 
        dummy_input, 
        output_path, 
        export_params=True, 
        opset_version=14, 
        do_constant_folding=True,
        input_names=["input_ids"], 
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size", 1: "sequence_length"}
        }
    )
    
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Done! ONNX model size: {size_mb:.2f} MB")

if __name__ == "__main__":
    export_onnx()
