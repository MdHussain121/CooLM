"""Check sample length distribution (non-padding content)."""
import torch, math
data = torch.load("data/train_data.pt", map_location="cpu", weights_only=True)
mask = torch.load("data/train_mask.pt", map_location="cpu", weights_only=True)

# Count non-padding tokens per sample
content_lengths = (data != 0).sum(dim=1).tolist()
output_lengths = mask.sum(dim=1).tolist()

content_lengths.sort()
output_lengths.sort()
n = len(content_lengths)

print(f"samples: {n}")
print(f"padded length: {data.size(1)}")
print(f"\n--- Content tokens (non-padding) ---")
print(f"min: {content_lengths[0]}, max: {content_lengths[-1]}")
print(f"mean: {sum(content_lengths)/n:.1f}")
for p in [50, 75, 90, 95, 99, 100]:
    idx = min(int(n * p / 100), n - 1)
    print(f"  {p}th percentile: {content_lengths[idx]}")

print(f"\n--- Output tokens (mask=1, loss-active) ---")
print(f"min: {output_lengths[0]}, max: {output_lengths[-1]}")
print(f"mean: {sum(output_lengths)/n:.1f}")
for p in [50, 75, 90, 95, 99, 100]:
    idx = min(int(n * p / 100), n - 1)
    print(f"  {p}th percentile: {output_lengths[idx]}")
