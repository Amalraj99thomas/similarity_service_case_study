import numpy as np

# OpenAI text-embedding-3-small scores
pairs = [
    # HIGH pairs (label=1)
    (0.8817, 1), (0.7875, 1), (0.7962, 1), (0.7273, 1),
    (0.4582, 1), (0.5617, 1), (0.7195, 1), (0.6345, 1),
    (0.8163, 1), (0.5888, 1),
    # MEDIUM pairs (label=0)
    (0.5977, 0), (0.6024, 0), (0.5918, 0), (0.5559, 0),
    (0.5695, 0), (0.6089, 0), (0.6346, 0), (0.5602, 0), (0.6872, 0),
    # LOW pairs (label=0)
    (0.4542, 0), (0.4373, 0),
]

best_f1, best_threshold = 0, 0
for threshold in np.arange(0.40, 0.95, 0.01):
    tp = sum(1 for s, l in pairs if s >= threshold and l == 1)
    fp = sum(1 for s, l in pairs if s >= threshold and l == 0)
    fn = sum(1 for s, l in pairs if s <  threshold and l == 1)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    if f1 > best_f1:
        best_f1, best_threshold = f1, threshold

print(f"Best threshold: {best_threshold:.2f}  F1: {best_f1:.3f}")