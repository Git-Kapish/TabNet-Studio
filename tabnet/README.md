# TabNet Standalone Package

This directory contains a standalone, reusable PyTorch implementation of the TabNet architecture based on the paper **"TabNet: Attentive Interpretable Tabular Learning"**.

## Installation

To install this package locally in editable mode, run:
```bash
pip install -e .
```

## Basic Usage

```python
import torch
from tabnet import TabNetClassifier, Trainer

# Initialize the classifier
model = TabNetClassifier(
    num_features=10,
    num_classes=2,
    cat_idxs=[0, 3],
    cat_dims=[5, 10],
    cat_emb_dims=[4, 8],
    n_d=8,
    n_a=8,
    n_steps=3
)

# Run a forward pass
x = torch.randn(16, 10)
# Mock categorical values to be indices within limits
x[:, 0] = torch.randint(0, 5, (16,)).float()
x[:, 3] = torch.randint(0, 10, (16,)).float()

logits, step_masks, decision_outputs = model(x)
print("Logits shape:", logits.shape)  # torch.Size([16, 2])
```
