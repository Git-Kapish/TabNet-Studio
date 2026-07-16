import torch
import numpy as np
from tabnet.layers import Sparsemax, GhostBatchNorm1d

def test_sparsemax_properties():
    """
    Test properties of Sparsemax:
    - Sums to 1.0 along the specified dimension.
    - Values are between 0.0 and 1.0.
    - Promotes sparsity (some values are exactly 0.0).
    """
    sparsemax = Sparsemax(dim=-1)
    
    # Simple input vector where we expect sparsity
    x = torch.tensor([[2.0, 1.0, 0.0, -1.0, -2.0]])
    out = sparsemax(x)
    
    # 1. Check sums to 1.0
    sums = torch.sum(out, dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums))
    
    # 2. Check bounds
    assert torch.all(out >= 0.0)
    assert torch.all(out <= 1.0)
    
    # 3. Check sparsity: smaller inputs should be mapped to exactly 0.0
    # For [2, 1, 0, -1, -2], sparsemax should keep the largest values and zero out the others.
    assert out[0, 3] == 0.0
    assert out[0, 4] == 0.0


def test_sparsemax_gradients():
    """
    Test that gradients flow correctly through Sparsemax.
    """
    sparsemax = Sparsemax(dim=-1)
    
    x = torch.tensor([[1.5, 2.0, 0.5]], requires_grad=True)
    out = sparsemax(x)
    
    loss = torch.sum(out ** 2)
    loss.backward()
    
    # Check that gradient is not None and is non-zero
    assert x.grad is not None
    assert torch.sum(torch.abs(x.grad)) > 0.0


def test_ghost_batch_norm():
    """
    Test shape and functionality of GhostBatchNorm1d.
    """
    input_dim = 16
    batch_size = 32
    virtual_batch_size = 8
    
    ghost_bn = GhostBatchNorm1d(
        input_dim=input_dim,
        virtual_batch_size=virtual_batch_size,
        momentum=0.1
    )
    
    x = torch.randn(batch_size, input_dim)
    
    # 1. Test training mode
    ghost_bn.train()
    out_train = ghost_bn(x)
    assert out_train.shape == (batch_size, input_dim)
    
    # 2. Test evaluation mode
    ghost_bn.eval()
    out_eval = ghost_bn(x)
    assert out_eval.shape == (batch_size, input_dim)
