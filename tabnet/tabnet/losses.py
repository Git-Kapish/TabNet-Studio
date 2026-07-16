import torch
import torch.nn as nn
from typing import List

class SparsityLoss(nn.Module):
    """
    Sparsity regularization loss for TabNet.
    Penalizes non-sparse selections of features by computing the entropy of the step masks.
    Formula: L_sparse = (1 / (N_steps * B)) * sum_i sum_b sum_j -M_{b,j}[i] * log(M_{b,j}[i] + eps)
    """
    def __init__(self, eps: float = 1e-10):
        super(SparsityLoss, self).__init__()
        self.eps = eps

    def forward(self, step_masks: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            step_masks: List of feature selection masks for each step,
                         each of shape (batch_size, input_dim).
        Returns:
            Scalar sparsity loss tensor.
        """
        loss = 0.0
        n_steps = len(step_masks)
        
        if n_steps == 0:
            return torch.tensor(0.0, device=step_masks[0].device if len(step_masks) > 0 else "cpu")
            
        for mask in step_masks:
            # Compute entropy per sample in the batch: sum_j -M_{b,j} * log(M_{b,j} + eps)
            entropy = torch.sum(-mask * torch.log(mask + self.eps), dim=-1)
            # Take the mean over the batch
            loss += torch.mean(entropy)
            
        # Average over all decision steps
        return loss / n_steps
