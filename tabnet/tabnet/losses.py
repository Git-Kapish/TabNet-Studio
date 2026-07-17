import torch
import torch.nn as nn
from typing import List


class SparsityLoss(nn.Module):
    """
    Sparsity regularisation loss for TabNet feature selection masks.

    Penalises feature selection masks that are close to uniform (i.e. not sparse)
    by measuring the entropy of each mask.  Lower entropy means sparser, more
    selective attention — which is the desired inductive bias of TabNet.

    Formula (Eq. 9, §3.4):
        L_sparse = (1 / N_steps) · Σ_i Σ_b Σ_j  −M_{b,j}[i] · log(M_{b,j}[i] + ε)

    The total training loss is:
        L = L_supervised + λ_sparse · L_sparse   (Eq. 9)

    where λ_sparse (lambda_sparse) is a hyperparameter that controls the
    strength of the sparsity regularisation.

    Reference: Arik & Pfister (2019), §3.4 "Training Details", Eq. 9.
    """

    def __init__(self, eps: float = 1e-10):
        """
        Args:
            eps: Small constant added inside log for numerical stability.
        """
        super(SparsityLoss, self).__init__()
        self.eps = eps

    def forward(self, step_masks: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute the sparsity regularisation loss over all decision steps.

        Args:
            step_masks: List of N_steps feature selection masks,
                        each of shape (B, input_dim).  Values should lie in [0,1]
                        (guaranteed by sparsemax).

        Returns:
            Scalar sparsity loss tensor (averaged across steps and batch).
        """
        n_steps = len(step_masks)
        if n_steps == 0:
            return torch.tensor(0.0)

        loss = torch.tensor(0.0, device=step_masks[0].device)
        for mask in step_masks:
            # Entropy per sample: Σ_j −M_{b,j} · log(M_{b,j} + ε)
            entropy = torch.sum(-mask * torch.log(mask + self.eps), dim=-1)
            # Average over the mini-batch
            loss = loss + torch.mean(entropy)

        # Average over all decision steps (Eq. 9)
        return loss / n_steps
