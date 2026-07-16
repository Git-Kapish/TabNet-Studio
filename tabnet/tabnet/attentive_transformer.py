import torch
import torch.nn as nn
from tabnet.layers import Sparsemax

class AttentiveTransformer(nn.Module):
    """
    Attentive Transformer block.
    Generates feature selection masks using the processed context of the previous step.
    Formula: M[i] = sparsemax(P[i-1] * h_i(a[i-1]))
    """
    def __init__(
        self,
        input_dim: int,
        n_a: int,
        momentum: float = 0.02
    ):
        """
        Args:
            input_dim: Dimension of features to select from (D).
            n_a: Input dimension of context vector (Na) from previous step.
            momentum: Momentum for Batch Normalization.
        """
        super(AttentiveTransformer, self).__init__()
        
        # FC layer mapping from context dimension to feature dimension
        self.fc = nn.Linear(n_a, input_dim, bias=False)
        
        # Standard Batch Normalization (applied along features)
        self.bn = nn.BatchNorm1d(input_dim, momentum=momentum)
        
        # Sparsemax to ensure sparse, probability-simplex selection masks
        self.sparsemax = Sparsemax(dim=-1)

    def forward(self, a: torch.Tensor, prior_scales: torch.Tensor) -> torch.Tensor:
        """
        Args:
            a: Context vector from previous step, shape (batch_size, n_a)
            prior_scales: Prior usage scales of features, shape (batch_size, input_dim)
        Returns:
            Feature selection mask M[i], shape (batch_size, input_dim)
        """
        x = self.fc(a)
        x = self.bn(x)
        # Apply prior scales
        x = x * prior_scales
        # Normalize with sparsemax
        mask = self.sparsemax(x)
        return mask
