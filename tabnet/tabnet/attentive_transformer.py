import torch
import torch.nn as nn
from tabnet.layers import Sparsemax


class AttentiveTransformer(nn.Module):
    """
    Attentive Transformer — feature selection mask generator.

    At each decision step i, produces a sparse mask M[i] that selects which
    input features to attend to.  Implements Eq. (2) from the paper:

        h_i(a[i-1]) = BN(W_a · a[i-1])
        M[i] = sparsemax(P[i-1] ⊙ h_i(a[i-1]))

    where:
        a[i-1]    : processed representation from the previous step  (n_a)
        P[i-1]    : prior scale encoding how much each feature has
                    already been used (initialised to ones, §3.2)
        W_a       : learnable weight matrix (no bias, §3.1)
        sparsemax : ensures the mask sums to 1 and is sparse (§3.1)

    Reference: Arik & Pfister (2019), §3.1 "Feature Selection", Eq. 2.
    """

    def __init__(
        self,
        input_dim: int,
        n_a: int,
        momentum: float = 0.02
    ):
        """
        Args:
            input_dim : Number of input features D (mask dimension).
            n_a       : Dimension of the context vector a[i-1] (N_a).
            momentum  : Momentum for Batch Normalization (paper default 0.02).
        """
        super(AttentiveTransformer, self).__init__()

        # W_a: maps context dimension n_a → feature dimension (no bias, §3.1)
        self.fc = nn.Linear(n_a, input_dim, bias=False)

        # Batch Normalization on the projected context (standard BN, §3.1)
        self.bn = nn.BatchNorm1d(input_dim, momentum=momentum)

        # Sparsemax: maps to a sparse probability simplex (Martins & Astudillo, 2016)
        self.sparsemax = Sparsemax(dim=-1)

    def forward(self, a: torch.Tensor, prior_scales: torch.Tensor) -> torch.Tensor:
        """
        Compute the feature selection mask M[i].

        Args:
            a            : Context vector from step i-1, shape (B, n_a).
            prior_scales : Prior usage scales P[i-1], shape (B, input_dim).

        Returns:
            mask : Sparse feature selection mask M[i], shape (B, input_dim).
                   Each row sums to 1 (sparsemax simplex constraint).
        """
        # h_i(a[i-1]) = BN(W_a · a[i-1])
        x = self.bn(self.fc(a))
        # Element-wise modulation by prior scales then sparsemax (Eq. 2)
        mask = self.sparsemax(x * prior_scales)
        return mask
