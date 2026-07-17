import torch
import torch.nn as nn
from typing import List, Tuple
from tabnet.embeddings import TabularEmbedding
from tabnet.encoder import TabNetEncoder


class TabNetClassifier(nn.Module):
    """
    TabNet classifier — top-level PyTorch module.

    Wires together:
      1. **TabularEmbedding** — categorical embeddings + Input BN (§3.1, "Input BN")
      2. **TabNetEncoder**    — N_steps sequential attention blocks (§3.1, Eqs. 2–5)
      3. **Final FC layer**   — maps aggregated decision output to class logits (§3.2)

    The aggregated decision output is computed as (§3.2, Eq. 7):

        d_out = Σ_i ReLU(d[i])

    and mapped to logits by a bias-free linear layer.
    Cross-entropy loss is used for supervised classification.

    Reference: Arik & Pfister (2019), §3.1–§3.2.
    """

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        cat_idxs: List[int] = [],
        cat_dims: List[int] = [],
        cat_emb_dims: List[int] = [],
        n_d: int = 8,
        n_a: int = 8,
        n_steps: int = 5,
        gamma: float = 1.5,
        n_shared: int = 2,
        n_dependent: int = 2,
        virtual_batch_size: int = 128,
        momentum: float = 0.02
    ):
        """
        Args:
            num_features      : Total number of raw input features.
            num_classes       : Number of output classes.
            cat_idxs          : Indices of categorical columns in the input tensor.
            cat_dims          : Cardinalities (vocab sizes) for each categorical column.
            cat_emb_dims      : Embedding dimension for each categorical column.
            n_d               : Decision representation width N_d (§3.1).
            n_a               : Attentive context width N_a (§3.1).
            n_steps           : Number of sequential decision steps N_steps (§3.1).
            gamma             : Prior-scale relaxation factor γ (§3.1, Eq. 5).
            n_shared          : Shared GLU layers per Feature Transformer (§3.1).
            n_dependent       : Step-specific GLU layers per decision step (§3.1).
            virtual_batch_size: Virtual sub-batch size for Ghost BN (§3.1).
            momentum          : BN momentum (paper default 0.02).
        """
        super(TabNetClassifier, self).__init__()

        # 1. Categorical embeddings + Input Batch Normalisation (§3.1)
        self.embeddings = TabularEmbedding(
            num_features=num_features,
            cat_idxs=cat_idxs,
            cat_dims=cat_dims,
            cat_emb_dims=cat_emb_dims
        )

        # 2. Sequential decision-step encoder (§3.1, Eqs. 2–5)
        self.encoder = TabNetEncoder(
            input_dim=self.embeddings.post_embed_dim,
            n_d=n_d,
            n_a=n_a,
            n_steps=n_steps,
            gamma=gamma,
            n_shared=n_shared,
            n_dependent=n_dependent,
            virtual_batch_size=virtual_batch_size,
            momentum=momentum
        )

        # 3. Final linear mapping: d_out → class logits (§3.2, no bias)
        self.final_mapping = nn.Linear(n_d, num_classes, bias=False)

    def forward(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """
        Forward pass.

        Args:
            x: Raw input tensor of shape (B, num_features).

        Returns:
            logits          : Class logits of shape (B, num_classes).
            step_masks      : List of N_steps sparse masks M[i], each (B, input_dim).
            decision_outputs: List of N_steps decision tensors d[i], each (B, N_d).
        """
        # Input BN + categorical embedding (§3.1)
        x_embed = self.embeddings(x)

        # Sequential attention encoder → masks and per-step decision vectors
        step_masks, decision_outputs = self.encoder(x_embed)

        # Aggregate: d_out = Σ_i ReLU(d[i])  (§3.2, Eq. 7)
        d_out = sum(torch.relu(d) for d in decision_outputs)

        # Map to logits
        logits = self.final_mapping(d_out)

        return logits, step_masks, decision_outputs
