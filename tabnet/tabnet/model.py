import torch
import torch.nn as nn
from typing import List, Tuple
from tabnet.embeddings import TabularEmbedding
from tabnet.encoder import TabNetEncoder

class TabNetClassifier(nn.Module):
    """
    Pure PyTorch implementation of TabNet Classifier.
    Integrates categorical embedding layers, input normalization, the TabNet Encoder,
    and a final linear classification layer.
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
            num_features: Total number of features in the input dataset.
            num_classes: Number of classification targets.
            cat_idxs: List of indices of categorical features.
            cat_dims: List of cardinality (number of categories) for each categorical feature.
            cat_emb_dims: List of embedding dimensions for each categorical feature.
            n_d: Dimension of the decision representation.
            n_a: Dimension of the attentive transformer feedback context.
            n_steps: Number of decision steps.
            gamma: Relaxation parameter for prior scale updates.
            n_shared: Number of shared layers in the feature transformer.
            n_dependent: Number of step-dependent layers in the feature transformer.
            virtual_batch_size: Virtual batch size for Ghost BatchNorm.
            momentum: Momentum for standard and Ghost BatchNorm layers.
        """
        super(TabNetClassifier, self).__init__()
        
        # 1. Embeddings and Input Normalization
        self.embeddings = TabularEmbedding(
            num_features=num_features,
            cat_idxs=cat_idxs,
            cat_dims=cat_dims,
            cat_emb_dims=cat_emb_dims
        )
        
        # 2. Sequential Decision Encoder
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
        
        # 3. Final Classification Mapping
        # Maps the aggregated decision output dout to num_classes class logits.
        self.final_mapping = nn.Linear(n_d, num_classes, bias=False)

    def forward(
        self, 
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """
        Args:
            x: Input tensor of shape (batch_size, num_features)
        Returns:
            logits: Output logits of shape (batch_size, num_classes)
            step_masks: List of feature selection masks for each step, each of shape (batch_size, post_embed_dim)
            decision_outputs: List of decision representation tensors, each of shape (batch_size, n_d)
        """
        # Embed and normalize input
        x_embed = self.embeddings(x)
        
        # Pass through the encoder
        step_masks, decision_outputs = self.encoder(x_embed)
        
        # Aggregate decision outputs: dout = sum_i(ReLU(d[i]))
        d_out = sum(torch.relu(d) for d in decision_outputs)
        
        # Class logits
        logits = self.final_mapping(d_out)
        
        return logits, step_masks, decision_outputs
