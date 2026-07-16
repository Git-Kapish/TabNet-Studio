import torch
import torch.nn as nn
from typing import List, Tuple
from tabnet.feature_transformer import SharedFeatureTransformer, FeatureTransformer
from tabnet.attentive_transformer import AttentiveTransformer

class TabNetEncoder(nn.Module):
    """
    TabNet Encoder module.
    Runs the sequential decision steps using attentive transformers for feature selection
    and feature transformers for representation learning.
    """
    def __init__(
        self,
        input_dim: int,
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
            input_dim: Dimensionality of input features (after tabular embedding).
            n_d: Dimension of the decision representation.
            n_a: Dimension of the attentive transformer feedback context.
            n_steps: Number of decision steps (N_steps).
            gamma: Relaxation parameter for prior scale updates.
            n_shared: Number of shared layers in the feature transformer block.
            n_dependent: Number of step-dependent layers in the feature transformer block.
            virtual_batch_size: Virtual batch size for Ghost BatchNorm.
            momentum: Momentum for standard and Ghost BatchNorm layers.
        """
        super(TabNetEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        
        # Instantiate the shared feature transformer block
        self.shared_part = SharedFeatureTransformer(
            input_dim=input_dim,
            output_dim=n_d + n_a,
            n_layers=n_shared,
            virtual_batch_size=virtual_batch_size,
            momentum=momentum
        ) if n_shared > 0 else None
        
        # List of step-specific feature transformers (each uses the shared_part under the hood)
        self.feature_transformers = nn.ModuleList([
            FeatureTransformer(
                input_dim=input_dim,
                output_dim=n_d + n_a,
                shared_part=self.shared_part,
                n_dependent=n_dependent,
                virtual_batch_size=virtual_batch_size,
                momentum=momentum
            )
            for _ in range(n_steps)
        ])
        
        # List of step-specific attentive transformers
        self.attentive_transformers = nn.ModuleList([
            AttentiveTransformer(
                input_dim=input_dim,
                n_a=n_a,
                momentum=momentum
            )
            for _ in range(n_steps)
        ])

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Args:
            x: Input feature tensor of shape (batch_size, input_dim)
        Returns:
            step_masks: List of feature selection masks for each step, each of shape (batch_size, input_dim)
            decision_outputs: List of decision representation tensors for each step, each of shape (batch_size, n_d)
        """
        batch_size = x.size(0)
        
        # Initialize prior scales P[0] to all ones: shape (batch_size, input_dim)
        prior_scales = torch.ones((batch_size, self.input_dim), device=x.device, dtype=x.dtype)
        
        # Initialize attentive context a[0] to all zeros: shape (batch_size, n_a)
        a = torch.zeros((batch_size, self.n_a), device=x.device, dtype=x.dtype)
        
        step_masks = []
        decision_outputs = []
        
        for i in range(self.n_steps):
            # 1. Get feature selection mask M[i] using context from the previous step
            mask = self.attentive_transformers[i](a, prior_scales)
            step_masks.append(mask)
            
            # 2. Apply the attention mask to input features (multiplicative selection)
            x_masked = mask * x
            
            # 3. Pass masked features through the feature transformer
            out = self.feature_transformers[i](x_masked)
            
            # 4. Split output into decision output d[i] and context a[i]
            d, a = torch.split(out, [self.n_d, self.n_a], dim=-1)
            decision_outputs.append(d)
            
            # 5. Update prior scales for the next step
            # P[i] = P[i-1] * (gamma - M[i])
            prior_scales = prior_scales * (self.gamma - mask)
            
        return step_masks, decision_outputs
