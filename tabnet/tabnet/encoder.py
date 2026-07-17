import torch
import torch.nn as nn
from typing import List, Tuple
from tabnet.feature_transformer import SharedFeatureTransformer, FeatureTransformer
from tabnet.attentive_transformer import AttentiveTransformer


class TabNetEncoder(nn.Module):
    """
    TabNet sequential decision-step encoder.

    Runs N_steps decision steps.  At each step i the encoder:
      1. Generates a sparse feature selection mask M[i] (Eq. 2)
      2. Applies the mask to the BN-normalised input: masked_x = M[i] ⊙ f  (Eq. 3)
      3. Passes masked_x through the step-specific Feature Transformer         (Eq. 4)
      4. Splits the output into decision representation d[i] and context a[i]
      5. Updates the prior scale:  P[i] = P[i-1] · (γ − M[i])               (Eq. 5)

    The prior scale (Eq. 5) discourages repeated selection of the same feature
    across steps — γ controls the relaxation of this penalty.

    Reference: Arik & Pfister (2019), §3.1 "Sequential Attention", Eqs. 2–5.
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
            input_dim         : Dimensionality of the BN-normalised input features.
            n_d               : Width of the decision representation d[i]  (N_d, §3.1).
            n_a               : Width of the attentive context a[i]        (N_a, §3.1).
            n_steps           : Number of sequential decision steps         (N_steps, §3.1).
            gamma             : Relaxation factor γ for prior scale updates (Eq. 5).
                                γ=1 forces strict non-overlap; γ→∞ allows free reuse.
            n_shared          : Number of shared GLU layers in Feature Transformer (§3.1).
            n_dependent       : Number of step-specific GLU layers per step (§3.1).
            virtual_batch_size: Virtual sub-batch size for Ghost BatchNorm  (§3.1).
            momentum          : BN momentum (paper default 0.02).
        """
        super(TabNetEncoder, self).__init__()

        self.input_dim = input_dim
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma

        # Shared Feature Transformer layers — weights shared across all steps (§3.1)
        self.shared_part = SharedFeatureTransformer(
            input_dim=input_dim,
            output_dim=n_d + n_a,
            n_layers=n_shared,
            virtual_batch_size=virtual_batch_size,
            momentum=momentum
        ) if n_shared > 0 else None

        # One Feature Transformer per step (step-dependent layers + shared part)
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

        # One Attentive Transformer per step (§3.1, Eq. 2)
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
        Run all N_steps sequential decision steps.

        Args:
            x : BN-normalised input features, shape (B, input_dim).

        Returns:
            step_masks       : List of N_steps sparse masks M[i], each (B, input_dim).
            decision_outputs : List of N_steps decision tensors d[i], each (B, n_d).
        """
        batch_size = x.size(0)

        # P[0] = 1  (Eq. 5 initialisation — every feature equally available)
        prior_scales = torch.ones((batch_size, self.input_dim), device=x.device, dtype=x.dtype)

        # a[0] = 0  (no prior context at the first step, §3.1)
        a = torch.zeros((batch_size, self.n_a), device=x.device, dtype=x.dtype)

        step_masks: List[torch.Tensor] = []
        decision_outputs: List[torch.Tensor] = []

        for i in range(self.n_steps):
            # Eq. 2 — Attentive mask using prior context and prior scales
            mask = self.attentive_transformers[i](a, prior_scales)
            step_masks.append(mask)

            # Eq. 3 — Masked feature input: M[i] ⊙ f
            x_masked = mask * x

            # Eq. 4 — Feature Transformer (shared + step-dependent GLU layers)
            out = self.feature_transformers[i](x_masked)

            # Split into decision representation d[i] and next-step context a[i]
            d, a = torch.split(out, [self.n_d, self.n_a], dim=-1)
            decision_outputs.append(d)

            # Eq. 5 — Prior scale update: P[i] = P[i-1] · (γ − M[i])
            prior_scales = prior_scales * (self.gamma - mask)

        return step_masks, decision_outputs
