import torch
import torch.nn as nn
from tabnet.layers import GhostBatchNorm1d


class GLUBlock(nn.Module):
    """
    Gated Linear Unit block — the elementary building block of the Feature Transformer.

    Implements the GLU transformation from §3.1 of the paper:

        GLU(x) = (FC(x)[..., :N_d] + FC(x)[..., N_d:]) ⊙ σ(FC(x)[..., N_d:])

    In practice this is: FC(2·N_d) → Ghost BN → nn.GLU, where nn.GLU splits the
    last dimension in half and applies an element-wise sigmoid gate to the second half.

    Reference: Arik & Pfister (2019), §3.1 "Feature Transformer", Eq. 6.
    Also: Dauphin et al. (2017), "Language Modeling with Gated Convolutional Networks".
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        virtual_batch_size: int = 128,
        momentum: float = 0.02
    ):
        """
        Args:
            input_dim         : Dimension of input features.
            output_dim        : Dimension of output features (N_d or N_d + N_a).
            virtual_batch_size: Virtual sub-batch size for Ghost BatchNorm.
            momentum          : Ghost BN momentum.
        """
        super(GLUBlock, self).__init__()
        # FC outputs 2·output_dim so GLU can split evenly into gate + value halves
        self.fc = nn.Linear(input_dim, output_dim * 2, bias=False)
        self.bn = GhostBatchNorm1d(output_dim * 2, virtual_batch_size=virtual_batch_size, momentum=momentum)
        self.glu = nn.GLU(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # FC → Ghost BN → GLU  (§3.1, Eq. 6)
        return self.glu(self.bn(self.fc(x)))


class SharedFeatureTransformer(nn.Module):
    """
    Shared portion of the Feature Transformer.

    Weights in this sub-network are shared across all N_steps decision steps,
    providing a common representation basis.  Residual connections are scaled
    by √0.5 to keep activation variances stable (§3.1, Eq. 6):

        h = (h_prev + GLUBlock(h_prev)) · √0.5

    Reference: Arik & Pfister (2019), §3.1 "Feature Transformer".
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        n_layers: int = 2,
        virtual_batch_size: int = 128,
        momentum: float = 0.02
    ):
        """
        Args:
            input_dim         : Input feature dimension.
            output_dim        : Output dimension (N_d + N_a).
            n_layers          : Number of GLU layers (paper uses 2 shared layers).
            virtual_batch_size: Virtual sub-batch size for Ghost BatchNorm.
            momentum          : Ghost BN momentum.
        """
        super(SharedFeatureTransformer, self).__init__()
        self.glu_blocks = nn.ModuleList()

        # First layer: input_dim → output_dim  (dimension change, no residual)
        self.glu_blocks.append(GLUBlock(input_dim, output_dim, virtual_batch_size, momentum))

        # Subsequent layers: output_dim → output_dim  (with √0.5 residual)
        for _ in range(1, n_layers):
            self.glu_blocks.append(GLUBlock(output_dim, output_dim, virtual_batch_size, momentum))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # First block — no residual (dimensions differ)
        x = self.glu_blocks[0](x)

        # Remaining blocks — √0.5 scaled residual connection (§3.1, Eq. 6)
        scale = torch.sqrt(torch.tensor(0.5, device=x.device, dtype=x.dtype))
        for block in self.glu_blocks[1:]:
            x = (x + block(x)) * scale
        return x


class FeatureTransformer(nn.Module):
    """
    Complete Feature Transformer for a single decision step.

    Combines the cross-step shared layers with step-specific (dependent) layers.
    The full transformer output has dimension N_d + N_a, which is then split into
    the decision vector d[i] (N_d) and attentive context a[i] (N_a).

    All internal residual connections use the √0.5 scaling factor (§3.1, Eq. 6).

    Reference: Arik & Pfister (2019), §3.1 "Feature Transformer", Fig. 2.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        shared_part: nn.Module = None,
        n_dependent: int = 2,
        virtual_batch_size: int = 128,
        momentum: float = 0.02
    ):
        """
        Args:
            input_dim         : Input feature dimension.
            output_dim        : Output dimension (N_d + N_a).
            shared_part       : Pre-built SharedFeatureTransformer (shared weights).
            n_dependent       : Number of step-specific GLU layers.
            virtual_batch_size: Virtual sub-batch size for Ghost BatchNorm.
            momentum          : Ghost BN momentum.
        """
        super(FeatureTransformer, self).__init__()
        self.shared_part = shared_part
        self.dependent_blocks = nn.ModuleList()

        # If shared_part exists, its output is output_dim; otherwise start from input_dim
        start_dim = output_dim if shared_part is not None else input_dim

        if n_dependent > 0:
            # First dependent block: start_dim → output_dim
            self.dependent_blocks.append(GLUBlock(start_dim, output_dim, virtual_batch_size, momentum))
            # Subsequent dependent blocks: output_dim → output_dim
            for _ in range(1, n_dependent):
                self.dependent_blocks.append(GLUBlock(output_dim, output_dim, virtual_batch_size, momentum))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Run shared layers first (if any)
        if self.shared_part is not None:
            x = self.shared_part(x)

        # Run step-specific layers with √0.5 residual connections (§3.1, Eq. 6)
        if len(self.dependent_blocks) > 0:
            scale = torch.sqrt(torch.tensor(0.5, device=x.device, dtype=x.dtype))
            for i, block in enumerate(self.dependent_blocks):
                if self.shared_part is not None or i > 0:
                    x = (x + block(x)) * scale
                else:
                    x = block(x)  # First block when no shared part — no residual
        return x
