import torch
import torch.nn as nn
from tabnet.layers import GhostBatchNorm1d

class GLUBlock(nn.Module):
    """
    A single block within the Feature Transformer:
    Linear Layer -> Ghost Batch Normalization -> Gated Linear Unit (GLU).
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        virtual_batch_size: int = 128,
        momentum: float = 0.02
    ):
        super(GLUBlock, self).__init__()
        # GLU splits input of size 2 * output_dim into two halves of size output_dim.
        # So the FC layer must output 2 * output_dim.
        self.fc = nn.Linear(input_dim, output_dim * 2, bias=False)
        self.bn = GhostBatchNorm1d(output_dim * 2, virtual_batch_size=virtual_batch_size, momentum=momentum)
        self.glu = nn.GLU(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.glu(self.bn(self.fc(x)))


class SharedFeatureTransformer(nn.Module):
    """
    Shared part of the Feature Transformer, shared across all decision steps.
    Usually composed of 2 layers.
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        n_layers: int = 2,
        virtual_batch_size: int = 128,
        momentum: float = 0.02
    ):
        super(SharedFeatureTransformer, self).__init__()
        self.glu_blocks = nn.ModuleList()
        
        # First layer maps input_dim to output_dim
        self.glu_blocks.append(GLUBlock(input_dim, output_dim, virtual_batch_size, momentum))
        
        # Subsequent layers map output_dim to output_dim
        for _ in range(1, n_layers):
            self.glu_blocks.append(GLUBlock(output_dim, output_dim, virtual_batch_size, momentum))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # First layer (no residual connection as dimensions differ)
        x = self.glu_blocks[0](x)
        
        # Subsequent layers with residual connections scaled by sqrt(0.5)
        scale = torch.sqrt(torch.tensor(0.5, device=x.device, dtype=x.dtype))
        for block in self.glu_blocks[1:]:
            x = (x + block(x)) * scale
        return x


class FeatureTransformer(nn.Module):
    """
    Complete Feature Transformer block for a single decision step.
    Combines the shared layers with decision step-dependent layers.
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
        super(FeatureTransformer, self).__init__()
        self.shared_part = shared_part
        self.dependent_blocks = nn.ModuleList()
        
        start_dim = output_dim if shared_part is not None else input_dim
        
        if n_dependent > 0:
            self.dependent_blocks.append(GLUBlock(start_dim, output_dim, virtual_batch_size, momentum))
            for _ in range(1, n_dependent):
                self.dependent_blocks.append(GLUBlock(output_dim, output_dim, virtual_batch_size, momentum))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.shared_part is not None:
            x = self.shared_part(x)
            
        if len(self.dependent_blocks) > 0:
            scale = torch.sqrt(torch.tensor(0.5, device=x.device, dtype=x.dtype))
            for i, block in enumerate(self.dependent_blocks):
                # Apply residual connection if dimensions match:
                # Either we have a shared part (so input dimension to the dependent block is output_dim),
                # or we are on step index > 0.
                if self.shared_part is not None or i > 0:
                    x = (x + block(x)) * scale
                else:
                    x = block(x)
        return x
