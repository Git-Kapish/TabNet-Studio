import torch
import torch.nn as nn


class SparsemaxFunction(torch.autograd.Function):
    """
    Sparsemax activation — a sparse alternative to softmax that projects onto the
    probability simplex and can produce exactly-zero outputs.

    Algorithm (Martins & Astudillo, 2016, Algorithm 1):
        1. Sort z in descending order to get z_sorted.
        2. Find support size k = max { k : 1 + k·z_k > Σ_{j≤k} z_j }.
        3. Compute threshold  τ(z) = (Σ_{j≤k} z_j − 1) / k.
        4. Return  sparsemax(z) = max(0, z − τ(z)).

    TabNet uses Sparsemax instead of softmax for feature selection masks so that
    unimportant features receive exactly zero weight (§3.1, Eq. 2).

    Reference:
        Martins & Astudillo (2016), "From Softmax to Sparsemax", ICML.
        Arik & Pfister (2019), §3.1 — Sparsemax applied to M[i].
    """

    @staticmethod
    def forward(ctx, input, dim=-1):
        ctx.dim = dim

        # Translate for numerical stability (subtract row-max, does not change argmax)
        input_max, _ = torch.max(input, dim=dim, keepdim=True)
        translated_input = input - input_max

        # Sort descending
        sorted_input, _ = torch.sort(translated_input, dim=dim, descending=True)

        # Cumulative sum of sorted values
        cumsum_sorted = torch.cumsum(sorted_input, dim=dim)

        # k-index tensor [1, 2, …, D]
        D = input.size(dim)
        k_indices = torch.arange(1, D + 1, device=input.device, dtype=input.dtype)
        shape = [1] * input.dim()
        shape[dim] = D
        k_indices = k_indices.view(*shape)

        # Support size condition: 1 + k·z_k > cumsum_k  (Algorithm 1, Step 2)
        condition = (1 + k_indices * sorted_input) > cumsum_sorted
        k_max, _ = torch.max(condition * k_indices, dim=dim, keepdim=True)

        # Threshold τ(z) = (cumsum_k − 1) / k  (Algorithm 1, Step 3)
        cumsum_k = torch.gather(cumsum_sorted, dim=dim, index=(k_max - 1).long())
        tau = (cumsum_k - 1.0) / k_max.to(input.dtype)

        # sparsemax(z) = max(0, z − τ(z))  (Algorithm 1, Step 4)
        output = torch.clamp(translated_input - tau, min=0.0)
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Gradient of sparsemax (Martins & Astudillo, 2016, Proposition 2):
            ∂L/∂z_j = δ_j · (∂L/∂s_j − (Σ_{i∈S} ∂L/∂s_i) / |S|)
        where S = support set {j : sparsemax(z)_j > 0} and δ_j = 1_{j∈S}.
        """
        output, = ctx.saved_tensors
        dim = ctx.dim

        non_zeros = output > 0  # Support indicator S(z)

        grad_sum = torch.sum(grad_output * non_zeros, dim=dim, keepdim=True)
        support_size = torch.clamp(
            torch.sum(non_zeros, dim=dim, keepdim=True).to(grad_output.dtype),
            min=1.0
        )
        grad_input = non_zeros * (grad_output - grad_sum / support_size)
        return grad_input, None


class Sparsemax(nn.Module):
    """
    nn.Module wrapper around SparsemaxFunction.

    Args:
        dim: Dimension along which sparsemax is applied (default: last dimension).
    """

    def __init__(self, dim=-1):
        super(Sparsemax, self).__init__()
        self.dim = dim

    def forward(self, input):
        return SparsemaxFunction.apply(input, self.dim)


class GhostBatchNorm1d(nn.Module):
    """
    Ghost Batch Normalization for 1-D feature tensors.

    Splits the training mini-batch into virtual sub-batches of size
    `virtual_batch_size` and applies standard Batch Normalization
    independently to each sub-batch, then concatenates the results.

    This mimics training with very small batches (which yield noisy, regularising
    BN statistics) even when the actual mini-batch is large, improving generalisation
    on tabular data.

    TabNet applies Ghost BN inside every GLU block in the Feature Transformer
    and uses standard BN (without splitting) in the Attentive Transformer (§3.1).

    Reference:
        Hoffer et al. (2017), "Train longer, generalize better".
        Arik & Pfister (2019), §3.1 — Ghost Batch Normalization, virtual batch size B_v.
    """

    def __init__(self, input_dim: int, virtual_batch_size: int = 128, momentum: float = 0.01):
        """
        Args:
            input_dim         : Feature dimension of the input tensor.
            virtual_batch_size: Size of each virtual sub-batch B_v (§3.1).
            momentum          : BN momentum for running statistics.
        """
        super(GhostBatchNorm1d, self).__init__()
        self.virtual_batch_size = virtual_batch_size
        self.bn = nn.BatchNorm1d(input_dim, momentum=momentum)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            batch_size = x.size(0)
            if batch_size <= self.virtual_batch_size:
                return self.bn(x)
            # Split into virtual sub-batches, normalise each, then recombine
            chunks = torch.chunk(x, max(1, batch_size // self.virtual_batch_size), dim=0)
            return torch.cat([self.bn(chunk) for chunk in chunks], dim=0)
        else:
            return self.bn(x)
