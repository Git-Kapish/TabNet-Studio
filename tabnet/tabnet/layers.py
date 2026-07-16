import torch
import torch.nn as nn

class SparsemaxFunction(torch.autograd.Function):
    """
    An efficient implementation of Sparsemax activation function.
    Based on "From Softmax to Sparsemax: A Sparse Model of Attention and Multi-Label Classification"
    (Martins & Astudillo, 2016).
    """
    @staticmethod
    def forward(ctx, input, dim=-1):
        """
        Forward pass for Sparsemax.
        Args:
            input: Input tensor of arbitrary shape.
            dim: Dimension along which to apply Sparsemax.
        """
        ctx.dim = dim
        
        # Translate input by max along the dimension for numerical stability
        input_max, _ = torch.max(input, dim=dim, keepdim=True)
        translated_input = input - input_max
        
        # Sort in descending order
        sorted_input, sorted_indices = torch.sort(translated_input, dim=dim, descending=True)
        
        # Calculate cumulative sums
        cumsum_sorted = torch.cumsum(sorted_input, dim=dim)
        
        # Create k tensor: [1, 2, ..., D]
        D = input.size(dim)
        # Create range tensor with same device and dtype as input
        k_indices = torch.arange(1, D + 1, device=input.device, dtype=input.dtype)
        # Reshape k_indices to broadcast correctly with cumsum_sorted
        shape = [1] * input.dim()
        shape[dim] = D
        k_indices = k_indices.view(*shape)
        
        # Find k(z) = max { k | 1 + k * z_k > sum(z_r for r=1..k) }
        condition = (1 + k_indices * sorted_input) > cumsum_sorted
        
        # We find the threshold tau by identifying the last index where condition is true.
        indices_mask = condition * k_indices
        k_max, _ = torch.max(indices_mask, dim=dim, keepdim=True)
        
        # Gather the cumulative sum at k_max
        gather_index = (k_max - 1).long()
        cumsum_k = torch.gather(cumsum_sorted, dim=dim, index=gather_index)
        
        # Calculate threshold tau(z) = (cumsum_k - 1) / k_max
        tau = (cumsum_k - 1.0) / k_max.to(input.dtype)
        
        # Compute forward output: max(0, translated_input - tau)
        output = torch.clamp(translated_input - tau, min=0.0)
        
        # Save output for backward pass (to know the support)
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for Sparsemax.
        """
        output, = ctx.saved_tensors
        dim = ctx.dim
        
        # Support indicator S(z): where output > 0
        non_zeros = output > 0
        
        # Compute grad_input
        # We sum grad_output across the support, and divide by the size of the support.
        grad_sum = torch.sum(grad_output * non_zeros, dim=dim, keepdim=True)
        support_size = torch.sum(non_zeros, dim=dim, keepdim=True).to(grad_output.dtype)
        
        # To avoid division by zero (support_size should always be >= 1 for valid inputs)
        support_size = torch.clamp(support_size, min=1.0)
        
        grad_input = non_zeros * (grad_output - (grad_sum / support_size))
        return grad_input, None


class Sparsemax(nn.Module):
    """
    Sparsemax activation module.
    """
    def __init__(self, dim=-1):
        super(Sparsemax, self).__init__()
        self.dim = dim

    def forward(self, input):
        return SparsemaxFunction.apply(input, self.dim)


class GhostBatchNorm1d(nn.Module):
    """
    Ghost Batch Normalization 1D.
    Splits batch into virtual batches and applies Batch Normalization.
    """
    def __init__(self, input_dim, virtual_batch_size=128, momentum=0.01):
        super(GhostBatchNorm1d, self).__init__()
        self.input_dim = input_dim
        self.virtual_batch_size = virtual_batch_size
        self.bn = nn.BatchNorm1d(self.input_dim, momentum=momentum)

    def forward(self, x):
        if self.training:
            batch_size = x.size(0)
            if batch_size <= self.virtual_batch_size:
                return self.bn(x)
            
            chunks = torch.chunk(x, max(1, batch_size // self.virtual_batch_size), dim=0)
            res = [self.bn(x_chunk) for x_chunk in chunks]
            return torch.cat(res, dim=0)
        else:
            return self.bn(x)
