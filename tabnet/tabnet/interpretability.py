import torch
from typing import List


def compute_local_feature_importance(
    step_masks: List[torch.Tensor],
    decision_outputs: List[torch.Tensor]
) -> torch.Tensor:
    """
    Compute instance-level (local) feature importance.

    Implements the feature attribution formula from §3.3 of the paper.
    At each step i the decision representation d[i] ∈ ℝ^{B×N_d} captures
    "how much" information was extracted.  The step contribution coefficient
    η_b[i] summarises this as a scalar per sample:

        η_b[i] = Σ_c ReLU(d_{b,c}[i])               (step contribution)

    The aggregated attribution for feature j in sample b is then:

        M_agg[b, j] = Σ_i η_b[i] · M_{b,j}[i]       (Eq. 8, unnormalised)

    Finally normalise so each sample's importances sum to 1:

        M_agg[b, j] ← M_agg[b, j] / Σ_j M_agg[b, j]

    Reference: Arik & Pfister (2019), §3.3 "Interpretability", Eq. 8.

    Args:
        step_masks      : List of N_steps sparse masks M[i], each (B, input_dim).
        decision_outputs: List of N_steps decision tensors d[i], each (B, N_d).

    Returns:
        M_agg : Normalised local feature importance, shape (B, input_dim).
                Each row sums to 1.
    """
    if len(step_masks) == 0 or len(decision_outputs) == 0:
        raise ValueError("step_masks and decision_outputs must be non-empty.")

    batch_size, input_dim = step_masks[0].shape
    device = step_masks[0].device

    total_contribution = torch.zeros((batch_size, input_dim), device=device)

    for mask, d in zip(step_masks, decision_outputs):
        # η_b[i] = Σ_c ReLU(d_{b,c}[i])  — shape (B, 1)
        eta = torch.sum(torch.relu(d), dim=-1, keepdim=True)
        # Weighted mask for this step (Eq. 8)
        total_contribution = total_contribution + eta * mask

    # Normalise across features so importances sum to 1 per sample
    sum_contribution = torch.clamp(
        torch.sum(total_contribution, dim=-1, keepdim=True), min=1e-10
    )
    return total_contribution / sum_contribution


def compute_global_feature_importance(
    step_masks: List[torch.Tensor],
    decision_outputs: List[torch.Tensor]
) -> torch.Tensor:
    """
    Compute dataset-level (global) feature importances.

    Averages the local feature importances (Eq. 8) across all samples in
    the provided batch, then re-normalises so the result sums to 1.

    Reference: Arik & Pfister (2019), §3.3 "Interpretability".

    Args:
        step_masks      : List of N_steps sparse masks M[i], each (B, input_dim).
        decision_outputs: List of N_steps decision tensors d[i], each (B, N_d).

    Returns:
        Global importance vector of shape (input_dim,), summing to 1.
    """
    M_agg = compute_local_feature_importance(step_masks, decision_outputs)
    global_importance = torch.mean(M_agg, dim=0)
    global_importance = global_importance / torch.clamp(global_importance.sum(), min=1e-10)
    return global_importance


def get_attention_masks(step_masks: List[torch.Tensor]) -> torch.Tensor:
    """
    Stack all step selection masks into a single tensor.

    Args:
        step_masks: List of N_steps masks, each of shape (B, input_dim).

    Returns:
        Stacked mask tensor of shape (N_steps, B, input_dim).
    """
    return torch.stack(step_masks, dim=0)
