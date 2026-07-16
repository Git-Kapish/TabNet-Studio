import torch
from typing import List, Tuple

def compute_local_feature_importance(
    step_masks: List[torch.Tensor],
    decision_outputs: List[torch.Tensor]
) -> torch.Tensor:
    """
    Compute instance-wise (local) feature importance masks.
    Formula: M_agg[b, j] = sum_i(eta_b[i] * M_b,j[i]) / sum_j(sum_i(eta_b[i] * M_b,j[i]))
    where eta_b[i] = sum_c(ReLU(d_{b,c}[i]))
    
    Args:
        step_masks: List of feature selection masks for each step,
                     each of shape (batch_size, input_dim).
        decision_outputs: List of decision representations for each step,
                           each of shape (batch_size, n_d).
    Returns:
        M_agg: Normalised local feature importance tensor of shape (batch_size, input_dim).
    """
    if len(step_masks) == 0 or len(decision_outputs) == 0:
        raise ValueError("Step masks and decision outputs lists cannot be empty.")
        
    batch_size, input_dim = step_masks[0].shape
    device = step_masks[0].device
    
    # Initialize total feature contribution: shape (batch_size, input_dim)
    total_contribution = torch.zeros((batch_size, input_dim), device=device)
    
    for mask, d in zip(step_masks, decision_outputs):
        # Step contribution coefficient: eta_b[i] = sum_c(ReLU(d_{b,c}[i]))
        # shape (batch_size, 1)
        eta = torch.sum(torch.relu(d), dim=-1, keepdim=True)
        
        # Weighted mask at step i: shape (batch_size, input_dim)
        total_contribution += eta * mask
        
    # Normalise contribution across features for each sample to sum to 1
    # Add small epsilon to avoid division by zero
    sum_contribution = torch.sum(total_contribution, dim=-1, keepdim=True)
    sum_contribution = torch.clamp(sum_contribution, min=1e-10)
    
    M_agg = total_contribution / sum_contribution
    return M_agg


def compute_global_feature_importance(
    step_masks: List[torch.Tensor],
    decision_outputs: List[torch.Tensor]
) -> torch.Tensor:
    """
    Compute dataset-level (global) feature importances by averaging local feature importances
    across the entire batch/dataset.
    
    Args:
        step_masks: List of feature selection masks for each step.
        decision_outputs: List of decision representations for each step.
    Returns:
        Global importance vector of shape (input_dim,) summing to 1.
    """
    # Get local feature importances: shape (batch_size, input_dim)
    M_agg = compute_local_feature_importance(step_masks, decision_outputs)
    
    # Average across batch dimension
    global_importance = torch.mean(M_agg, dim=0)
    
    # Re-normalize to sum to 1
    global_importance = global_importance / torch.clamp(torch.sum(global_importance), min=1e-10)
    
    return global_importance


def get_attention_masks(step_masks: List[torch.Tensor]) -> torch.Tensor:
    """
    Stack selection masks for all decision steps.
    
    Args:
        step_masks: List of step selection masks.
    Returns:
        Stacked attention masks tensor of shape (n_steps, batch_size, input_dim).
    """
    return torch.stack(step_masks, dim=0)
