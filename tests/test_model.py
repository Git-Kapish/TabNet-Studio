import torch
from tabnet.model import TabNetClassifier
from tabnet.interpretability import (
    compute_local_feature_importance,
    compute_global_feature_importance,
    get_attention_masks
)

def test_model_forward_backward():
    """
    Test forward and backward passes of TabNetClassifier.
    """
    batch_size = 8
    num_features = 5
    num_classes = 3
    
    # 2 categorical columns, 3 numerical columns
    cat_idxs = [1, 4]
    cat_dims = [4, 6]
    cat_emb_dims = [2, 3]
    
    model = TabNetClassifier(
        num_features=num_features,
        num_classes=num_classes,
        cat_idxs=cat_idxs,
        cat_dims=cat_dims,
        cat_emb_dims=cat_emb_dims,
        n_d=4,
        n_a=4,
        n_steps=3,
        gamma=1.5,
        virtual_batch_size=2
    )
    
    # Generate mock inputs
    x = torch.randn(batch_size, num_features)
    x[:, 1] = torch.randint(0, 4, (batch_size,)).float()
    x[:, 4] = torch.randint(0, 6, (batch_size,)).float()
    
    logits, step_masks, decision_outputs = model(x)
    
    # Check shapes
    assert logits.shape == (batch_size, num_classes)
    assert len(step_masks) == 3
    assert len(decision_outputs) == 3
    
    # Assert each mask dimension matches the post-embedding dimension
    # post_embedding = sum(cat_emb_dims) + len(num_cols) = (2 + 3) + 3 = 8
    post_embed_dim = model.embeddings.post_embed_dim
    assert post_embed_dim == 8
    
    for mask in step_masks:
        assert mask.shape == (batch_size, post_embed_dim)
        
    for d in decision_outputs:
        assert d.shape == (batch_size, 4)
        
    # Check backward pass
    loss = torch.sum(logits)
    loss.backward()
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None


def test_interpretability():
    """
    Test feature importance calculations.
    """
    batch_size = 4
    post_embed_dim = 6
    n_steps = 3
    n_d = 4
    
    # Mock step masks: list of 3 tensors of shape (4, 6)
    step_masks = [torch.rand(batch_size, post_embed_dim) for _ in range(n_steps)]
    # Mock decision outputs: list of 3 tensors of shape (4, 4)
    decision_outputs = [torch.rand(batch_size, n_d) for _ in range(n_steps)]
    
    # Local feature importance
    M_agg = compute_local_feature_importance(step_masks, decision_outputs)
    assert M_agg.shape == (batch_size, post_embed_dim)
    
    # Verify local importance sums to 1.0 for each sample
    assert torch.allclose(torch.sum(M_agg, dim=-1), torch.ones(batch_size))
    
    # Global feature importance
    global_importance = compute_global_feature_importance(step_masks, decision_outputs)
    assert global_importance.shape == (post_embed_dim,)
    assert torch.allclose(torch.sum(global_importance), torch.tensor(1.0))
    
    # Stacked masks
    stacked_masks = get_attention_masks(step_masks)
    assert stacked_masks.shape == (n_steps, batch_size, post_embed_dim)
