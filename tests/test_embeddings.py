import torch
from tabnet.embeddings import TabularEmbedding

def test_tabular_embedding_dims():
    """
    Test tabular embedding output dimensions, input batch normalization,
    and concatenation properties.
    """
    num_features = 6
    cat_idxs = [0, 3, 5]
    cat_dims = [3, 10, 2] # Categorical cardinalities
    cat_emb_dims = [2, 5, 1] # Target embedding dimensions
    
    # post_embed_dim should be: sum(cat_emb_dims) + len(num_cols)
    # numerical columns are [1, 2, 4], so len(num_cols) = 3
    # post_embed_dim = (2 + 5 + 1) + 3 = 11
    expected_dim = 11
    
    tabular_emb = TabularEmbedding(
        num_features=num_features,
        cat_idxs=cat_idxs,
        cat_dims=cat_dims,
        cat_emb_dims=cat_emb_dims
    )
    
    assert tabular_emb.post_embed_dim == expected_dim
    assert len(tabular_emb.numerical_idxs) == 3
    
    # Create random batch of size 4
    x = torch.randn(4, num_features)
    # Ensure categorical columns have integer indices in valid range
    x[:, 0] = torch.randint(0, 3, (4,)).float()
    x[:, 3] = torch.randint(0, 10, (4,)).float()
    x[:, 5] = torch.randint(0, 2, (4,)).float()
    
    out = tabular_emb(x)
    assert out.shape == (4, expected_dim)
