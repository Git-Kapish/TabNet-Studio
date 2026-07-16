import torch
import torch.nn as nn
from typing import List

class TabularEmbedding(nn.Module):
    """
    Tabular feature embedding layer.
    Categorical features are mapped to trainable embeddings, concatenated with numerical features,
    and normalized using standard Input Batch Normalization.
    """
    def __init__(
        self,
        num_features: int,
        cat_idxs: List[int],
        cat_dims: List[int],
        cat_emb_dims: List[int]
    ):
        """
        Args:
            num_features: Total number of features in the input tabular data.
            cat_idxs: List of indices of categorical features.
            cat_dims: List of cardinialities (num of unique categories) for each categorical feature.
            cat_emb_dims: List of embedding dimensions for each categorical feature.
        """
        super(TabularEmbedding, self).__init__()
        
        self.num_features = num_features
        self.cat_idxs = cat_idxs
        self.cat_dims = cat_dims
        self.cat_emb_dims = cat_emb_dims
        
        # Determine numerical indices
        self.numerical_idxs = [i for i in range(num_features) if i not in cat_idxs]
        
        # Trainable embedding layers for categorical features
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=dim, embedding_dim=emb_dim)
            for dim, emb_dim in zip(cat_dims, cat_emb_dims)
        ])
        
        # Calculate post-embedding dimension
        self.post_embed_dim = sum(cat_emb_dims) + len(self.numerical_idxs)
        
        # Input Batch Normalization (applied to all concatenated features)
        # Using standard nn.BatchNorm1d with default momentum
        self.input_bn = nn.BatchNorm1d(self.post_embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, num_features)
        Returns:
            Embedded and batch normalized tensor of shape (batch_size, post_embed_dim)
        """
        # Separate numerical columns
        if len(self.numerical_idxs) > 0:
            x_num = x[:, self.numerical_idxs]
        else:
            x_num = torch.empty((x.size(0), 0), device=x.device, dtype=x.dtype)
            
        # Process and embed categorical columns
        x_cats = []
        for i, cat_idx in enumerate(self.cat_idxs):
            # Categorical index must be integer type
            cat_col = x[:, cat_idx].long()
            emb = self.embeddings[i](cat_col)
            x_cats.append(emb)
            
        # Concatenate embeddings and numerical columns
        if len(x_cats) > 0:
            x_cat_concat = torch.cat(x_cats, dim=-1)
            if x_num.size(1) > 0:
                x_all = torch.cat([x_cat_concat, x_num], dim=-1)
            else:
                x_all = x_cat_concat
        else:
            x_all = x_num
            
        # Apply Input BatchNorm (standard BatchNorm1d, NOT Ghost BatchNorm)
        return self.input_bn(x_all)
