import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Any, Union
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer

class TabularDataset(Dataset):
    """
    PyTorch Dataset for tabular data.
    Wraps features and target tensors.
    """
    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Args:
            X: Input feature array of shape (N, D).
            y: Target array of shape (N,).
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


class TabularPreprocessor:
    """
    Preprocessing service for tabular datasets.
    Handles:
    - Automatically identifying numerical and categorical columns.
    - Imputing missing values.
    - Ordinal encoding of categorical columns (shifting values by +1 to reserve 0 for unknown/missing values).
    - Encoding labels for classification.
    - Determining categorical cardinalities (cat_dims) and suggesting embedding dimensions.
    """
    def __init__(
        self,
        cat_cols: List[str] = None,
        num_cols: List[str] = None,
        target_col: str = None,
        embedding_dim_factor: float = 1.0
    ):
        """
        Args:
            cat_cols: List of categorical column names. If None, will be inferred.
            num_cols: List of numerical column names. If None, will be inferred.
            target_col: Name of the target column.
            embedding_dim_factor: Factor to scale recommended embedding sizes.
        """
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.target_col = target_col
        self.embedding_dim_factor = embedding_dim_factor
        
        self.cat_imputer = SimpleImputer(strategy="most_frequent")
        self.num_imputer = SimpleImputer(strategy="median")
        
        # OrdinalEncoder mapping unseen/unknown items to -1
        self.ordinal_encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1
        )
        
        self.label_encoder = LabelEncoder()
        
        # Preprocessing metadata
        self.cat_idxs: List[int] = []
        self.cat_dims: List[int] = []
        self.cat_emb_dims: List[int] = []
        self.feature_names: List[str] = []
        self.is_fitted = False

    def fit(self, df: pd.DataFrame) -> "TabularPreprocessor":
        """
        Fit imputers and encoders on the input DataFrame.
        """
        df = df.copy()
        
        # Determine features and target
        if self.target_col and self.target_col in df.columns:
            X_df = df.drop(columns=[self.target_col])
            y_df = df[self.target_col]
        else:
            X_df = df
            y_df = None
            
        all_cols = list(X_df.columns)
        
        # Infer categorical and numerical columns if not specified
        if self.cat_cols is None or self.num_cols is None:
            inferred_cat = []
            inferred_num = []
            for col in all_cols:
                # Check data type
                if X_df[col].dtype == "object" or X_df[col].dtype == "category" or X_df[col].dtype == "bool":
                    inferred_cat.append(col)
                else:
                    # If numerical, check number of unique values
                    # Columns with very few unique values can sometimes be categorical, but we stick to dtype inference by default
                    inferred_num.append(col)
            
            if self.cat_cols is None:
                self.cat_cols = inferred_cat
            if self.num_cols is None:
                self.num_cols = inferred_num
                
        # Re-verify and maintain feature order
        self.feature_names = self.cat_cols + self.num_cols
        
        # Store index mappings for feature ordering (categorical first, then numerical)
        self.cat_idxs = list(range(len(self.cat_cols)))
        
        # Fit target label encoder if target is provided
        if y_df is not None:
            # Clean targets if string/categorical classes (e.g. drop periods, strip spaces)
            if not pd.api.types.is_numeric_dtype(y_df.dtype):
                y_clean = y_df.astype(str).str.strip().str.rstrip(".")
            else:
                y_clean = y_df
            self.label_encoder.fit(y_clean)
            
        # Fit categorical columns
        if len(self.cat_cols) > 0:
            cat_data = X_df[self.cat_cols].astype(str).values
            # Impute missing strings
            imputed_cat = self.cat_imputer.fit_transform(cat_data)
            # Fit encoder
            self.ordinal_encoder.fit(imputed_cat)
            
            # Determine dimensions and calculate embedding sizes
            self.cat_dims = []
            self.cat_emb_dims = []
            for i, col in enumerate(self.cat_cols):
                categories = self.ordinal_encoder.categories_[i]
                # Because unknown values encode as -1 and we shift by +1, we have:
                # 0 for unknown/missing
                # 1 to len(categories) for encoded values
                cardinality = len(categories) + 1
                self.cat_dims.append(cardinality)
                
                # Embedding dimension rule of thumb: min(16, (cardinality + 1) // 2)
                # Ensure it is at least 1
                emb_dim = max(1, int(min(16, (cardinality + 1) // 2) * self.embedding_dim_factor))
                self.cat_emb_dims.append(emb_dim)
                
        # Fit numerical columns
        if len(self.num_cols) > 0:
            num_data = X_df[self.num_cols].values
            self.num_imputer.fit(num_data)
            
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, Union[np.ndarray, None]]:
        """
        Transform the input DataFrame using fitted estimators.
        Returns:
            X_processed: Numpy array of shape (N, D_processed) with categorical columns first.
            y_processed: Target array or None.
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transform can be called.")
            
        df = df.copy()
        
        # 1. Process target
        y_proc = None
        if self.target_col and self.target_col in df.columns:
            y_data = df[self.target_col]
            if not pd.api.types.is_numeric_dtype(y_data.dtype):
                y_clean = y_data.astype(str).str.strip().str.rstrip(".")
            else:
                y_clean = y_data
            y_proc = self.label_encoder.transform(y_clean)
            
        # 2. Process features
        # Categorical columns
        if len(self.cat_cols) > 0:
            cat_data = df[self.cat_cols].astype(str).values
            imputed_cat = self.cat_imputer.transform(cat_data)
            encoded_cat = self.ordinal_encoder.transform(imputed_cat)
            # Shift by +1: mapping unknown/missing (-1) to index 0, and categories to 1..C
            encoded_cat = encoded_cat + 1
        else:
            encoded_cat = np.empty((len(df), 0))
            
        # Numerical columns
        if len(self.num_cols) > 0:
            num_data = df[self.num_cols].values
            imputed_num = self.num_imputer.transform(num_data)
        else:
            imputed_num = np.empty((len(df), 0))
            
        # Concatenate: categorical features first, then numerical features
        X_proc = np.hstack([encoded_cat, imputed_num])
        
        return X_proc, y_proc


def get_data_loaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int = 256,
    val_batch_size: int = 512,
    num_workers: int = 0,
    pin_memory: bool = False
) -> Tuple[DataLoader, DataLoader]:
    """
    Create PyTorch DataLoader instances for training and validation datasets.
    """
    train_dataset = TabularDataset(X_train, y_train)
    val_dataset = TabularDataset(X_val, y_val)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    return train_loader, val_loader
