import os
import joblib
import pandas as pd
import torch
from tabnet import TabNetClassifier

def main():
    print("=== TabNet Studio: Testing Model on Different Dataset (Covertype) ===")
    
    # Paths
    covertype_path = "data/raw/covertype/covertype.csv"
    model_path = "artifacts/checkpoints/logistic_regression.joblib"
    tabnet_path = "artifacts/checkpoints/benchmark_tabnet_run_best_model.pt"
    
    # 1. Load the new Covertype dataset
    print(f"\n[1/3] Loading Covertype dataset from {covertype_path}...")
    if not os.path.exists(covertype_path):
        print(f"Error: Dataset not found at {covertype_path}")
        return
    df_cover = pd.read_csv(covertype_path)
    print(f"Covertype dataset shape: {df_cover.shape} (54 features + target)")
    
    # Separate features and target
    X_cover = df_cover.drop(columns=["Cover_Type"])
    
    # 2. Try predicting with the Scikit-learn Logistic Regression Model
    print(f"\n[2/3] Attempting to load and test Logistic Regression model (trained on Adult)...")
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        try:
            print("Running predictions on Covertype features...")
            preds = model.predict(X_cover)
            print("Successfully predicted!")
        except Exception as e:
            print(f"Expected Error Encountered: {type(e).__name__}")
            print(f"Details: {str(e)}")
    else:
        print(f"Model file not found at {model_path}")
        
    # 3. Try predicting with the TabNet Model
    print(f"\n[3/3] Attempting to load and test TabNet model (trained on Adult)...")
    if os.path.exists(tabnet_path):
        try:
            print("Loading PyTorch TabNet checkpont...")
            checkpoint = torch.load(tabnet_path, map_location="cpu")
            
            # Adult TabNet expects input dimension 57 (after embedding)
            # Let's instantiate the same model class
            # Since it's trained on Adult, it has 14 features (with 8 categorical embeddings)
            print("Attempting to run forward pass with Covertype features...")
            # We convert Covertype to a tensor of shape (batch_size, 54)
            X_tensor = torch.tensor(X_cover.head(5).values, dtype=torch.float32)
            
            # The classification head output class count is 2 (binary)
            # Let's try passing the tensor of size 54 into a model trained for size 14 (and 57 after embeddings)
            # Since we did not train a Covertype TabNet yet, we use the Adult model state
            # Let's see how the embedding layer complains about input features count
            # Adult preprocessor has 14 features, so model expects input shape (B, 14)
            # Let's pass the first 5 samples of Covertype (shape: (5, 54)) into model.embeddings
            # Adult TabNet expects 14 features. Let's see what happens.
            
            # We recreate the classifier skeleton that matches the checkpoint
            # (which has input_dim=14, cat_idxs=[0..7], cat_dims=[10, 17, 8, 16, 7, 6, 3, 43], cat_emb_dims=[5, 9, 4, 8, 4, 3, 2, 16])
            model_tn = TabNetClassifier(
                num_features=14,
                num_classes=2,
                cat_idxs=[0, 1, 2, 3, 4, 5, 6, 7],
                cat_dims=[10, 17, 8, 16, 7, 6, 3, 43],
                cat_emb_dims=[5, 9, 4, 8, 4, 3, 2, 16],
                n_d=16,
                n_a=16,
                n_steps=5
            )
            model_tn.load_state_dict(checkpoint["model_state_dict"])
            model_tn.eval()
            
            # Passing X_tensor which has 54 features into model_tn (expecting 14 features)
            logits, _, _ = model_tn(X_tensor)
        except Exception as e:
            print(f"Expected Error Encountered: {type(e).__name__}")
            print(f"Details: {str(e)}")
    else:
        print(f"TabNet model file not found at {tabnet_path}")

if __name__ == "__main__":
    main()
