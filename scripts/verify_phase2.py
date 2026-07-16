import os
import json
import shutil
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
from tabnet import (
    TabNetClassifier,
    Trainer,
    TabularPreprocessor,
    get_data_loaders,
    compute_local_feature_importance,
    compute_global_feature_importance
)

def main():
    print("=== TabNet Studio: Verification Script for Phase 2 ===")
    
    # 1. Load and preprocess the real Adult Census Income dataset
    dataset_path = "data/raw/adult/adult.csv"
    print(f"\n[1/5] Loading dataset from {dataset_path}...")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}. Please make sure you have run the download script.")
        
    df = pd.read_csv(dataset_path)
    print(f"Dataset shape: {df.shape}")
    print(df.head())
    
    # Define features based on Adult Census Schema
    cat_cols = [
        "workclass", "education", "marital-status", "occupation",
        "relationship", "race", "sex", "native-country"
    ]
    num_cols = [
        "age", "fnlwgt", "education-num", "capital-gain",
        "capital-loss", "hours-per-week"
    ]
    
    preprocessor = TabularPreprocessor(
        cat_cols=cat_cols,
        num_cols=num_cols,
        target_col="income"
    )
    
    # Fit and transform preprocessor on the dataset
    preprocessor.fit(df)
    X, y = preprocessor.transform(df)
    print(f"Preprocessor fit: Categorical indices={preprocessor.cat_idxs}")
    print(f"Categorical cardinalities={preprocessor.cat_dims}")
    print(f"Recommended embedding dimensions={preprocessor.cat_emb_dims}")
    print(f"Processed features shape: {X.shape}, labels shape: {y.shape}")
    
    # Split into train/validation sets (80/20)
    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    train_loader, val_loader = get_data_loaders(
        X_train, y_train, X_val, y_val,
        batch_size=1024, val_batch_size=2048 # Using larger batch sizes for faster CPU training
    )
    
    # 2. Initialize Model
    print("\n[2/5] Initializing TabNetClassifier...")
    model = TabNetClassifier(
        num_features=X.shape[1],
        num_classes=2,
        cat_idxs=preprocessor.cat_idxs,
        cat_dims=preprocessor.cat_dims,
        cat_emb_dims=preprocessor.cat_emb_dims,
        n_d=16,
        n_a=16,
        n_steps=5,
        gamma=1.5,
        virtual_batch_size=128
    )
    print(model)
    
    # 3. Setup Trainer
    print("\n[3/5] Setting up Trainer...")
    optimizer = optim.Adam(model.parameters(), lr=0.02, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)
    
    # Cleanup any old test runs
    shutil.rmtree("artifacts/checkpoints_test", ignore_errors=True)
    shutil.rmtree("artifacts/tensorboard_test", ignore_errors=True)
    shutil.rmtree("artifacts/runs_test", ignore_errors=True)
    
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device="cpu",
        checkpoint_dir="artifacts/checkpoints_test",
        tensorboard_dir="artifacts/tensorboard_test",
        runs_dir="artifacts/runs_test",
        patience=3,
        clip_value=2.0,
        seed=42,
        deterministic=True
    )
    
    # 4. Train Model
    print("\n[4/5] Running model fit loop for 5 epochs on real dataset...")
    run_name = "adult_test_run"
    run_metadata = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        max_epochs=5,
        lambda_sparse=1e-4,
        run_name=run_name
    )
    
    # 5. Verify Output Artifacts & Interpretability
    print("\n[5/5] Verifying generated artifacts & interpretability outputs...")
    
    checkpoint_file = f"artifacts/checkpoints_test/{run_name}_best_model.pt"
    run_file = f"artifacts/runs_test/{run_name}_run.json"
    
    print(f"Checkpoint saved: {os.path.exists(checkpoint_file)} (size: {os.path.getsize(checkpoint_file) if os.path.exists(checkpoint_file) else 0} bytes)")
    print(f"Metadata file saved: {os.path.exists(run_file)}")
    
    # Print the saved metrics
    if os.path.exists(run_file):
        with open(run_file, "r") as f:
            meta = json.load(f)
            print("\nSaved Metadata Summary:")
            print(f"  - Accuracy: {meta['evaluation_metrics']['accuracy']:.4f}")
            print(f"  - F1 Score: {meta['evaluation_metrics']['f1']:.4f}")
            print(f"  - Checkpoint Location: {meta['checkpoint_location']}")
            print(f"  - TensorBoard Location: {meta['tensorboard_log_location']}")
            
    # Load model and run batch inference to extract importances
    model.eval()
    with torch.no_grad():
        X_batch_tensor = torch.tensor(X_val[:5], dtype=torch.float32)
        logits, step_masks, decision_outputs = model(X_batch_tensor)
        
        # Calculate local/global importances
        local_imp = compute_local_feature_importance(step_masks, decision_outputs)
        global_imp = compute_global_feature_importance(step_masks, decision_outputs)
        
        print("\nInterpretability test on first validation batch sample:")
        print(f"  - Feature importances summing to 1.0 (local):\n    {local_imp[0].numpy()} (Sum: {torch.sum(local_imp[0]).item():.2f})")
        print(f"  - Feature names: {preprocessor.feature_names}")
        print(f"  - Global dataset-level importances:\n    {global_imp.numpy()} (Sum: {torch.sum(global_imp).item():.2f})")
        
    print("\n=== Phase 2 Verification Complete on Real Dataset! ===")

if __name__ == "__main__":
    main()
