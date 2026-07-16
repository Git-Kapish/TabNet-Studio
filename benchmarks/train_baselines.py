import os
import json
import time
import yaml
import torch
import torch.optim as optim
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# Try to import XGBoost, fallback if not installed
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from tabnet import (
    TabNetClassifier,
    Trainer,
    TabularPreprocessor,
    get_data_loaders,
    compute_classification_metrics
)

# Set directories
DATASET_PATH = "data/raw/adult/adult.csv"
CONFIG_PATH = "configs/adult_income.yaml"
ARTIFACTS_DIR = "artifacts"
CHECKPOINTS_DIR = "artifacts/checkpoints"
RUNS_DIR = "artifacts/runs"
BENCHMARKS_DIR = "benchmarks"
RESULTS_PATH = "benchmarks/results.json"

def load_and_split_data() -> Tuple[pd.DataFrame, pd.DataFrame, str, list, list]:
    """
    Load raw Adult dataset and split into train/validation sets (80/20).
    """
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}. Please run the downloader first.")
        
    df = pd.read_csv(DATASET_PATH)
    
    # Define columns
    cat_cols = [
        "workclass", "education", "marital-status", "occupation",
        "relationship", "race", "sex", "native-country"
    ]
    num_cols = [
        "age", "fnlwgt", "education-num", "capital-gain",
        "capital-loss", "hours-per-week"
    ]
    target_col = "income"
    
    # Clean target
    df[target_col] = df[target_col].astype(str).str.strip().str.rstrip(".")
    
    # Split
    split_idx = int(0.8 * len(df))
    train_df = df.iloc[:split_idx].copy()
    val_df = df.iloc[split_idx:].copy()
    
    return train_df, val_df, target_col, cat_cols, num_cols

def evaluate_scikit_model(
    model: Any,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    model_name: str
) -> Dict[str, Any]:
    """
    Train and evaluate a scikit-learn pipeline/model.
    Tracks training time, inference time, evaluation metrics, and joblib file size.
    """
    print(f"Training {model_name}...")
    
    # Measure training time
    start_train = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_train
    
    # Measure inference time
    start_inf = time.time()
    preds = model.predict(X_val)
    inf_time = time.time() - start_inf
    
    # Calculate metrics
    metrics = compute_classification_metrics(y_val, preds)
    
    # Save checkpoint to get file size
    checkpoint_path = os.path.join(CHECKPOINTS_DIR, f"{model_name.lower().replace(' ', '_')}.joblib")
    joblib.dump(model, checkpoint_path)
    model_size = os.path.getsize(checkpoint_path)
    
    print(f"  - Training Time: {train_time:.2f}s")
    print(f"  - Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1']:.4f}")
    print(f"  - Model Size: {model_size / 1024:.1f} KB")
    
    return {
        "model_name": model_name,
        "metrics": metrics,
        "training_time_seconds": train_time,
        "inference_time_seconds": inf_time,
        "model_size_bytes": model_size,
        "checkpoint_path": os.path.abspath(checkpoint_path)
    }

def main():
    print("=== TabNet Studio: Baseline Benchmarking ===")
    
    # Ensure directories exist
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    os.makedirs(RUNS_DIR, exist_ok=True)
    os.makedirs(BENCHMARKS_DIR, exist_ok=True)
    
    # 1. Load Data
    train_df, val_df, target_col, cat_cols, num_cols = load_and_split_data()
    
    y_train = train_df[target_col].values
    y_val = val_df[target_col].values
    
    # Encode target labels
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_val_enc = le.transform(y_val)
    
    X_train_df = train_df.drop(columns=[target_col])
    X_val_df = val_df.drop(columns=[target_col])
    
    results = {}
    
    # ==========================================
    # Baseline 1: Logistic Regression
    # ==========================================
    lr_preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]), num_cols),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ]), cat_cols)
        ]
    )
    lr_pipeline = Pipeline([
        ("preprocessor", lr_preprocessor),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42))
    ])
    
    results["logistic_regression"] = evaluate_scikit_model(
        lr_pipeline, X_train_df, y_train_enc, X_val_df, y_val_enc, "Logistic Regression"
    )
    
    # ==========================================
    # Baseline 2: Random Forest
    # ==========================================
    rf_preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), num_cols),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
            ]), cat_cols)
        ]
    )
    rf_pipeline = Pipeline([
        ("preprocessor", rf_preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
    ])
    
    results["random_forest"] = evaluate_scikit_model(
        rf_pipeline, X_train_df, y_train_enc, X_val_df, y_val_enc, "Random Forest"
    )
    
    # ==========================================
    # Baseline 3: XGBoost
    # ==========================================
    if XGBOOST_AVAILABLE:
        xgb_pipeline = Pipeline([
            ("preprocessor", rf_preprocessor),  # Re-use ordinal preprocessor
            ("classifier", XGBClassifier(n_estimators=100, random_state=42, eval_metric="logloss", n_jobs=-1))
        ])
        results["xgboost"] = evaluate_scikit_model(
            xgb_pipeline, X_train_df, y_train_enc, X_val_df, y_val_enc, "XGBoost"
        )
    else:
        print("\n[Warning] XGBoost is not installed. Skipping XGBoost baseline.")
        results["xgboost"] = {
            "model_name": "XGBoost",
            "metrics": {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0},
            "training_time_seconds": 0.0,
            "inference_time_seconds": 0.0,
            "model_size_bytes": 0,
            "checkpoint_path": ""
        }
        
    # ==========================================
    # Model 4: TabNet
    # ==========================================
    print("\nTraining TabNet...")
    # Load config file
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
        
    # Preprocess using TabNet Preprocessor
    tabnet_prep = TabularPreprocessor(
        cat_cols=cat_cols,
        num_cols=num_cols,
        target_col=target_col
    )
    tabnet_prep.fit(train_df)
    X_train_tn, y_train_tn = tabnet_prep.transform(train_df)
    X_val_tn, y_val_tn = tabnet_prep.transform(val_df)
    
    train_loader, val_loader = get_data_loaders(
        X_train_tn, y_train_tn, X_val_tn, y_val_tn,
        batch_size=config["training"]["batch_size"],
        val_batch_size=config["training"]["batch_size"] * 2
    )
    
    # Initialize TabNetClassifier
    tabnet_model = TabNetClassifier(
        num_features=X_train_tn.shape[1],
        num_classes=len(np.unique(y_train_tn)),
        cat_idxs=tabnet_prep.cat_idxs,
        cat_dims=tabnet_prep.cat_dims,
        cat_emb_dims=tabnet_prep.cat_emb_dims,
        n_d=config["model"]["n_d"],
        n_a=config["model"]["n_a"],
        n_steps=config["model"]["n_steps"],
        gamma=config["model"]["gamma"],
        n_shared=config["model"]["n_shared"],
        n_dependent=config["model"]["n_dependent"],
        virtual_batch_size=config["model"]["virtual_batch_size"],
        momentum=config["model"]["momentum"]
    )
    
    optimizer = optim.Adam(
        tabnet_model.parameters(), 
        lr=config["training"]["lr"], 
        weight_decay=1e-5
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
    
    trainer = Trainer(
        model=tabnet_model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=config["training"].get("device", "cpu"),
        checkpoint_dir=CHECKPOINTS_DIR,
        tensorboard_dir=os.path.join(ARTIFACTS_DIR, "tensorboard"),
        runs_dir=RUNS_DIR,
        patience=config["training"]["patience"],
        clip_value=config["training"]["clip_value"],
        seed=config["training"]["seed"],
        deterministic=config["training"]["deterministic"]
    )
    
    # Train TabNet (limit to 10 epochs for benchmarking speed, but configurable)
    # The config specifies max_epochs: 100, let's train for 15 epochs to get decent results fast
    epochs = 15
    print(f"Fitting TabNet for {epochs} epochs...")
    run_name = "benchmark_tabnet_run"
    run_metadata = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        max_epochs=epochs,
        lambda_sparse=config["training"]["lambda_sparse"],
        run_name=run_name
    )
    
    # Measure inference time of TabNet
    start_tn_inf = time.time()
    val_loss, val_metrics = trainer.evaluate(val_loader)
    tn_inf_time = time.time() - start_tn_inf
    
    # Find TabNet model size
    best_checkpoint = os.path.join(CHECKPOINTS_DIR, f"{run_name}_best_model.pt")
    tn_size = os.path.getsize(best_checkpoint) if os.path.exists(best_checkpoint) else 0
    
    results["tabnet"] = {
        "model_name": "TabNet",
        "metrics": val_metrics,
        "training_time_seconds": run_metadata["training_duration_seconds"],
        "inference_time_seconds": tn_inf_time,
        "model_size_bytes": tn_size,
        "checkpoint_path": os.path.abspath(best_checkpoint)
    }
    
    # Save results.json
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"\n[OK] Benchmarks compiled successfully and saved to {RESULTS_PATH}")

if __name__ == "__main__":
    main()
