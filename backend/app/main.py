import os
import sys
import time

# Resolve CWD import collision by adding the outer package path first
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../tabnet")))

import json
import joblib
import threading
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from fastapi import FastAPI, BackgroundTasks, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

import torch
import torch.optim as optim

from tabnet import (
    TabNetClassifier,
    Trainer,
    TabularPreprocessor,
    get_data_loaders,
    compute_local_feature_importance,
    compute_global_feature_importance,
    get_attention_masks
)

app = FastAPI(title="TabNet Studio API", version="1.0")

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production security if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories
DATA_DIR = "data/raw"
UPLOAD_DIR = "data/raw/uploaded"
CHECKPOINTS_DIR = "artifacts/checkpoints"
RUNS_DIR = "artifacts/runs"
TENSORBOARD_DIR = "artifacts/tensorboard"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
os.makedirs(RUNS_DIR, exist_ok=True)

# In-memory progress tracking: run_name -> status dictionary
training_status: Dict[str, Dict[str, Any]] = {}

@app.get("/api/status")
def api_status():
    """
    Return live runtime information: PyTorch version and CUDA device details.
    """
    cuda = torch.cuda.is_available()
    device = torch.cuda.get_device_name(0) if cuda else "cpu"
    return {
        "status": "ok",
        "torch_version": torch.__version__,
        "cuda": cuda,
        "device": device,
    }

class TrainRequest(BaseModel):
    dataset_name: str  # "adult", "covertype", or custom uploaded filename
    target_col: str
    n_d: int = 8
    n_a: int = 8
    n_steps: int = 3
    gamma: float = 1.3
    lambda_sparse: float = 0.001
    lr: float = 0.02
    batch_size: int = 256
    epochs: int = 10
    patience: int = 5
    seed: int = 42
    deterministic: bool = False

def train_tabnet_task(run_name: str, req: TrainRequest):
    """
    Background worker thread to perform TabNet training.
    """
    try:
        # 1. Resolve dataset path
        if req.dataset_name == "adult":
            path = os.path.join(DATA_DIR, "adult", "adult.csv")
        elif req.dataset_name == "covertype":
            path = os.path.join(DATA_DIR, "covertype", "covertype.csv")
        else:
            path = os.path.join(UPLOAD_DIR, req.dataset_name)
            
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset file not found at {path}")
            
        # 2. Load dataset
        df = pd.read_csv(path)
        
        # Validate target column
        if req.target_col not in df.columns:
            raise ValueError(f"Target column '{req.target_col}' not found in the dataset.")
            
        # Clean target if it's non-numeric (e.g. Adult dataset labels)
        if not pd.api.types.is_numeric_dtype(df[req.target_col].dtype):
            df[req.target_col] = df[req.target_col].astype(str).str.strip().str.rstrip(".")
            
        # 3. Automatically segregate features
        X_df = df.drop(columns=[req.target_col])
        cat_cols = []
        num_cols = []
        for col in X_df.columns:
            if pd.api.types.is_string_dtype(X_df[col].dtype) or isinstance(X_df[col].dtype, pd.CategoricalDtype) or X_df[col].dtype == "bool":
                cat_cols.append(col)
            else:
                num_cols.append(col)
                
        # 4. Fit Preprocessor
        preprocessor = TabularPreprocessor(
            cat_cols=cat_cols,
            num_cols=num_cols,
            target_col=req.target_col
        )
        preprocessor.fit(df)
        X, y = preprocessor.transform(df)
        
        # Save preprocessor for inference later
        prep_path = os.path.join(CHECKPOINTS_DIR, f"{run_name}_preprocessor.joblib")
        joblib.dump(preprocessor, prep_path)
        
        # 5. Split train/validation (80/20)
        split_idx = int(0.8 * len(X))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        train_loader, val_loader = get_data_loaders(
            X_train, y_train, X_val, y_val,
            batch_size=req.batch_size,
            val_batch_size=req.batch_size * 2
        )
        
        # 6. Initialize model
        model = TabNetClassifier(
            num_features=X.shape[1],
            num_classes=len(np.unique(y)),
            cat_idxs=preprocessor.cat_idxs,
            cat_dims=preprocessor.cat_dims,
            cat_emb_dims=preprocessor.cat_emb_dims,
            n_d=req.n_d,
            n_a=req.n_a,
            n_steps=req.n_steps,
            gamma=req.gamma,
            n_shared=2,
            n_dependent=2,
            virtual_batch_size=128 if req.batch_size >= 128 else max(2, req.batch_size // 2)
        )
        
        optimizer = optim.Adam(model.parameters(), lr=req.lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
        
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device="cpu",
            checkpoint_dir=CHECKPOINTS_DIR,
            tensorboard_dir=TENSORBOARD_DIR,
            runs_dir=RUNS_DIR,
            patience=req.patience,
            clip_value=2.0,
            seed=req.seed,
            deterministic=req.deterministic
        )
        
        # Callback to stream training status updates
        def progress_callback(epoch, train_loss, val_loss, metrics, lr):
            training_status[run_name].update({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "accuracy": metrics["accuracy"],
                "f1": metrics["f1"],
                "lr": lr
            })
            
        # Fit TabNet
        run_metadata = trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            max_epochs=req.epochs,
            lambda_sparse=req.lambda_sparse,
            run_name=run_name,
            epoch_callback=progress_callback
        )
        
        # Inject hyperparams into saved run.json
        run_file = os.path.join(RUNS_DIR, f"{run_name}_run.json")
        if os.path.exists(run_file):
            with open(run_file, "r") as f:
                meta = json.load(f)
            meta["model_config"] = {
                "n_d": req.n_d,
                "n_a": req.n_a,
                "n_steps": req.n_steps,
                "gamma": req.gamma
            }
            with open(run_file, "w") as f:
                json.dump(meta, f, indent=4)
                
        training_status[run_name].update({
            "status": "completed",
            "val_loss": run_metadata["best_val_loss"],
            "accuracy": run_metadata["evaluation_metrics"]["accuracy"],
            "f1": run_metadata["evaluation_metrics"]["f1"],
            "early_stopping_triggered": run_metadata.get("early_stopping_triggered", False),
            "stopped_epoch": run_metadata.get("stopped_epoch"),
            "best_epoch": run_metadata.get("best_epoch"),
            "patience": run_metadata["hyperparameters"]["patience"]
        })
        
    except Exception as e:
        training_status[run_name] = {
            "status": "failed",
            "error": str(e)
        }

@app.post("/api/dataset/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """
    Upload a custom CSV dataset and return header analysis.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")
        
    save_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(save_path, "wb") as f:
        f.write(await file.read())
        
    # Analyze uploaded CSV
    try:
        df = pd.read_csv(save_path)
        dtypes = {}
        for col in df.columns:
            if pd.api.types.is_string_dtype(df[col].dtype) or isinstance(df[col].dtype, pd.CategoricalDtype) or df[col].dtype == "bool":
                dtypes[col] = "categorical"
            else:
                dtypes[col] = "numeric"
                
        return {
            "filename": file.filename,
            "row_count": len(df),
            "columns": list(df.columns),
            "dtypes": dtypes
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse CSV: {str(e)}")

@app.post("/api/train")
async def start_training(req: TrainRequest, background_tasks: BackgroundTasks):
    """
    Trigger dynamic training loop for the selected dataset.
    """
    timestamp = int(time.time())
    run_name = f"{req.dataset_name.split('.')[0]}_run_{timestamp}"
    
    # Initialize run status in-memory
    training_status[run_name] = {
        "status": "training",
        "epoch": 0,
        "max_epochs": req.epochs,
        "train_loss": 0.0,
        "val_loss": 0.0,
        "accuracy": 0.0,
        "f1": 0.0,
        "lr": req.lr,
        "early_stopping_triggered": False,
        "stopped_epoch": None,
        "best_epoch": None,
        "patience": req.patience,
        "error": None
    }
    
    # Spawn background training thread
    thread = threading.Thread(target=train_tabnet_task, args=(run_name, req))
    thread.start()
    
    return {"run_name": run_name}

@app.get("/api/train/status/{run_name}")
async def get_training_status(run_name: str):
    """
    Fetch active progress updates for a running training process.
    """
    if run_name not in training_status:
        raise HTTPException(status_code=404, detail="Run not found.")
    return training_status[run_name]

@app.get("/api/models")
async def list_models():
    """
    Retrieve all compiled training run metadata.
    """
    models = []
    if not os.path.exists(RUNS_DIR):
        return []
        
    for file in os.listdir(RUNS_DIR):
        if file.endswith("_run.json"):
            with open(os.path.join(RUNS_DIR, file), "r") as f:
                try:
                    meta = json.load(f)
                    models.append(meta)
                except Exception:
                    pass
    return models

@app.get("/api/benchmarks")
async def get_benchmarks():
    """
    Retrieve baseline comparison metrics from the benchmark suite.
    """
    path = "benchmarks/results.json"
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Benchmark results not compiled yet.")
    with open(path, "r") as f:
        try:
            return json.load(f)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load benchmarks: {str(e)}")

@app.get("/api/models/{run_name}/export")
async def export_model(run_name: str):
    """
    Download checkpoint weights as a standard .pt file.
    """
    checkpoint_file = f"{run_name}_best_model.pt"
    checkpoint_path = os.path.join(CHECKPOINTS_DIR, checkpoint_file)
    
    if not os.path.exists(checkpoint_path):
        raise HTTPException(status_code=404, detail="Checkpoint file not found.")
        
    return FileResponse(
        path=checkpoint_path,
        filename=checkpoint_file,
        media_type="application/octet-stream"
    )

@app.post("/api/predict")
async def predict(run_name: str, file: UploadFile = File(...)):
    """
    Perform batch inference on an uploaded CSV file using a saved checkpoint.
    """
    # 1. Load checkpoints
    prep_path = os.path.join(CHECKPOINTS_DIR, f"{run_name}_preprocessor.joblib")
    model_path = os.path.join(CHECKPOINTS_DIR, f"{run_name}_best_model.pt")
    
    if not os.path.exists(prep_path) or not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model checkpoint or preprocessor not found.")
        
    preprocessor: TabularPreprocessor = joblib.load(prep_path)
    checkpoint = torch.load(model_path, map_location="cpu")
    
    # 2. Read and transform input file
    try:
        df = pd.read_csv(file.file)
        
        # Verify columns match training schema (excluding target column if present)
        # TabularPreprocessor needs the target column to run transform, but we can mock it if missing
        has_target = preprocessor.target_col in df.columns
        if not has_target:
            # Inject a mock target column filled with dummy values for transform compatibility
            df[preprocessor.target_col] = 0
            
        X, y = preprocessor.transform(df)
        X_tensor = torch.tensor(X, dtype=torch.float32)
        
        # 3. Instantiate model
        run_file = os.path.join(RUNS_DIR, f"{run_name}_run.json")
        cfg = {}
        if os.path.exists(run_file):
            with open(run_file, "r") as f:
                cfg = json.load(f).get("model_config", {})
                
        model = TabNetClassifier(
            num_features=X.shape[1],
            # Extract number of classes from labels inside state dict mapping (dim size of final_mapping weights)
            num_classes=checkpoint["model_state_dict"]["final_mapping.weight"].shape[0],
            cat_idxs=preprocessor.cat_idxs,
            cat_dims=preprocessor.cat_dims,
            cat_emb_dims=preprocessor.cat_emb_dims,
            n_d=cfg.get("n_d", 8),
            n_a=cfg.get("n_a", 8),
            n_steps=cfg.get("n_steps", 3),
            gamma=cfg.get("gamma", 1.3)
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        
        # 4. Predict
        with torch.no_grad():
            logits, _, _ = model(X_tensor)
            probs = torch.softmax(logits, dim=-1).numpy().tolist()
            preds = torch.argmax(logits, dim=-1).numpy().tolist()
            
        # Decode target class names if available
        decoded_preds = preprocessor.label_encoder.inverse_transform(preds)
        decoded_preds = [str(p) for p in decoded_preds]
        
        return {
            "predictions": decoded_preds,
            "probabilities": probs,
            "has_ground_truth": has_target
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

@app.get("/api/feature-importance/{run_name}")
async def get_feature_importance(run_name: str, num_samples: int = 5):
    """
    Compute local and global feature importances for Architecture Explorer displays.
    """
    prep_path = os.path.join(CHECKPOINTS_DIR, f"{run_name}_preprocessor.joblib")
    model_path = os.path.join(CHECKPOINTS_DIR, f"{run_name}_best_model.pt")
    run_file = os.path.join(RUNS_DIR, f"{run_name}_run.json")
    
    if not os.path.exists(prep_path) or not os.path.exists(model_path) or not os.path.exists(run_file):
        raise HTTPException(status_code=404, detail="Run files not found.")
        
    preprocessor: TabularPreprocessor = joblib.load(prep_path)
    checkpoint = torch.load(model_path, map_location="cpu")
    
    with open(run_file, "r") as f:
        run_meta = json.load(f)
        
    dataset_name = run_name.split("_run_")[0]
    
    # Resolve source dataset file to pull samples for evaluation
    if dataset_name == "adult":
        path = os.path.join(DATA_DIR, "adult", "adult.csv")
    elif dataset_name == "covertype":
        path = os.path.join(DATA_DIR, "covertype", "covertype.csv")
    else:
        path = os.path.join(UPLOAD_DIR, f"{dataset_name}.csv")
        
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"Original dataset source not found at {path}")
        
    df = pd.read_csv(path)
    
    # Slice the validation block (20%) just like training split
    split_idx = int(0.8 * len(df))
    val_df = df.iloc[split_idx:].copy()
    
    # Take a small subset of samples to return to the UI
    subset_df = val_df.head(num_samples).copy()
    
    # Process
    if not pd.api.types.is_numeric_dtype(subset_df[preprocessor.target_col].dtype):
        subset_df[preprocessor.target_col] = subset_df[preprocessor.target_col].astype(str).str.strip().str.rstrip(".")
        
    X_val, y_val = preprocessor.transform(subset_df)
    X_tensor = torch.tensor(X_val, dtype=torch.float32)
    
    # Instantiate and evaluate
    cfg = run_meta.get("model_config", {})
    model = TabNetClassifier(
        num_features=X_val.shape[1],
        num_classes=checkpoint["model_state_dict"]["final_mapping.weight"].shape[0],
        cat_idxs=preprocessor.cat_idxs,
        cat_dims=preprocessor.cat_dims,
        cat_emb_dims=preprocessor.cat_emb_dims,
        n_d=cfg.get("n_d", 8),
        n_a=cfg.get("n_a", 8),
        n_steps=cfg.get("n_steps", 3),
        gamma=cfg.get("gamma", 1.3)
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    with torch.no_grad():
        logits, step_masks, decision_outputs = model(X_tensor)
        
        # local & global importance
        local_imp = compute_local_feature_importance(step_masks, decision_outputs).numpy()
        global_imp = compute_global_feature_importance(step_masks, decision_outputs).numpy()
        
        # Convert step masks to shape (num_samples, n_steps, num_features)
        stacked_masks = get_attention_masks(step_masks).permute(1, 0, 2).numpy()
        
        # Convert decision outputs to step contributions (num_samples, n_steps)
        # step_contrib[b, i] = sum_c(ReLU(d_{b,c}[i]))
        step_contributions = []
        for d in decision_outputs:
            eta = torch.sum(torch.relu(d), dim=-1).numpy()
            step_contributions.append(eta)
        step_contributions = np.stack(step_contributions, axis=1) # shape (num_samples, n_steps)
        
        preds = torch.argmax(logits, dim=-1).numpy()
        
    # Format samples for Architecture Explorer
    samples_list = []
    for b in range(len(subset_df)):
        raw_row = subset_df.iloc[b].to_dict()
        # Decode prediction and target labels
        pred_label = str(preprocessor.label_encoder.inverse_transform([preds[b]])[0])
        actual_label = str(subset_df.iloc[b][preprocessor.target_col])
        
        samples_list.append({
            "index": b,
            "raw_features": {k: v for k, v in raw_row.items() if k != preprocessor.target_col},
            "step_masks": stacked_masks[b].tolist(), # shape (n_steps, num_features)
            "step_contributions": step_contributions[b].tolist(), # shape (n_steps)
            "local_importance": local_imp[b].tolist(), # shape (num_features)
            "prediction": pred_label,
            "actual": actual_label
        })
        
    return {
        "feature_names": preprocessor.feature_names,
        "global_importance": global_imp.tolist(),
        "samples": samples_list
    }

@app.get("/api/benchmark/hardware")
async def benchmark_hardware():
    """
    Benchmark TabNet forward and backward pass speed on CPU vs GPU.
    """
    import time
    
    # 1. CPU Benchmarking
    device_cpu = torch.device("cpu")
    model_cpu = TabNetClassifier(num_features=20, num_classes=2, n_d=8, n_a=8, n_steps=3)
    model_cpu.to(device_cpu)
    
    X_cpu = torch.randn(1024, 20, device=device_cpu)
    y_cpu = torch.randint(0, 2, (1024,), device=device_cpu)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Warmup
    for _ in range(2):
        logits, _, _ = model_cpu(X_cpu)
        loss = criterion(logits, y_cpu)
        loss.backward()
        
    # Measure
    start_cpu = time.time()
    for _ in range(10):
        logits, _, _ = model_cpu(X_cpu)
        loss = criterion(logits, y_cpu)
        loss.backward()
    cpu_time = (time.time() - start_cpu) / 10.0
    cpu_throughput = 1024 / cpu_time
    
    # 2. GPU Benchmarking
    cuda_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else "N/A"
    
    gpu_time = 0.0
    gpu_throughput = 0.0
    
    if cuda_available:
        try:
            device_gpu = torch.device("cuda")
            model_gpu = TabNetClassifier(num_features=20, num_classes=2, n_d=8, n_a=8, n_steps=3)
            model_gpu.to(device_gpu)
            X_gpu = torch.randn(1024, 20, device=device_gpu)
            y_gpu = torch.randint(0, 2, (1024,), device=device_gpu)
            
            # Warmup
            for _ in range(2):
                logits, _, _ = model_gpu(X_gpu)
                loss = criterion(logits, y_gpu)
                loss.backward()
            torch.cuda.synchronize()
            
            # Measure
            start_gpu = time.time()
            for _ in range(10):
                logits, _, _ = model_gpu(X_gpu)
                loss = criterion(logits, y_gpu)
                loss.backward()
            torch.cuda.synchronize()
            gpu_time = (time.time() - start_gpu) / 10.0
            gpu_throughput = 1024 / gpu_time
        except Exception:
            cuda_available = False
            gpu_name = "CUDA execution failed"
            
    return {
        "cuda_available": cuda_available,
        "gpu_name": gpu_name,
        "cpu": {
            "device_name": "CPU Host (Single Instance)",
            "time_per_batch_ms": cpu_time * 1000.0,
            "throughput_samples_per_sec": cpu_throughput
        },
        "gpu": {
            "device_name": gpu_name,
            "time_per_batch_ms": gpu_time * 1000.0,
            "throughput_samples_per_sec": gpu_throughput
        }
    }
