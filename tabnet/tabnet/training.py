import os
import json
import time
import random
from typing import Dict, Any, List, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tabnet.losses import SparsityLoss
from tabnet.metrics import compute_classification_metrics

def set_seed(seed: int, deterministic: bool = False):
    """
    Set seeds for reproducibility across random, numpy, and torch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class Trainer:
    """
    Trainer class responsible for managing the TabNet training and validation loops,
    tracking metrics, handling early stopping, logging to TensorBoard, and exporting model weights.
    """
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = "cpu",
        checkpoint_dir: str = "artifacts/checkpoints",
        tensorboard_dir: str = "artifacts/tensorboard",
        runs_dir: str = "artifacts/runs",
        patience: int = 10,
        clip_value: float = 2.0,
        seed: int = 42,
        deterministic: bool = False
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.tensorboard_dir = tensorboard_dir
        self.runs_dir = runs_dir
        self.patience = patience
        self.clip_value = clip_value
        self.seed = seed
        self.deterministic = deterministic
        
        # Ensure directories exist
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(tensorboard_dir, exist_ok=True)
        os.makedirs(runs_dir, exist_ok=True)
        
        # Set seeds
        set_seed(seed, deterministic)
        
        self.sparsity_loss_fn = SparsityLoss()
        self.clf_loss_fn = nn.CrossEntropyLoss()
        self.writer: Optional[SummaryWriter] = None

    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        max_epochs: int = 100,
        lambda_sparse: float = 1e-3,
        run_name: Optional[str] = None,
        epoch_callback = None
    ) -> Dict[str, Any]:
        """
        Train and validate the TabNet model.
        """
        if run_name is None:
            run_name = f"run_{int(time.time())}"
            
        tb_log_path = os.path.join(self.tensorboard_dir, run_name)
        self.writer = SummaryWriter(log_dir=tb_log_path)
        
        best_val_loss = float("inf")
        epochs_no_improve = 0
        best_metrics: Dict[str, float] = {}
        
        start_time = time.time()
        
        for epoch in range(1, max_epochs + 1):
            epoch_start_time = time.time()
            
            # --- Training Loop ---
            self.model.train()
            train_clf_loss = 0.0
            train_sparse_loss = 0.0
            train_total_loss = 0.0
            
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                
                self.optimizer.zero_grad()
                
                # Forward pass
                logits, step_masks, _ = self.model(X_batch)
                
                # Losses
                clf_loss = self.clf_loss_fn(logits, y_batch)
                sparse_loss = self.sparsity_loss_fn(step_masks)
                total_loss = clf_loss + lambda_sparse * sparse_loss
                
                # Backward pass
                total_loss.backward()
                
                # Gradient clipping to prevent gradient explosion
                if self.clip_value > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_value)
                    
                self.optimizer.step()
                
                train_clf_loss += clf_loss.item() * X_batch.size(0)
                train_sparse_loss += sparse_loss.item() * X_batch.size(0)
                train_total_loss += total_loss.item() * X_batch.size(0)
                
            n_train = len(train_loader.dataset)
            train_clf_loss /= n_train
            train_sparse_loss /= n_train
            train_total_loss /= n_train
            
            # Epoch LR decay if scheduler is present
            if self.scheduler is not None:
                # Some schedulers accept metric (e.g. ReduceLROnPlateau), others don't
                # We assume step is epoch-based step
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    pass # Will step after validation
                else:
                    self.scheduler.step()
                    
            # --- Validation Loop ---
            val_loss, val_metrics = self.evaluate(val_loader)
            
            # Step ReduceLROnPlateau if present
            if self.scheduler is not None and isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss)
                
            current_lr = self.optimizer.param_groups[0]["lr"]
            epoch_duration = time.time() - epoch_start_time
            
            # --- Log metrics to TensorBoard ---
            self.writer.add_scalar("Loss/Train_Classifier", train_clf_loss, epoch)
            self.writer.add_scalar("Loss/Train_Sparsity", train_sparse_loss, epoch)
            self.writer.add_scalar("Loss/Train_Total", train_total_loss, epoch)
            self.writer.add_scalar("Loss/Val_Loss", val_loss, epoch)
            self.writer.add_scalar("Metrics/Val_Accuracy", val_metrics["accuracy"], epoch)
            self.writer.add_scalar("Metrics/Val_Precision", val_metrics["precision"], epoch)
            self.writer.add_scalar("Metrics/Val_Recall", val_metrics["recall"], epoch)
            self.writer.add_scalar("Metrics/Val_F1", val_metrics["f1"], epoch)
            self.writer.add_scalar("Params/Learning_Rate", current_lr, epoch)
            
            print(
                f"Epoch {epoch:03d}/{max_epochs:03d} | "
                f"Train Loss: {train_total_loss:.4f} (Clf: {train_clf_loss:.4f}, Sparsity: {train_sparse_loss:.4f}) | "
                f"Val Loss: {val_loss:.4f} | Accuracy: {val_metrics['accuracy']:.4f} | F1: {val_metrics['f1']:.4f} | "
                f"LR: {current_lr:.6f} | {epoch_duration:.1f}s"
            )
            
            if epoch_callback is not None:
                epoch_callback(epoch, train_total_loss, val_loss, val_metrics, current_lr)
            
            # --- Early Stopping & Checkpoint Save ---
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                best_metrics = val_metrics
                best_metrics["epoch"] = epoch
                best_metrics["val_loss"] = val_loss
                
                # Save best model
                checkpoint_path = os.path.join(self.checkpoint_dir, f"{run_name}_best_model.pt")
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_loss": val_loss,
                    "metrics": val_metrics
                }, checkpoint_path)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    print(f"Early stopping triggered at epoch {epoch}. No validation loss improvement for {self.patience} epochs.")
                    break
                    
        total_duration = time.time() - start_time
        
        # Load the best model weights back into memory
        best_checkpoint = os.path.join(self.checkpoint_dir, f"{run_name}_best_model.pt")
        if os.path.exists(best_checkpoint):
            checkpoint = torch.load(best_checkpoint, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            
        # --- Save run.json Metadata ---
        metadata = {
            "run_name": run_name,
            "hyperparameters": {
                "lambda_sparse": lambda_sparse,
                "max_epochs": max_epochs,
                "patience": self.patience,
                "clip_value": self.clip_value,
                "seed": self.seed,
                "deterministic": self.deterministic,
                "optimizer": self.optimizer.__class__.__name__,
                "initial_lr": self.optimizer.param_groups[0]["initial_lr"] if "initial_lr" in self.optimizer.param_groups[0] else None
            },
            "best_epoch": best_metrics.get("epoch"),
            "best_val_loss": best_metrics.get("val_loss"),
            "evaluation_metrics": {
                "accuracy": best_metrics.get("accuracy"),
                "precision": best_metrics.get("precision"),
                "recall": best_metrics.get("recall"),
                "f1": best_metrics.get("f1")
            },
            "training_duration_seconds": total_duration,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
            "checkpoint_location": os.path.abspath(best_checkpoint),
            "tensorboard_log_location": os.path.abspath(tb_log_path),
            "project_version": "1.0"
        }
        
        metadata_path = os.path.join(self.runs_dir, f"{run_name}_run.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)
            
        if self.writer is not None:
            self.writer.close()
            
        print(f"[OK] Training finished. Best Val Loss: {best_val_loss:.4f} | Accuracy: {best_metrics.get('accuracy'):.4f} | Run saved to {metadata_path}")
        return metadata

    def evaluate(self, val_loader: torch.utils.data.DataLoader) -> Tuple[float, Dict[str, float]]:
        """
        Evaluate the model on a validation dataset.
        Returns:
            val_loss: Average validation loss.
            metrics: Dict containing accuracy, precision, recall, and f1 scores.
        """
        self.model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                
                # Forward pass
                logits, _, _ = self.model(X_batch)
                
                # Classification loss
                loss = self.clf_loss_fn(logits, y_batch)
                val_loss += loss.item() * X_batch.size(0)
                
                # Compute predictions
                preds = torch.argmax(logits, dim=-1)
                
                all_preds.append(preds.cpu().numpy())
                all_targets.append(y_batch.cpu().numpy())
                
        n_val = len(val_loader.dataset)
        val_loss /= n_val
        
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        
        metrics = compute_classification_metrics(all_targets, all_preds)
        return val_loss, metrics
