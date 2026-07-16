import os
import json
from typing import Dict, Any

RESULTS_PATH = "benchmarks/results.json"

def format_size(size_bytes: int) -> str:
    if size_bytes == 0:
        return "N/A"
    elif size_bytes < 1024:
        return f"{size_bytes} Bytes"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.2f} MB"

def main():
    print("=== TabNet Studio: Benchmark Evaluation Report ===")
    
    if not os.path.exists(RESULTS_PATH):
        print(f"Error: Results file not found at {RESULTS_PATH}. Please run the training script first:\n")
        print("  python benchmarks/train_baselines.py\n")
        return
        
    with open(RESULTS_PATH, "r") as f:
        results = json.load(f)
        
    print("\n## Model Comparison Summary\n")
    print(
        f"| {'Model Name':<22} | {'Accuracy':<8} | {'Precision':<9} | {'Recall':<8} | {'F1-Score':<8} | {'Train Time':<10} | {'Inf Time':<8} | {'Model Size':<10} |"
    )
    print(
        f"| {'-'*22} | {'-'*8} | {'-'*9} | {'-'*8} | {'-'*8} | {'-'*10} | {'-'*8} | {'-'*10} |"
    )
    
    for key, data in results.items():
        metrics = data["metrics"]
        name = data["model_name"]
        train_time = f"{data['training_time_seconds']:.2f}s"
        inf_time = f"{data['inference_time_seconds']:.2f}s"
        size = format_size(data["model_size_bytes"])
        
        # If model was skipped (like XGBoost)
        if data["model_size_bytes"] == 0 and key == "xgboost":
            print(
                f"| {name:<22} | {'N/A':<8} | {'N/A':<9} | {'N/A':<8} | {'N/A':<8} | {'N/A':<10} | {'N/A':<8} | {'Skipped':<10} |"
            )
        else:
            print(
                f"| {name:<22} | {metrics['accuracy']:.4f} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | {metrics['f1']:.4f} | {train_time:<10} | {inf_time:<8} | {size:<10} |"
            )
            
    print("\n## Key Insights:")
    
    # Simple logic to find the best model based on F1-score
    best_model_name = ""
    best_f1 = -1.0
    for key, data in results.items():
        if data["model_size_bytes"] > 0: # Only count executed models
            f1 = data["metrics"]["f1"]
            if f1 > best_f1:
                best_f1 = f1
                best_model_name = data["model_name"]
                
    if best_model_name:
        print(f"- The model with the highest validation F1-Score is **{best_model_name}** with an F1 score of **{best_f1:.4f}**.")
    print("- **Logistic Regression** represents the simplest linear model with the smallest storage footprint and fast training, serving as a primary baseline.")
    print("- **Random Forest** trains quickly using parallel processors and generates competitive tree-based metrics.")
    print("- **TabNet** utilizes sparse instance-wise attention to reason on tabular columns. While neural networks have higher training overhead, they provide raw step-by-step interpretability (decision paths and feature attention masks) that tree ensemble models cannot natively match.")

if __name__ == "__main__":
    main()
