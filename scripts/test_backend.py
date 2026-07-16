import os
import time
import requests
import pandas as pd

API_URL = "http://127.0.0.1:8000"

def check_server_running() -> bool:
    try:
        response = requests.get(API_URL + "/docs", timeout=2)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

def main():
    print("=== TabNet Studio: Backend API Test Client ===")
    
    # 1. Verify server is online
    if not check_server_running():
        print(f"\n[Error] The FastAPI server is not running on {API_URL}.")
        print("Please start the server first in another terminal using:")
        print("  .\\.venv\\Scripts\\python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000")
        print("\nThen, run this test script again.")
        return
        
    print("\n[✓] FastAPI server is online and responding.")
    
    # 2. Test Dataset Upload
    dataset_path = "data/raw/adult/adult.csv"
    print(f"\n[1/5] Testing CSV Upload with {dataset_path}...")
    if not os.path.exists(dataset_path):
        print(f"Error: dataset file not found at {dataset_path}")
        return
        
    with open(dataset_path, "rb") as f:
        files = {"file": (os.path.basename(dataset_path), f, "text/csv")}
        response = requests.post(API_URL + "/api/dataset/upload", files=files)
        
    if response.status_code == 200:
        data = response.json()
        print("✓ Upload Success!")
        print(f"  - Uploaded File: {data['filename']}")
        print(f"  - Rows: {data['row_count']}")
        print(f"  - Detected Categorical Columns: {len([k for k, v in data['dtypes'].items() if v == 'categorical'])}")
        print(f"  - Detected Numeric Columns: {len([k for k, v in data['dtypes'].items() if v == 'numeric'])}")
    else:
        print(f"✗ Upload Failed: {response.text}")
        return
        
    # 3. Test Model Training Trigger (1 epoch)
    print("\n[2/5] Triggering a short training run (1 epoch) on Adult dataset...")
    train_payload = {
        "dataset_name": "adult",
        "target_col": "income",
        "n_d": 8,
        "n_a": 8,
        "n_steps": 3,
        "gamma": 1.3,
        "lambda_sparse": 0.001,
        "lr": 0.02,
        "batch_size": 1024,
        "epochs": 5,         # 1 epoch for verification speed
        "patience": 2,
        "seed": 42,
        "deterministic": True
    }
    
    response = requests.post(API_URL + "/api/train", json=train_payload)
    if response.status_code == 200:
        run_name = response.json()["run_name"]
        print(f"✓ Training triggered successfully! Run Name: {run_name}")
    else:
        print(f"✗ Failed to start training: {response.text}")
        return
        
    # 4. Test Training Status Polling
    print("\n[3/5] Polling training status (waiting for epoch completion)...")
    while True:
        status_resp = requests.get(API_URL + f"/api/train/status/{run_name}")
        if status_resp.status_code == 200:
            status_data = status_resp.json()
            status = status_data.get("status", "unknown")
            epoch = status_data.get("epoch", 0)
            loss = status_data.get("train_loss", 0.0)
            
            print(f"  - Run Status: {status} | Epoch: {epoch} | Train Loss: {loss:.4f}")
            if status in ["completed", "failed"]:
                if status == "failed":
                    print(f"✗ Training failed: {status_data.get('error', 'unknown error')}")
                    return
                break
        else:
            print(f"✗ Failed to query status: {status_resp.text}")
            return
        time.sleep(2)
        
    # 5. Test Models List & Export
    print("\n[4/5] Testing Models Listing & Weight Export...")
    models_resp = requests.get(API_URL + "/api/models")
    if models_resp.status_code == 200:
        models_list = models_resp.json()
        print(f"✓ Models list retrieved. Found {len(models_list)} trained runs.")
        run_found = any(m["run_name"] == run_name for m in models_list)
        print(f"  - Current run listed: {run_found}")
    else:
        print(f"✗ Failed to fetch models: {models_resp.text}")
        
    export_resp = requests.get(API_URL + f"/api/models/{run_name}/export")
    if export_resp.status_code == 200:
        print(f"✓ Weight export success. Checkpoint file (.pt) is downloadable (received {len(export_resp.content)} bytes).")
    else:
        print(f"✗ Failed to export model: {export_resp.text}")
        
    # 6. Test Batch Inference & Feature Importance
    print("\n[5/5] Testing batch prediction and interpretability mask fetches...")
    # Load first 5 rows of adult.csv to upload for predictions
    df_sample = pd.read_csv(dataset_path).head(5)
    df_sample.to_csv("temp_sample.csv", index=False)
    
    with open("temp_sample.csv", "rb") as f:
        files = {"file": ("temp_sample.csv", f, "text/csv")}
        pred_resp = requests.post(API_URL + f"/api/predict?run_name={run_name}", files=files)
        
    if os.path.exists("temp_sample.csv"):
        os.remove("temp_sample.csv")
        
    if pred_resp.status_code == 200:
        pred_data = pred_resp.json()
        print("✓ Prediction Success!")
        print(f"  - Predicted classes: {pred_data['predictions']}")
        print(f"  - Prediction probabilities length: {len(pred_data['probabilities'])}")
    else:
        print(f"✗ Prediction Failed: {pred_resp.text}")
        
    importance_resp = requests.get(API_URL + f"/api/feature-importance/{run_name}?num_samples=3")
    if importance_resp.status_code == 200:
        imp_data = importance_resp.json()
        print("✓ Feature Importance & Attention Masks retrieved successfully!")
        print(f"  - Features: {imp_data['feature_names']}")
        print(f"  - Global Feature Importances: {imp_data['global_importance']}")
        print(f"  - Sample 0 predictions and contribution weights: Pred={imp_data['samples'][0]['prediction']} | Step contribs={imp_data['samples'][0]['step_contributions']}")
    else:
        print(f"✗ Feature importance retrieval failed: {importance_resp.text}")
        
    print("\n=== Backend API Integration Test Complete! ===")

if __name__ == "__main__":
    main()
