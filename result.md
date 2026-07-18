| Model                   | Accuracy   | F1-Score   | Training Time | Inference Latency (Batch) | Model Size |
| ----------------------- | ---------- | ---------- | ------------- | ------------------------- | ---------- |
| **XGBoost**             | 87.16%     | 0.7108     | 0.33s         | 24.21 ms                  | 326 KB     |
| **Random Forest**       | 85.56%     | 0.6684     | 0.72s         | 68.77 ms                  | 91.4 MB    |
| **Logistic Regression** | 84.94%     | 0.6548     | 0.86s         | 22.28 ms                  | 8.1 KB     |
| **TabNet (PyTorch)**    | 84.14%     | 0.5920     | 31.86s        | 151.45 ms                 | 532 KB     |