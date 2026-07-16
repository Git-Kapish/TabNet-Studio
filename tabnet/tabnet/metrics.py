import numpy as np
from typing import Dict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Compute classification performance metrics:
    - Accuracy
    - Precision
    - Recall
    - F1 Score
    
    Automatically detects if binary or multi-class based on unique labels in y_true.
    
    Args:
        y_true: Ground truth integer labels, shape (N,).
        y_pred: Predicted class labels, shape (N,).
    Returns:
        Dict containing accuracy, precision, recall, and f1 metrics.
    """
    unique_labels = np.unique(y_true)
    is_binary = len(unique_labels) <= 2
    
    # Use binary average for 2 classes, and macro average for multi-class classification
    average = "binary" if is_binary else "macro"
    
    # Calculate precision, recall, f1
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average=average,
        zero_division=0
    )
    
    # Calculate accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1)
    }
