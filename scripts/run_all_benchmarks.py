import os
import json
import time
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from typing import Dict, Any
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from tabnet import TabNetClassifier, Trainer, TabularPreprocessor, get_data_loaders

def benchmark_adult():
    print("=== Benchmarking Adult Census Income (Binary Classification) ===")
    df = pd.read_csv("data/raw/adult/adult.csv")
    cat_cols = ["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]
    num_cols = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
    target_col = "income"
    df[target_col] = df[target_col].astype(str).str.strip().str.rstrip(".")
    
    split_idx = int(0.8 * len(df))
    train_df = df.iloc[:split_idx].copy()
    val_df = df.iloc[split_idx:].copy()
    
    le = LabelEncoder()
    y_train = le.fit_transform(train_df[target_col])
    y_val = le.transform(val_df[target_col])
    X_train = train_df.drop(columns=[target_col])
    X_val = val_df.drop(columns=[target_col])
    
    results = {}
    
    # 1. Logistic Regression
    lr_pre = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num_cols),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat_cols)
    ])
    lr_pipe = Pipeline([("pre", lr_pre), ("clf", LogisticRegression(max_iter=1000, random_state=42))])
    
    t0 = time.time()
    lr_pipe.fit(X_train, y_train)
    t_train = time.time() - t0
    t0 = time.time()
    preds = lr_pipe.predict(X_val)
    probs = lr_pipe.predict_proba(X_val)[:, 1]
    t_inf = time.time() - t0
    results["Logistic Regression"] = {
        "acc": accuracy_score(y_val, preds),
        "f1": f1_score(y_val, preds, average="macro"),
        "auc": roc_auc_score(y_val, probs),
        "train_time": t_train,
        "inf_time": t_inf
    }
    
    # 2. Random Forest
    rf_pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num_cols),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))]), cat_cols)
    ])
    rf_pipe = Pipeline([("pre", rf_pre), ("clf", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))])
    
    t0 = time.time()
    rf_pipe.fit(X_train, y_train)
    t_train = time.time() - t0
    t0 = time.time()
    preds = rf_pipe.predict(X_val)
    probs = rf_pipe.predict_proba(X_val)[:, 1]
    t_inf = time.time() - t0
    results["Random Forest"] = {
        "acc": accuracy_score(y_val, preds),
        "f1": f1_score(y_val, preds, average="macro"),
        "auc": roc_auc_score(y_val, probs),
        "train_time": t_train,
        "inf_time": t_inf
    }
    
    # 3. XGBoost
    if XGBOOST_AVAILABLE:
        xgb_pipe = Pipeline([("pre", rf_pre), ("clf", XGBClassifier(n_estimators=100, random_state=42, eval_metric="logloss", n_jobs=-1))])
        t0 = time.time()
        xgb_pipe.fit(X_train, y_train)
        t_train = time.time() - t0
        t0 = time.time()
        preds = xgb_pipe.predict(X_val)
        probs = xgb_pipe.predict_proba(X_val)[:, 1]
        t_inf = time.time() - t0
        results["XGBoost"] = {
            "acc": accuracy_score(y_val, preds),
            "f1": f1_score(y_val, preds, average="macro"),
            "auc": roc_auc_score(y_val, probs),
            "train_time": t_train,
            "inf_time": t_inf
        }
        
    # 4. TabNet
    prep = TabularPreprocessor(cat_cols=cat_cols, num_cols=num_cols, target_col=target_col)
    prep.fit(train_df)
    X_tr_tn, y_tr_tn = prep.transform(train_df)
    X_va_tn, y_va_tn = prep.transform(val_df)
    tr_loader, va_loader = get_data_loaders(X_tr_tn, y_tr_tn, X_va_tn, y_va_tn, batch_size=256, val_batch_size=512)
    
    model = TabNetClassifier(num_features=X_tr_tn.shape[1], num_classes=len(np.unique(y_tr_tn)), cat_idxs=prep.cat_idxs, cat_dims=prep.cat_dims, cat_emb_dims=prep.cat_emb_dims, n_d=8, n_a=8, n_steps=3, gamma=1.3)
    opt = optim.Adam(model.parameters(), lr=0.02)
    trainer = Trainer(model=model, optimizer=opt, device="cpu")
    
    t0 = time.time()
    trainer.fit(tr_loader, va_loader, max_epochs=10, lambda_sparse=0.001, run_name="bench_adult_tn")
    t_train = time.time() - t0
    
    t0 = time.time()
    model.eval()
    all_preds, all_probs, all_targets = [], [], []
    with torch.no_grad():
        for bx, by in va_loader:
            logits, _, _ = model(bx)
            p = torch.softmax(logits, dim=-1)
            all_probs.append(p[:, 1].numpy())
            all_preds.append(torch.argmax(p, dim=-1).numpy())
            all_targets.append(by.numpy())
    t_inf = time.time() - t0
    
    preds = np.concatenate(all_preds)
    probs = np.concatenate(all_probs)
    targets = np.concatenate(all_targets)
    
    results["TabNet"] = {
        "acc": accuracy_score(targets, preds),
        "f1": f1_score(targets, preds, average="macro"),
        "auc": roc_auc_score(targets, probs),
        "train_time": t_train,
        "inf_time": t_inf
    }
    
    return results

def benchmark_covertype():
    print("\n=== Benchmarking Forest Covertype (Multi-class Classification) ===")
    df = pd.read_csv("data/raw/covertype/covertype.csv")
    # Take a subsample of 50,000 for quick benchmarking
    df = df.sample(n=50000, random_state=42).reset_index(drop=True)
    target_col = "Cover_Type"
    num_cols = [c for c in df.columns if c != target_col]
    cat_cols = []
    
    split_idx = int(0.8 * len(df))
    train_df = df.iloc[:split_idx].copy()
    val_df = df.iloc[split_idx:].copy()
    
    le = LabelEncoder()
    y_train = le.fit_transform(train_df[target_col])
    y_val = le.transform(val_df[target_col])
    X_train = train_df.drop(columns=[target_col])
    X_val = val_df.drop(columns=[target_col])
    
    results = {}
    
    # 1. Logistic Regression
    lr_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=500, random_state=42))
    ])
    t0 = time.time()
    lr_pipe.fit(X_train, y_train)
    t_train = time.time() - t0
    t0 = time.time()
    preds = lr_pipe.predict(X_val)
    t_inf = time.time() - t0
    results["Logistic Regression"] = {
        "acc": accuracy_score(y_val, preds),
        "f1": f1_score(y_val, preds, average="macro"),
        "train_time": t_train,
        "inf_time": t_inf
    }
    
    # 2. Random Forest
    rf_pipe = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    t0 = time.time()
    rf_pipe.fit(X_train, y_train)
    t_train = time.time() - t0
    t0 = time.time()
    preds = rf_pipe.predict(X_val)
    t_inf = time.time() - t0
    results["Random Forest"] = {
        "acc": accuracy_score(y_val, preds),
        "f1": f1_score(y_val, preds, average="macro"),
        "train_time": t_train,
        "inf_time": t_inf
    }
    
    # 3. XGBoost
    if XGBOOST_AVAILABLE:
        xgb_pipe = XGBClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        t0 = time.time()
        xgb_pipe.fit(X_train, y_train)
        t_train = time.time() - t0
        t0 = time.time()
        preds = xgb_pipe.predict(X_val)
        t_inf = time.time() - t0
        results["XGBoost"] = {
            "acc": accuracy_score(y_val, preds),
            "f1": f1_score(y_val, preds, average="macro"),
            "train_time": t_train,
            "inf_time": t_inf
        }
        
    # 4. TabNet
    prep = TabularPreprocessor(cat_cols=cat_cols, num_cols=num_cols, target_col=target_col)
    prep.fit(train_df)
    X_tr_tn, y_tr_tn = prep.transform(train_df)
    X_va_tn, y_va_tn = prep.transform(val_df)
    tr_loader, va_loader = get_data_loaders(X_tr_tn, y_tr_tn, X_va_tn, y_va_tn, batch_size=512, val_batch_size=1024)
    
    model = TabNetClassifier(num_features=X_tr_tn.shape[1], num_classes=len(np.unique(y_tr_tn)), n_d=16, n_a=16, n_steps=4)
    opt = optim.Adam(model.parameters(), lr=0.02)
    trainer = Trainer(model=model, optimizer=opt, device="cpu")
    
    t0 = time.time()
    trainer.fit(tr_loader, va_loader, max_epochs=10, lambda_sparse=0.001, run_name="bench_cov_tn")
    t_train = time.time() - t0
    
    t0 = time.time()
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for bx, by in va_loader:
            logits, _, _ = model(bx)
            all_preds.append(torch.argmax(logits, dim=-1).numpy())
            all_targets.append(by.numpy())
    t_inf = time.time() - t0
    
    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    
    results["TabNet"] = {
        "acc": accuracy_score(targets, preds),
        "f1": f1_score(targets, preds, average="macro"),
        "train_time": t_train,
        "inf_time": t_inf
    }
    
    return results

if __name__ == "__main__":
    adult_res = benchmark_adult()
    cov_res = benchmark_covertype()
    
    print("\n\n====================== FINAL RESULTS ======================")
    print("Adult Results:", json.dumps(adult_res, indent=2))
    print("Covertype Results:", json.dumps(cov_res, indent=2))
