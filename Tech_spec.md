# TECH_SPEC.md

# TabNet Studio

### Technical Specification

**Version:** 2.0

---

# 1. Overview

TabNet Studio is an end-to-end machine learning application built around a **standalone implementation of the TabNet architecture**.

The project is divided into two independent parts:

1. **TabNet Library** – A reusable PyTorch implementation of the TabNet paper.
2. **Application Layer** – A web application that uses the library for training, evaluation, inference, and visualization.

This separation keeps the deep learning implementation independent of any frontend or backend framework, making it reusable, testable, and easy to maintain.

---

# 2. System Architecture

```text
                   CSV Dataset
                        │
                        ▼
              Data Preprocessing
                        │
                        ▼
                TabNet Library
              (Pure PyTorch Code)
                        │
        ┌───────────────┴───────────────┐
        ▼                               ▼
   Training Pipeline             Inference Engine
        │                               │
        └───────────────┬───────────────┘
                        ▼
                 FastAPI Backend
                        │
                        ▼
                 React Frontend
```

---

# 3. Repository Structure

```text
tabnet-studio/

├── tabnet/                     # Standalone TabNet implementation
│
│   ├── tabnet/
│   │   ├── layers.py
│   │   ├── feature_transformer.py
│   │   ├── attentive_transformer.py
│   │   ├── decision_step.py
│   │   ├── encoder.py
│   │   ├── model.py
│   │   ├── losses.py
│   │   ├── metrics.py
│   │   ├── utils.py
│   │   └── __init__.py
│   │
│   ├── tests/
│   ├── examples/
│   ├── pyproject.toml
│   └── README.md
│
├── backend/
│
│   ├── app/
│   │   ├── api/
│   │   ├── services/
│   │   ├── preprocessing/
│   │   ├── experiments/
│   │   ├── inference/
│   │   ├── storage/
│   │   └── main.py
│   │
│   ├── datasets/
│   ├── checkpoints/
│   └── requirements.txt
│
├── frontend/
│
├── docs/
│
└── docker/
```

---

# 4. Module Responsibilities

## TabNet Library

The library contains **only deep learning code**.

It has **no dependency on**:

* FastAPI
* React
* Databases
* HTTP
* Docker
* File uploads

It should be installable independently.

Example:

```python
from tabnet import TabNetClassifier

model = TabNetClassifier(...)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
```

The goal is that another developer could install this package and use it without knowing anything about TabNet Studio.

---

## Backend

The backend is responsible for orchestrating the machine learning workflow.

Responsibilities:

* Receive datasets
* Preprocess data
* Train models
* Save checkpoints
* Serve predictions
* Expose REST APIs

The backend **never implements neural network logic**.

Instead, it imports the TabNet library.

---

## Frontend

The frontend provides a graphical interface for interacting with the backend.

Responsibilities:

* Upload datasets
* Configure experiments
* View training progress
* Compare models
* Visualize feature importance
* Generate predictions

The frontend has no machine learning logic.

---

# 5. TabNet Library Design

The implementation follows the original paper as closely as possible.

```text
TabNet

│

├── Embedding Layer

├── Feature Transformer

├── Attentive Transformer

├── Decision Step

├── Encoder

├── Classifier

└── Loss Functions
```

Each module is implemented and tested independently.

---

## Feature Transformer

Responsibilities:

* Shared feature blocks
* Independent feature blocks
* GLU blocks
* Residual connections

---

## Attentive Transformer

Responsibilities:

* Sparse feature selection
* Attention masks
* Prior updates

---

## Decision Step

Contains:

* Feature Transformer
* Attentive Transformer

Produces:

* Decision output
* Updated feature mask

---

## Encoder

Runs all decision steps sequentially.

Produces:

* Final representation
* Feature importance masks

---

## Classifier

Maps encoder output to prediction probabilities.

---

# 6. Training Pipeline

```text
CSV

↓

Preprocessing

↓

Train / Validation Split

↓

TabNet.fit()

↓

Validation

↓

Metrics

↓

Checkpoint
```

The training pipeline lives in the backend and uses the public API exposed by the TabNet library.

---

# 7. Public Library API

The library should expose a clean, minimal interface.

```python
model = TabNetClassifier(...)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

probabilities = model.predict_proba(X_test)

importance = model.feature_importance()

masks = model.attention_masks()
```

Internal implementation details remain hidden.

---

# 8. Backend Services

## Dataset Service

* Load CSV
* Validate schema
* Store uploaded datasets

---

## Preprocessing Service

* Missing value handling
* Encoding
* Scaling
* Train-validation split

---

## Training Service

* Create model
* Train model
* Save checkpoints
* Evaluate performance

---

## Inference Service

* Load trained model
* Run predictions
* Return probabilities

---

## Experiment Service

Store:

* Dataset name
* Parameters
* Metrics
* Training duration
* Model path

---

# 9. Frontend Pages

## Home

Project overview

Architecture

Paper summary

---

## Train

Dataset upload

Training configuration

Training progress

---

## Results

Accuracy

Precision

Recall

F1 Score

ROC-AUC

Loss curves

---

## Model Comparison

Compare:

* TabNet
* Logistic Regression
* Random Forest
* XGBoost

---

## Explainability

Display:

* Feature importance
* Attention masks
* Decision-step visualization

---

## Predict

Upload CSV

Generate predictions

Download results

---

# 10. REST API

```
POST /dataset

POST /train

GET /experiments

GET /metrics

POST /predict
```

The API communicates only with backend services.

It never interacts directly with neural network modules.

---

# 11. Testing Strategy

## Library Tests

Validate:

* Feature Transformer
* Attentive Transformer
* Decision Step
* Encoder
* Forward pass
* Loss functions
* Gradient flow

These tests ensure correctness of the TabNet implementation.

---

## Backend Tests

Validate:

* Dataset upload
* Preprocessing
* Training pipeline
* Prediction endpoints
* Experiment storage

---

## Frontend Tests

Validate:

* Dataset upload flow
* Results rendering
* Explainability visualizations

---

# 12. Deployment

Deploy as two containers:

* Frontend
* Backend

The backend installs the local TabNet package during the Docker build.

No code duplication exists between the application and the library.

---

# 13. Engineering Principles

* Keep the TabNet implementation framework-agnostic.
* Keep business logic outside the neural network code.
* Design modules with single responsibilities.
* Favor composition over tightly coupled code.
* Write unit tests before integrating modules.
* Make the TabNet library reusable in other projects.

---

# 14. Development Roadmap

### Phase 1 — Research

* Read the paper thoroughly.
* Understand every architectural component.
* Reproduce benchmark datasets.

---

### Phase 2 — Library Development

* Implement TabNet from scratch.
* Write unit tests for every module.
* Validate against the paper.

---

### Phase 3 — Training Pipeline

* Build preprocessing.
* Implement training loops.
* Evaluate against baseline models.

---

### Phase 4 — Backend

* Build FastAPI endpoints.
* Integrate the TabNet library.
* Save experiments and checkpoints.

---

### Phase 5 — Frontend

* Build the React interface.
* Visualize metrics and feature importance.
* Support predictions from uploaded CSV files.

---

### Phase 6 — Deployment

* Dockerize the application.
* Deploy publicly.
* Complete documentation.
* Record a demonstration video.

---

# 15. Final Deliverables

* Standalone `tabnet` Python library
* Complete implementation of the TabNet paper
* Benchmark reproduction report
* FastAPI backend
* React frontend
* REST API
* Dockerized deployment
* Public GitHub repository
* Live demo
* Technical documentation
