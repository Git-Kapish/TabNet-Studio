# PRD.md

# TabNet Studio

**Version:** 1.0

---

# Vision

TabNet Studio is an end-to-end machine learning application that demonstrates how a research paper can be transformed into a usable production system.

The project consists of a complete implementation of the TabNet architecture from the original paper, along with a web interface that allows users to train, evaluate, explain, and deploy models on tabular datasets.

Rather than focusing only on model training inside a notebook, the project showcases the complete ML workflow, including data preprocessing, model training, experiment comparison, inference through an API, and deployment.

---

# Problem Statement

Most machine learning portfolio projects stop after training a model in a Jupyter notebook.

This project demonstrates how modern deep learning research can be implemented from scratch and packaged into a deployable application that others can interact with.

---

# Goal

Build a complete implementation of the TabNet paper and expose it through an intuitive web application.

The project should demonstrate:

* Reading and understanding a research paper
* Implementing a deep learning architecture from scratch
* Training and evaluating models
* Comparing against traditional ML baselines
* Serving predictions through an API
* Deploying a complete ML application

---

# Target Users

The primary audience is:

* Recruiters
* Hiring managers
* Machine Learning Engineers
* Software Engineers
* Students interested in TabNet

Users should be able to visit the application, upload a dataset, train a model, and explore the results without needing to understand the implementation details.

---

# Core Features

## 1. Paper Reproduction

* Implement TabNet from scratch in PyTorch
* Reproduce results on benchmark datasets
* Document architectural decisions

---

## 2. Dataset Upload

Users can:

* Upload a CSV dataset
* Preview the data
* View dataset statistics

---

## 3. Data Processing

Automatically:

* Handle missing values
* Encode categorical features
* Normalize numerical features
* Split data into train and validation sets

---

## 4. Model Training

Train:

* TabNet
* Logistic Regression
* Random Forest
* XGBoost (optional)

Display:

* Training progress
* Validation metrics
* Loss curves

---

## 5. Model Comparison

Compare models using:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC
* Training time

---

## 6. Explainability

Visualize:

* Feature importance
* Decision steps
* Attention masks
* Prediction explanations

This is one of TabNet's key strengths and should be highlighted.

---

## 7. Prediction

Allow users to:

* Upload unseen data
* Generate predictions
* Download prediction results

---

## 8. REST API

Expose simple endpoints:

* Train a model
* Predict
* Retrieve model metrics

---

## 9. Deployment

Deploy the application using Docker.

Host it on a cloud platform so recruiters can try it without any local setup.

---

# Non-Goals

The following are intentionally excluded:

* Authentication
* User accounts
* Payments
* Multi-user support
* Kubernetes
* Distributed training
* Background workers
* Auto-retraining
* Notifications

The focus is demonstrating strong ML engineering rather than building a SaaS product.

---

# Technology Stack

## Machine Learning

* PyTorch
* NumPy
* Pandas
* Scikit-learn

## Backend

* FastAPI

## Frontend

* React
* TypeScript
* Tailwind CSS

## Deployment

* Docker
* GitHub Actions

---

# Success Criteria

The project is successful if it demonstrates that the developer can:

* Understand a research paper
* Implement the architecture correctly
* Train models successfully
* Compare against classical ML algorithms
* Build a clean API
* Create an intuitive frontend
* Deploy the application publicly

---

# Deliverables

* TabNet implementation from scratch
* Benchmark experiments
* Interactive web application
* REST API
* Dockerized deployment
* Technical documentation
* Public GitHub repository
* Live demo
