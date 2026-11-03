# PhishGuard-AI

### Evaluating Federated Learning for Phishing Detection under IID and Non-IID Data Distributions

## Overview
PhishGuard-AI is a machine learning project that detects phishing websites using URL-based features. It explores both **centralized machine learning** and **federated learning**, comparing performance when client data is distributed in IID and Non-IID ways.

The project aims to study how federated learning can support collaborative phishing detection while keeping the original data distributed across simulated clients.

## Objectives
- Detect legitimate and phishing websites using machine learning.
- Build a centralized Random Forest baseline.
- Simulate federated learning using multiple clients and Federated Averaging (FedAvg).
- Compare federated learning performance under IID and Non-IID data distributions.
- Evaluate models using accuracy, precision, recall, and F1-score.

## Repository Contents
| File | Description |
|---|---|
| `PhishGuard.py` | Centralized phishing detection model using Random Forest |
| `Federated_Learning_Research.ipynb` | Notebook containing the federated learning experiments |
| `dataset_B_05_2020.csv` | Dataset used for phishing detection |
| `base Research paper (2).pdf` | Research paper used as the project's base/reference |
| `PhishGuard_AI_Presentation.pptx` | Project presentation |

## Technologies Used
- Python
- Pandas
- Scikit-learn
- NumPy
- Matplotlib
- Google Colab / Jupyter Notebook
- Machine Learning
- Federated Learning (FedAvg)

## Dataset
The project uses `dataset_B_05_2020.csv`, which contains URL-based features and labels identifying websites as legitimate or phishing.

The URL column is excluded from model training, and the remaining features are used to train and evaluate the models.

## Models and Experiments

### 1. Centralized Model
A Random Forest classifier is trained on the dataset as a centralized baseline.

### 2. Federated Learning
Federated learning is simulated using multiple clients. Each client trains locally, and model updates are combined using **Federated Averaging (FedAvg)**.

Two data-distribution settings are studied:
- **IID:** Data is distributed so clients have similar class distributions.
- **Non-IID:** Data is distributed unevenly across clients.

## Evaluation Metrics
Models are evaluated using:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

## How to Run

### Run the centralized model
1. Download or clone this repository.
2. Install the required libraries:
   ```bash
   pip install pandas scikit-learn
   ```
3. Place `dataset_B_05_2020.csv` in the same folder as `PhishGuard.py`.
4. Run:
   ```bash
   python PhishGuard.py
   ```

### Run the federated learning notebook
1. Open `Federated_Learning_Research.ipynb` in Google Colab or Jupyter Notebook.
2. Upload `dataset_B_05_2020.csv` when prompted, or update the dataset path.
3. Run the notebook cells in order.

## Research Scope and Limitations
This project uses a **simulated federated learning environment**. It compares model performance under different client data distributions; it does not implement production-grade secure aggregation or differential privacy.

## Authors
- Rudraksh Singh Bargali
- Vedansh Sharma

## Project
CSET225 — Intelligent Model Design using AI
