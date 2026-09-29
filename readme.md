# 🏥 30-Day Hospital Readmission Prediction

A machine learning project that predicts whether a patient will be readmitted to the hospital within 30 days.

## 🎯 Objective

Build a binary classification model using patient and hospital-related data.

**Target:** `Readmitted_30_Days`

## 📊 Dataset

- ~12,000 patient records
- 36 columns
- Binary classification problem

Dataset:

`dataset_12000_records.csv`

## 🛠️ Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib

## 🤖 Models

The project evaluates:

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree

Experiments include KNN `K` values, Logistic Regression `C` values, and class imbalance analysis.

## 📈 Results

## Model Performance Comparison

| Model | Train Accuracy | Test Accuracy | Test Precision | Test Recall | Test F1 | Test ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8946 | 0.8950 | 0.8952 | 0.8950 | 0.8948 | 0.9506 |
| KNN | 0.8924 | 0.8417 | 0.8422 | 0.8417 | 0.8415 | 0.8977 |
| Decision Tree | 1.0000 | 0.7883 | 0.7900 | 0.7883 | 0.7877 | 0.8365 |
### Evaluation Metrics

Accuracy, Precision, Recall, F1-Score, ROC-AUC and Confusion Matrix were used to evaluate model performance.

## 📁 Project Structure

```text
Heart-Failure-30-Day-Readmission-Prediction/
│
├── README.md
├── .gitignore
├── dataset_12000_records.csv
├── hospital_readmission_submission.py
└── outputs/
    ├── model_comparison.csv
    ├── knn_k_experiment.csv
    ├── logistic_C_experiment.csv
    ├── imbalance_comparison.csv
    ├── metric_comparison.png
    ├── roc_curves_all_models.png
    └── hospital_readmission_model.pkl
