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

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 89.50% | 85.24% | 78.61% | 81.79% | 95.62% |
| KNN | 84.17% | 82.32% | 60.14% | 69.50% | 88.78% |
| Decision Tree | 78.83% | 64.17% | 66.67% | 65.40% | 75.36% |

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