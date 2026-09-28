# 🏥 30-Day Hospital Readmission Prediction

A machine learning classification project that predicts whether a patient is likely to be readmitted to the hospital within 30 days.

## 📌 Project Overview

Hospital readmissions are an important healthcare problem because they can increase healthcare costs and indicate the need for better patient care and follow-up.

This project develops and evaluates machine learning classification models to predict 30-day hospital readmission using patient and hospital-related data.

The project evaluates models using multiple metrics rather than relying only on accuracy:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Confusion Matrix

It also includes hyperparameter experiments, class imbalance analysis, and train-test performance comparison.

## 🎯 Objective

The main objective is to predict:

> Whether a patient will be readmitted to the hospital within 30 days.

### Target Variable

`Readmitted_30_Days`

### Problem Type

- Supervised Machine Learning
- Binary Classification
- Healthcare Prediction

## 📊 Dataset

The project uses approximately **12,000 patient records** with **36 columns**.

| Property | Value |
|---|---|
| Records | ~12,000 |
| Columns | 36 |
| Target | `Readmitted_30_Days` |
| Problem | Binary Classification |

Dataset file:

```text
dataset_12000_records.csv

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib

## 🤖 Machine Learning Models

The following models were implemented and evaluated:

### Logistic Regression

Used as a classification model and evaluated with different regularization values of `C`.

### K-Nearest Neighbors (KNN)

A distance-based classification algorithm. Different values of `K` were tested.

### Decision Tree

A tree-based classification model. Training and testing performance were compared to investigate overfitting.

## 🔬 Experiments

### KNN K-Value Experiment

Different values of `K` were tested.

```text
outputs/knn_k_experiment.csv
## 📈 Model Results

The models were evaluated on the test dataset.

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 89.50% | 85.24% | 78.61% | 81.79% | 95.62% |
| KNN | 84.17% | 82.32% | 60.14% | 69.50% | 88.78% |
| Decision Tree | 78.83% | 64.17% | 66.67% | 65.40% | 75.36% |

## 🔍 Train vs Test Performance

| Model | Train Accuracy | Test Accuracy | Accuracy Gap |
|---|---:|---:|---:|
| Logistic Regression | 89.46% | 89.50% | ~0.04% |
| KNN | 89.24% | 84.17% | 5.07% |
| Decision Tree | 100.00% | 78.83% | 21.17% |

The Decision Tree shows a substantially larger difference between training and testing accuracy, which is consistent with overfitting on this dataset.

## 📊 Visualizations

### Model Performance

![Model Performance](outputs/metric_comparison.png)

### ROC Curves

![ROC Curves](outputs/roc_curves_all_models.png)

### Target Distribution

![Target Distribution](outputs/target_distribution.png)

## 🧪 Data Quality Analysis

A data quality report was generated before model training.

```text
outputs/data_quality_report.txt

Heart-Failure-30-Day-Readmission-Prediction/
│
├── README.md
├── .gitignore
├── dataset_12000_records.csv
├── hospital_readmission_submission.py
│
└── outputs/
    ├── confusion_matrix_decision_tree.png
    ├── confusion_matrix_knn.png
    ├── confusion_matrix_logistic_regression.png
    ├── data_quality_report.txt
    ├── final_summary.txt
    ├── hospital_readmission_model.pkl
    ├── imbalance_comparison.csv
    ├── knn_k_experiment.csv
    ├── logistic_C_experiment.csv
    ├── metric_comparison.png
    ├── model_comparison.csv
    ├── roc_curves_all_models.png
    ├── target_distribution.png
    └── train_vs_test_accuracy.png

    📚 Learning Outcomes

This project provided practical experience with:

Binary classification
Data preprocessing
Train-test splitting
Logistic Regression
K-Nearest Neighbors
Decision Trees
Hyperparameter tuning
Class imbalance
Precision, Recall and F1-Score
ROC-AUC
Confusion matrices
ROC curves
Model comparison
Overfitting analysis
Model serialization
🚀 Future Improvements

Possible future improvements include:

Feature engineering
Cross-validation
More extensive hyperparameter tuning
Advanced class-imbalance handling
Ensemble models
Feature importance analysis
Explainable AI
FastAPI/Flask deployment
Interactive prediction dashboard

👩‍💻 Author

Maithili

Computer Engineering / Computer Science Student