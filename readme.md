<img width="1118" height="918" alt="Screenshot 2026-09-29 at 23 32 20" src="https://github.com/user-attachments/assets/635979d1-ec69-40f9-a825-5896cde42f16" /># 🏥 30-Day Hospital Readmission Prediction

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
│
└── outputs/
    ├── confusion_matrix_decision_tree.png
    ├── confusion_matrix_knn.png
    ├── confusion_matrix_logistic_regression.png
    ├── data_quality_report.txt
    ├── decision_tree_depth_experiment.csv
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

## 📊 Visualizations

### Model Metric Comparison
<img width="1118" height="918" alt="Screenshot 2026-09-29 at 23 32 37" src="https://github.com/user-attachments/assets/2c9709ea-5b5e-4073-b4ab-f5383caa04b0" />

### ROC Curves
<img width="1495" height="918" alt="Screenshot 2026-09-29 at 23 31 48" src="https://github.com/user-attachments/assets/1dc58c55-2c3c-4978-9b25-f0478094284f" />

### Target Distribution
!<img width="1168" height="918" alt="Screenshot 2026-09-29 at 23 33 50" src="https://github.com/user-attachments/assets/97a05af4-7c0d-440f-8837-b1daf35d8192" />

### Training vs Testing Accuracy

<img width="1470" height="918" alt="Screenshot 2026-09-29 at 23 34 26" src="https://github.com/user-attachments/assets/4fc1e961-af67-45c3-9f3d-f52077b3d84c" />
