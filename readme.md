# 🏥 30-Day Hospital Readmission Prediction

A machine learning project that predicts whether a patient will be readmitted to the hospital within 30 days.

## 🎯 Objective

Build a binary classification model using patient and hospital-related data.

**Target:** `Readmitted_30_Days`

## 📊 Dataset

- ~12,000 patient records
- 36 columns
- Binary classification problem

**Dataset:** `dataset_12000_records.csv`

## 🛠️ Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib

## 🤖 Models

The project evaluates the following machine learning models:

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree

Experiments include:

- KNN `K` values
- Logistic Regression `C` values
- Decision Tree depth
- Class imbalance analysis

## 📈 Results

### Model Performance Comparison

| Model | Train Accuracy | Test Accuracy | Test Precision | Test Recall | Test F1 | Test ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8946 | 0.8950 | 0.8952 | 0.8950 | 0.8948 | 0.9506 |
| KNN | 0.8924 | 0.8417 | 0.8422 | 0.8417 | 0.8415 | 0.8977 |
| Decision Tree | 1.0000 | 0.7883 | 0.7900 | 0.7883 | 0.7877 | 0.8365 |

### Evaluation Metrics

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Confusion Matrix

## 📊 Visualizations

### 1. Model Metric Comparison

![Model Metric Comparison](outputs/metric_comparison.png)

### 2. ROC Curves

![ROC Curves](outputs/roc_curves_all_models.png)

### 3. Target Distribution

![Target Distribution](outputs/target_distribution.png)

### 4. Training vs Testing Accuracy

![Train vs Test Accuracy](outputs/train_vs_test_accuracy.png)

### 5. Logistic Regression Confusion Matrix

![Logistic Regression Confusion Matrix](outputs/confusion_matrix_logistic_regression.png)

### 6. KNN Confusion Matrix

![KNN Confusion Matrix](outputs/confusion_matrix_knn.png)

### 7. Decision Tree Confusion Matrix

![Decision Tree Confusion Matrix](outputs/confusion_matrix_decision_tree.png)

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
```

## 🧪 Experiments

### KNN Hyperparameter Experiment

Different values of `K` were evaluated to study their effect on model performance.

**Output:** `outputs/knn_k_experiment.csv`

### Logistic Regression Hyperparameter Experiment

Different values of `C` were tested to analyze regularization effects.

**Output:** `outputs/logistic_C_experiment.csv`

### Decision Tree Depth Experiment

Different maximum tree depths were evaluated to study model complexity and generalization.

**Output:** `outputs/decision_tree_depth_experiment.csv`

### Class Imbalance Analysis

Different approaches were compared to understand the effect of class imbalance on model performance.

**Output:** `outputs/imbalance_comparison.csv`

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone <your-repository-url>
```

### 2. Navigate to the project directory

```bash
cd Heart-Failure-30-Day-Readmission-Prediction
```

### 3. Install required libraries

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

### 4. Run the Python program

```bash
python hospital_readmission_submission.py
```

## 📦 Outputs

The project generates:

- Model comparison results
- Hyperparameter experiment results
- Confusion matrices
- ROC curves
- Target distribution visualization
- Train vs test accuracy visualization
- Data quality report
- Final model summary
- Trained model `.pkl` file

## 🔍 Key Findings

- Logistic Regression achieved a test accuracy of **0.8950**.
- KNN achieved a test accuracy of **0.8417**.
- Decision Tree achieved a training accuracy of **1.0000** and test accuracy of **0.7883**.
- ROC-AUC was used along with accuracy, precision, recall, and F1-score for evaluation.
- Hyperparameter experiments were performed for KNN, Logistic Regression, and Decision Tree.
- Class imbalance was also analyzed as part of the evaluation.

## 🔮 Future Improvements

- Test additional machine learning algorithms.
- Perform more extensive hyperparameter tuning.
- Explore feature selection techniques.
- Investigate advanced class-imbalance handling techniques.
- Apply cross-validation.
- Deploy the trained model as a web application or API.

## 👩‍💻 Author

**Maithili Patil**
