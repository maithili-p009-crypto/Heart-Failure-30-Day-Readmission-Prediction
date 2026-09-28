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