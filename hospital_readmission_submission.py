# ============================================================
# MONTH 1 PROJECT
# HEART FAILURE 30-DAY READMISSION PREDICTION
#
# Required models:
#   1. Logistic Regression
#   2. K-Nearest Neighbors (KNN)
#   3. Decision Tree


import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve
)


# ============================================================
# 1. SETTINGS
# ============================================================

DATA_FILE = "dataset_12000_records.csv"
OUTPUT_DIR = "outputs"

RANDOM_STATE = 42
TEST_SIZE = 0.20

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. LOAD DATASET
# ============================================================

df = pd.read_csv(DATA_FILE)

print("=" * 70)
print("HEART FAILURE 30-DAY READMISSION PREDICTION")
print("=" * 70)

print("\nDataset shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nTotal missing values:", df.isnull().sum().sum())

print("\nDuplicate rows:", df.duplicated().sum())

print("\nDescriptive statistics:")
print(df.describe(include="all").transpose())

print("\nTarget distribution:")
print(df["Readmitted_30_Days"].value_counts())

print("\nTarget percentages:")
print(
    df["Readmitted_30_Days"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)


# ============================================================
# 3. DATA QUALITY + TARGET + LEAKAGE REVIEW
# ============================================================

# Remove duplicate rows
df = df.drop_duplicates().copy()

TARGET = "Readmitted_30_Days"

# Patient_ID is an identifier and should not be used as a predictor.
EXCLUDED_COLUMNS = []

if "Patient_ID" in df.columns:
    EXCLUDED_COLUMNS.append("Patient_ID")

X = df.drop(columns=[TARGET] + EXCLUDED_COLUMNS)
y = df[TARGET]

print("\nTarget column:", TARGET)
print("Target encoding:", sorted(y.unique().tolist()))
print("Excluded columns:", EXCLUDED_COLUMNS)

print("\nPredictor columns:")
print(X.columns.tolist())

print("\nLEAKAGE REVIEW:")
print("- Patient_ID is excluded because it is only an identifier.")
print("- Other predictors are retained because their schema alone does")
print("  not establish whether they were collected before or after the")
print("  prediction point.")
print("- For real deployment, confirm that every retained feature is")
print("  available at the time the readmission-risk prediction is made.")


# Save a short data-quality report
with open(
    os.path.join(OUTPUT_DIR, "data_quality_report.txt"),
    "w",
    encoding="utf-8"
) as f:

    f.write("HEART FAILURE READMISSION DATA QUALITY REPORT\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Original shape: {df.shape}\n")
    f.write(f"Target: {TARGET}\n")
    f.write(f"Target values: {sorted(y.unique().tolist())}\n")
    f.write(f"Duplicate rows after cleaning: {df.duplicated().sum()}\n")
    f.write(f"Total missing values: {df.isnull().sum().sum()}\n")
    f.write(f"Excluded columns: {EXCLUDED_COLUMNS}\n\n")

    f.write("Target distribution:\n")
    f.write(str(y.value_counts()) + "\n\n")

    f.write("Target percentage:\n")
    f.write(
        str(
            y.value_counts(normalize=True)
            .mul(100)
            .round(2)
        )
    )


# ============================================================
# 4. TARGET DISTRIBUTION VISUALIZATION
# ============================================================

target_counts = y.value_counts().sort_index()
target_percentages = (
    y.value_counts(normalize=True)
    .sort_index()
    .mul(100)
)

plt.figure(figsize=(7, 5))

bars = plt.bar(
    ["Not Readmitted (0)", "Readmitted (1)"],
    target_counts.values
)

plt.title("30-Day Readmission Target Distribution")
plt.xlabel("Target Class")
plt.ylabel("Number of Patients")

for bar, count, percentage in zip(
    bars,
    target_counts.values,
    target_percentages.values
):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{count}\n({percentage:.1f}%)",
        ha="center",
        va="bottom"
    )

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "target_distribution.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 5. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print("\nTrain shape:", X_train.shape)
print("Test shape :", X_test.shape)

print("\nTraining target distribution:")
print(y_train.value_counts())

print("\nTesting target distribution:")
print(y_test.value_counts())


# ============================================================
# 6. IDENTIFY NUMERICAL AND CATEGORICAL FEATURES
# ============================================================

numerical_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()

print("\nNumerical features:", len(numerical_features))
print(numerical_features)

print("\nCategorical features:", len(categorical_features))
print(categorical_features)


# ============================================================
# 7. PREPROCESSING
# ============================================================
#
# Numeric:
#   median imputation -> standard scaling
#
# Categorical:
#   most-frequent imputation -> one-hot encoding
#
# The ColumnTransformer is placed INSIDE each Pipeline.
# Therefore preprocessing is fitted only on training data.
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(strategy="median")
                    ),
                    (
                        "scaler",
                        StandardScaler()
                    )
                ]
            ),
            numerical_features
        ),
        (
            "categorical",
            Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(strategy="most_frequent")
                    ),
                    (
                        "encoder",
                        OneHotEncoder(
                            handle_unknown="ignore"
                        )
                    )
                ]
            ),
            categorical_features
        )
    ]
)


# ============================================================
# 8. DEFINE THE THREE REQUIRED BASELINE MODELS
# ============================================================
#
# Baseline models are intentionally evaluated WITHOUT
# class balancing first.
#
# Class balancing is handled separately in the extension.
# ============================================================

models = {
    "Logistic Regression": LogisticRegression(
        C=1.0,
        max_iter=2000
    ),

    "KNN": KNeighborsClassifier(
        n_neighbors=5
    ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=RANDOM_STATE
    )
}


# ============================================================
# 9. TRAIN + EVALUATE ALL THREE MODELS
# ============================================================

trained_models = {}
all_results = []

for name, model in models.items():

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    pipeline.fit(X_train, y_train)

    trained_models[name] = pipeline

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    for split_name, X_split, y_split in [
        ("Train", X_train, y_train),
        ("Test", X_test, y_test)
    ]:

        predictions = pipeline.predict(X_split)
        probabilities = pipeline.predict_proba(X_split)[:, 1]

        accuracy = accuracy_score(
            y_split,
            predictions
        )

        precision = precision_score(
            y_split,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y_split,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            y_split,
            predictions,
            zero_division=0
        )

        roc_auc = roc_auc_score(
            y_split,
            probabilities
        )

        all_results.append({
            "Model": name,
            "Split": split_name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "ROC_AUC": roc_auc
        })

        print(
            f"{split_name:5s} | "
            f"Accuracy={accuracy:.4f} | "
            f"Precision={precision:.4f} | "
            f"Recall={recall:.4f} | "
            f"F1={f1:.4f} | "
            f"ROC-AUC={roc_auc:.4f}"
        )


# ============================================================
# 10. CREATE FINAL MODEL COMPARISON TABLE
# ============================================================

all_results_df = pd.DataFrame(all_results)

train_results = (
    all_results_df[
        all_results_df["Split"] == "Train"
    ]
    .drop(columns=["Split"])
    .rename(
        columns={
            "Accuracy": "Train_Accuracy",
            "Precision": "Train_Precision",
            "Recall": "Train_Recall",
            "F1": "Train_F1",
            "ROC_AUC": "Train_ROC_AUC"
        }
    )
)

test_results = (
    all_results_df[
        all_results_df["Split"] == "Test"
    ]
    .drop(columns=["Split"])
    .rename(
        columns={
            "Accuracy": "Test_Accuracy",
            "Precision": "Test_Precision",
            "Recall": "Test_Recall",
            "F1": "Test_F1",
            "ROC_AUC": "Test_ROC_AUC"
        }
    )
)

comparison = train_results.merge(
    test_results,
    on="Model"
)


# ============================================================
# 11. FIT DIAGNOSIS
# ============================================================

def diagnose_fit(row):

    accuracy_gap = (
        row["Train_Accuracy"]
        - row["Test_Accuracy"]
    )

    if (
        accuracy_gap >= 0.10
        and row["Train_Accuracy"] > 0.90
    ):
        return "Likely overfitting"

    if (
        row["Train_Accuracy"] < 0.80
        and row["Test_Accuracy"] < 0.80
    ):
        return "Possible underfitting"

    return "Reasonable generalization"


comparison["Accuracy_Gap"] = (
    comparison["Train_Accuracy"]
    - comparison["Test_Accuracy"]
)

comparison["Fit_Diagnosis"] = comparison.apply(
    diagnose_fit,
    axis=1
)

comparison.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "model_comparison.csv"
    ),
    index=False
)

print("\n" + "=" * 70)
print("FINAL MODEL COMPARISON")
print("=" * 70)

print(
    comparison[
        [
            "Model",
            "Train_Accuracy",
            "Test_Accuracy",
            "Test_Precision",
            "Test_Recall",
            "Test_F1",
            "Test_ROC_AUC",
            "Accuracy_Gap",
            "Fit_Diagnosis"
        ]
    ].to_string(index=False)
)


# ============================================================
# 12. CONFUSION MATRIX FOR EVERY MODEL
# ============================================================

for name, pipeline in trained_models.items():

    predictions = pipeline.predict(X_test)

    cm = confusion_matrix(
        y_test,
        predictions
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            "Not Readmitted",
            "Readmitted"
        ]
    )

    display.plot()

    plt.title(
        f"Confusion Matrix - {name}"
    )

    plt.tight_layout()

    safe_name = (
        name.lower()
        .replace(" ", "_")
    )

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            f"confusion_matrix_{safe_name}.png"
        ),
        dpi=300
    )

    plt.close()

    print(
        f"\n{name} confusion matrix:"
    )
    print(cm)


# ============================================================
# 13. ROC CURVES FOR ALL THREE MODELS
# ============================================================

plt.figure(figsize=(8, 6))

for name, pipeline in trained_models.items():

    probabilities = pipeline.predict_proba(
        X_test
    )[:, 1]

    fpr, tpr, _ = roc_curve(
        y_test,
        probabilities
    )

    auc_value = roc_auc_score(
        y_test,
        probabilities
    )

    plt.plot(
        fpr,
        tpr,
        label=f"{name} (AUC={auc_value:.3f})"
    )

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves - All Three Models")
plt.legend()
plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "roc_curves_all_models.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 14. METRIC COMPARISON CHART
# ============================================================

test_plot = comparison[
    [
        "Model",
        "Test_Accuracy",
        "Test_Precision",
        "Test_Recall",
        "Test_F1",
        "Test_ROC_AUC"
    ]
].copy()

test_plot = test_plot.set_index("Model")

test_plot.columns = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "ROC-AUC"
]

test_plot.plot(
    kind="bar",
    figsize=(11, 6)
)

plt.title("Test-Set Metric Comparison")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.legend()
plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "metric_comparison.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 15. TRAIN VS TEST PERFORMANCE
# ============================================================

train_test_plot = comparison.set_index("Model")[
    ["Train_Accuracy", "Test_Accuracy"]
]

train_test_plot.columns = [
    "Train Accuracy",
    "Test Accuracy"
]

train_test_plot.plot(
    kind="bar",
    figsize=(9, 5)
)

plt.title("Training vs Test Accuracy")
plt.ylabel("Accuracy")
plt.ylim(0, 1.05)
plt.xticks(rotation=0)
plt.legend()
plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "train_vs_test_accuracy.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 16. KNN HYPERPARAMETER EXPERIMENT
# ============================================================
#
# k is selected using 5-fold stratified cross-validation
# on TRAINING DATA ONLY.
#
# This avoids repeatedly tuning on the final test set.
# ============================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)

k_values = [3, 5, 7, 9, 11, 15]

knn_experiment = []

for k in k_values:

    knn_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                KNeighborsClassifier(
                    n_neighbors=k
                )
            )
        ]
    )

    cv_scores = cross_val_score(
        knn_pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1
    )

    knn_experiment.append({
        "k": k,
        "Mean_CV_ROC_AUC": cv_scores.mean(),
        "Std_CV_ROC_AUC": cv_scores.std()
    })

knn_experiment_df = pd.DataFrame(
    knn_experiment
)

knn_experiment_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "knn_k_experiment.csv"
    ),
    index=False
)

best_k = int(
    knn_experiment_df.loc[
        knn_experiment_df["Mean_CV_ROC_AUC"].idxmax(),
        "k"
    ]
)

print("\nKNN hyperparameter experiment:")
print(knn_experiment_df.to_string(index=False))

print(
    f"\nSelected KNN k from 5-fold CV: {best_k}"
)


# ============================================================
# 17. LOGISTIC REGRESSION REGULARIZATION EXPERIMENT
# ============================================================

c_values = [0.01, 0.1, 1, 10, 100]

logistic_experiment = []

for c in c_values:

    lr_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                LogisticRegression(
                    C=c,
                    max_iter=2000
                )
            )
        ]
    )

    cv_scores = cross_val_score(
        lr_pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1
    )

    logistic_experiment.append({
        "C": c,
        "Mean_CV_ROC_AUC": cv_scores.mean(),
        "Std_CV_ROC_AUC": cv_scores.std()
    })

logistic_experiment_df = pd.DataFrame(
    logistic_experiment
)

logistic_experiment_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "logistic_C_experiment.csv"
    ),
    index=False
)

best_C = float(
    logistic_experiment_df.loc[
        logistic_experiment_df["Mean_CV_ROC_AUC"].idxmax(),
        "C"
    ]
)

print("\nLogistic Regression C experiment:")
print(
    logistic_experiment_df.to_string(
        index=False
    )
)

print(
    f"\nSelected Logistic Regression C from 5-fold CV: {best_C}"
)


# ============================================================
# 18. CLASS-IMBALANCE EXTENSION
# ============================================================
#
# Strategy:
#   class_weight="balanced"
#
# Compare Logistic Regression WITHOUT balancing vs WITH
# balancing using the SAME train/test split.
# ============================================================

unbalanced_lr = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            LogisticRegression(
                C=best_C,
                max_iter=2000
            )
        )
    ]
)

balanced_lr = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            LogisticRegression(
                C=best_C,
                max_iter=2000,
                class_weight="balanced"
            )
        )
    ]
)

unbalanced_lr.fit(X_train, y_train)
balanced_lr.fit(X_train, y_train)

imbalance_results = []

for name, model in [
    ("Without class balancing", unbalanced_lr),
    ("With class_weight='balanced'", balanced_lr)
]:

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    imbalance_results.append({
        "Strategy": name,
        "Accuracy": accuracy_score(
            y_test,
            predictions
        ),
        "Precision": precision_score(
            y_test,
            predictions,
            zero_division=0
        ),
        "Recall": recall_score(
            y_test,
            predictions,
            zero_division=0
        ),
        "F1": f1_score(
            y_test,
            predictions,
            zero_division=0
        ),
        "ROC_AUC": roc_auc_score(
            y_test,
            probabilities
        )
    })

imbalance_df = pd.DataFrame(
    imbalance_results
)

imbalance_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "imbalance_comparison.csv"
    ),
    index=False
)

print("\n" + "=" * 70)
print("CLASS IMBALANCE EXTENSION")
print("=" * 70)
print(
    imbalance_df.to_string(index=False)
)


# ============================================================
# 19. FINAL MODEL
# ============================================================
#
# Select the model with the highest TEST ROC-AUC for reporting.
#
# Note:
# This is a comparison criterion for the assignment. In a real
# clinical deployment, threshold choice and the consequences of
# false negatives/false positives must also be considered.
# ============================================================

best_model_name = comparison.loc[
    comparison["Test_ROC_AUC"].idxmax(),
    "Model"
]

final_model = trained_models[
    best_model_name
]

print("\n" + "=" * 70)
print("FINAL MODEL")
print("=" * 70)

print("Selected model:", best_model_name)

selected_row = comparison[
    comparison["Model"] == best_model_name
].iloc[0]

print(
    f"Test Accuracy : {selected_row['Test_Accuracy']:.4f}"
)
print(
    f"Test Precision: {selected_row['Test_Precision']:.4f}"
)
print(
    f"Test Recall   : {selected_row['Test_Recall']:.4f}"
)
print(
    f"Test F1       : {selected_row['Test_F1']:.4f}"
)
print(
    f"Test ROC-AUC  : {selected_row['Test_ROC_AUC']:.4f}"
)
print(
    "Fit diagnosis :",
    selected_row["Fit_Diagnosis"]
)


# ============================================================
# 20. SAVE FINAL MODEL
# ============================================================

joblib.dump(
    final_model,
    os.path.join(
        OUTPUT_DIR,
        "hospital_readmission_model.pkl"
    )
)


# ============================================================
# 21. SAVE FINAL SUMMARY
# ============================================================

summary = f"""
HEART FAILURE 30-DAY READMISSION PREDICTION
============================================

Dataset shape:
{df.shape}

Target:
{TARGET}

Target distribution:
{y.value_counts().to_dict()}

Models:
- Logistic Regression
- KNN
- Decision Tree

Selected final model:
{best_model_name}

Selection metric:
Test ROC-AUC

Required metrics:
Accuracy, Precision, Recall, F1-score, ROC-AUC

Class imbalance strategy:
class_weight='balanced' for Logistic Regression

KNN k selected by 5-fold CV:
{best_k}

Logistic Regression C selected by 5-fold CV:
{best_C}

Final model test results:
Accuracy  = {selected_row['Test_Accuracy']:.4f}
Precision = {selected_row['Test_Precision']:.4f}
Recall    = {selected_row['Test_Recall']:.4f}
F1        = {selected_row['Test_F1']:.4f}
ROC-AUC   = {selected_row['Test_ROC_AUC']:.4f}

Fit diagnosis:
{selected_row['Fit_Diagnosis']}
"""

with open(
    os.path.join(
        OUTPUT_DIR,
        "final_summary.txt"
    ),
    "w",
    encoding="utf-8"
) as f:
    f.write(summary)


# ============================================================
# 22. FINISHED
# ============================================================

print("\n" + "=" * 70)
print("PROJECT COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nOutputs saved in:", OUTPUT_DIR)

print("\nGenerated files:")
for filename in sorted(os.listdir(OUTPUT_DIR)):
    print("-", filename)
