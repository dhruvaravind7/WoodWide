import json
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import time
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import addcopyfighandler

np.set_printoptions(precision=4, suppress=True)

from memray import Tracker
from sklearn.calibration import calibration_curve
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    hamming_loss,
    jaccard_score,
    log_loss,
    matthews_corrcoef,
    multilabel_confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)

BASE = "/Users/dhruvaravind/Desktop/Work/WoodWide/Model_Testing"
DIR = "Forest_Cover"
TARGET = "Cover_Type"
RUN_NAME = "forest"

# Loading the training and testing data
train = pd.read_csv(f"{BASE}/{DIR}/train.csv")
test = pd.read_csv(f"{BASE}/{DIR}/test.csv")
X_train = train.drop(columns=[f"{TARGET}"])
y_train = train[f"{TARGET}"]
X_test = test.drop(columns=[f"{TARGET}"])
y_test = test[f"{TARGET}"]

cat_cols = X_train.select_dtypes(exclude="number").columns.tolist()

for col in cat_cols:
    X_train[col] = X_train[col].astype("category")
    X_test[col] = X_test[col].astype("category")

# BINARY-CLASSIFICATION ROLLBACK: uncomment for a two-class target.
# scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

# The pipeline that the model uses. It first preprocesses the data and then uses the model provided.
# clf = Pipeline([
#     ("model", LGBMClassifier(
#         n_estimators=100,
#         max_depth=6,
#         learning_rate=0.1,
#         scale_pos_weight=scale_pos_weight,
#         metric="binary_logloss",
#         random_state=42
#     ))
# ])

clf = Pipeline([
    ("model", LGBMClassifier(
        # MULTICLASS CHANGE: LGBMClassifier infers the "multiclass" objective
        # automatically from the number of distinct labels in y_train, so no
        # explicit objective/num_class is needed like with XGBoost -- only the
        # eval metric name and the binary-only scale_pos_weight need to change.
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        metric="multi_logloss",
        random_state=42
    ))
])

# Trains the model using the training data
print("Training...\n")
training_start = time.time()

clf.fit(X_train, y_train)

print("Testing...\n")
# Running the test-set tests
testing_start = time.time()

# BINARY-CLASSIFICATION ROLLBACK:
# test_probs = clf.predict_proba(X_test)[:, 1]
# test_preds = (test_probs >= 0.5).astype(int)

# MULTICLASS CHANGE: keep both fitting and prediction inside the memory profile.
test_probs = clf.predict_proba(X_test)
test_preds = clf.predict(X_test)

class_labels = clf.named_steps["model"].classes_
y_test_binarized = label_binarize(y_test, classes=class_labels)
print("Predicted classes (first 5):\n", test_preds[:5])
print("Class probabilities (first 5):\n", test_probs[:5])

# BINARY-CLASSIFICATION ROLLBACK:
# print("\nROC-AUC Score:\n", roc_auc_score(y_test, test_probs), "\n")
# print("PR-AUC Score:\n", average_precision_score(y_test, test_probs), "\n")

# MULTICLASS CHANGE: aggregate metrics show both overall and minority-class performance.
print(f"\nAccuracy Score:\n{accuracy_score(y_test, test_preds):.4f}\n")
print(f"Balanced Accuracy Score:\n{balanced_accuracy_score(y_test, test_preds):.4f}\n")
print(f"Hamming Loss:\n{hamming_loss(y_test, test_preds):.4f}\n")

print(f"Macro F1 Score:\n{f1_score(y_test, test_preds, average='macro', zero_division=0):.4f}\n")
print(f"Macro Precision Score:\n{precision_score(y_test, test_preds, average='macro', zero_division=0):.4f}\n")
print(f"Macro Recall Score:\n{recall_score(y_test, test_preds, average='macro', zero_division=0):.4f}\n")
print(f"Macro Jaccard Score:\n{jaccard_score(y_test, test_preds, average='macro', zero_division=0):.4f}\n")

print(f"Macro ROC-AUC (one-vs-rest):\n{roc_auc_score(y_test, test_probs, multi_class='ovr', average='macro'):.4f}\n")
print(f"Weighted ROC-AUC (one-vs-rest):\n{roc_auc_score(y_test, test_probs, multi_class='ovr', average='weighted'):.4f}\n")
print(f"Macro PR-AUC:\n{average_precision_score(y_test_binarized, test_probs, average='macro'):.4f}\n")
print(f"Weighted PR-AUC:\n{average_precision_score(y_test_binarized, test_probs, average='weighted'):.4f}\n")
print(f"Multiclass Log Loss:\n{log_loss(y_test, test_probs, labels=class_labels):.4f}\n")

# BINARY-CLASSIFICATION ROLLBACK:
# print("Matthews Correlation Coefficient:\n", matthews_corrcoef(y_test, test_preds), "\n")
# print("Cohen's Kappa Score:\n", cohen_kappa_score(y_test, test_preds), "\n")
# print("Classification Report:\n", classification_report(y_test, test_preds, digits=4))
# print("Confusion Matrix:\n", confusion_matrix(y_test, test_preds), "\n")

# MULTICLASS CHANGE: these metrics apply to both binary and multiclass predictions.
print(f"Matthews Correlation Coefficient:\n{matthews_corrcoef(y_test, test_preds):.4f}\n")
print(f"Cohen's Kappa Score:\n{cohen_kappa_score(y_test, test_preds):.4f}\n")
print("Classification Report:\n", classification_report(y_test, test_preds, digits=4, zero_division=0))
print("Confusion Matrix:\n", confusion_matrix(y_test, test_preds, labels=class_labels), "\n")
print("One-vs-Rest Confusion Matrices:\n", multilabel_confusion_matrix(y_test, test_preds, labels=class_labels), "\n")

print(f"Training time taken: {testing_start - training_start:.4f} seconds\n")
print(f"Testing time taken: {time.time() - testing_start:.4f} seconds\n")
print(f"Total time taken: {time.time() - training_start:.4f} seconds\n")

# BINARY-CLASSIFICATION ROLLBACK:
# true_prob, pred_prob = calibration_curve(y_test, test_probs, n_bins=10, strategy="quantile")
# brier = brier_score_loss(y_test, test_probs)
# print(f"Brier Score Loss: {brier:.4f}")

# MULTICLASS CHANGE: compute Brier score and calibration one-vs-rest for every class.
brier = brier_score_loss(y_test, test_probs, labels=class_labels, scale_by_half=False)
print(f"Multiclass Brier Score Loss: {brier:.4f}")

# RESULTS JSON: preserves other model entries so this file can become the
# dataset-level source for a later Google Sheets export.
results_path = f"{BASE}/{DIR}/results.json"
try:
    with open(results_path, "r", encoding="utf-8") as results_file:
        results = json.load(results_file)
except (FileNotFoundError, json.JSONDecodeError):
    results = {
        "dataset": DIR,
        "target": TARGET,
        "task_type": "multiclass",
        "models": {},
    }

training_time = testing_start - training_start
testing_time = time.time() - testing_start
total_time = time.time() - training_start
report = classification_report(y_test, test_preds, digits=4, zero_division=0, output_dict=True)
rounded_report = {
    label: (
        round(float(scores), 4)
        if isinstance(scores, (float, int))
        else {
            metric: int(value) if metric == "support" else round(float(value), 4)
            for metric, value in scores.items()
        }
    )
    for label, scores in report.items()
}

results["dataset"] = DIR
results["target"] = TARGET
results["task_type"] = "multiclass"
results.setdefault("models", {})
results["models"]["LightGBM"] = {
    "run_name": RUN_NAME,
    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    "n_train_rows": int(len(y_train)),
    "n_test_rows": int(len(y_test)),
    "class_labels": class_labels.tolist(),
    "metrics": {
        "accuracy": round(accuracy_score(y_test, test_preds), 4),
        "balanced_accuracy": round(balanced_accuracy_score(y_test, test_preds), 4),
        "hamming_loss": round(hamming_loss(y_test, test_preds), 4),
        "f1_macro": round(f1_score(y_test, test_preds, average="macro", zero_division=0), 4),
        "precision_macro": round(precision_score(y_test, test_preds, average="macro", zero_division=0), 4),
        "recall_macro": round(recall_score(y_test, test_preds, average="macro", zero_division=0), 4),
        "jaccard_macro": round(jaccard_score(y_test, test_preds, average="macro", zero_division=0), 4),
        "roc_auc_ovr_macro": round(roc_auc_score(y_test, test_probs, multi_class="ovr", average="macro"), 4),
        "roc_auc_ovr_weighted": round(roc_auc_score(y_test, test_probs, multi_class="ovr", average="weighted"), 4),
        "pr_auc_macro": round(average_precision_score(y_test_binarized, test_probs, average="macro"), 4),
        "pr_auc_weighted": round(average_precision_score(y_test_binarized, test_probs, average="weighted"), 4),
        "log_loss": round(log_loss(y_test, test_probs, labels=class_labels), 4),
        "matthews_correlation_coefficient": round(matthews_corrcoef(y_test, test_preds), 4),
        "cohens_kappa": round(cohen_kappa_score(y_test, test_preds), 4),
        "multiclass_brier_score_loss": round(brier, 4),
    },
    "classification_report": rounded_report,
    "confusion_matrix": confusion_matrix(y_test, test_preds, labels=class_labels).tolist(),
    "one_vs_rest_confusion_matrices": multilabel_confusion_matrix(y_test, test_preds, labels=class_labels).tolist(),
    "timing_seconds": {
        "training": round(training_time, 4),
        "testing": round(testing_time, 4),
        "total": round(total_time, 4),
    },
}

with open(results_path, "w", encoding="utf-8") as results_file:
    json.dump(results, results_file, indent=2, allow_nan=False)
print(f"Results saved to: {results_path}")

plt.figure(figsize=(8, 6))

# Plot the ideal baseline (perfect calibration)
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly Calibrated")

# BINARY-CLASSIFICATION ROLLBACK:
# plt.plot(pred_prob, true_prob, marker="o", color="blue", label="LightGBM")

# MULTICLASS CHANGE: plot a one-vs-rest calibration curve for each class.
for class_index, class_label in enumerate(class_labels):
    true_prob, pred_prob = calibration_curve(
        y_test_binarized[:, class_index], test_probs[:, class_index], n_bins=10, strategy="quantile"
    )
    plt.plot(pred_prob, true_prob, marker="o", label=f"LightGBM class {class_label}")

# Formatting the visual graph
plt.xlabel("Mean Predicted Probability")
plt.ylabel("Fraction of Positives (Actual Frequency)")
plt.title("LightGBM Calibration Curve")
plt.legend(loc="upper left")
plt.grid(True)

# Display the plot
plt.show()
