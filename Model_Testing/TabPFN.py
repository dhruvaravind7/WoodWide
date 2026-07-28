import os
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
from tabpfn import TabPFNClassifier
from tabpfn_extensions.interpretability.pdp import partial_dependence_plots
from tabpfn_extensions.interpretability.feature_selection import feature_selection
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

# TabPFN was pretrained on contexts of up to 10,000 rows. Past that it still runs
# (ignore_pretraining_limits), but it is outside the regime it was trained for.
TRAIN_SAMPLE_SIZE = 10000
if TRAIN_SAMPLE_SIZE is not None and len(train) > TRAIN_SAMPLE_SIZE:
    train = train.sample(n=TRAIN_SAMPLE_SIZE, random_state=42)

X_train = train.drop(columns=[TARGET])
y_train = train[TARGET]
X_test = test.drop(columns=[TARGET])
y_test = test[TARGET]

# Trains the model
training_start = time.time()
print("Training...")

# "auto" resolves to mps locally and cuda on Kaggle. CPU is refused above 1,000
# samples. TabPFN encodes the string columns itself, so no OrdinalEncoder needed.
model = TabPFNClassifier(device="auto", ignore_pretraining_limits=True, fit_mode="fit_with_cache")
model.fit(X_train, y_train)

testing_start = time.time()
print("Testing...")

# BINARY-CLASSIFICATION ROLLBACK: uncomment for a two-class target.
# test_probs = model.predict_proba(X_test)[:, 1]
# test_preds = (test_probs >= 0.5).astype(int)

# MULTICLASS CHANGE: TabPFN natively supports multiclass; predict_proba
# returns the full [n_rows, n_classes] matrix aligned to model.classes_.
# Deriving predictions via argmax avoids a second (expensive, in-context)
# forward pass through the model inside the memory-tracked block.
test_probs = model.predict_proba(X_test)
class_labels = model.classes_
test_preds = class_labels[np.argmax(test_probs, axis=1)]
y_test_binarized = label_binarize(y_test, classes=class_labels)

# Evaluate the model on the test set


# BINARY-CLASSIFICATION ROLLBACK:
# print("\nROC-AUC Score:\n", roc_auc_score(y_test, test_probs), "\n")
# print("PR-AUC Score:\n", average_precision_score(y_test, test_probs), "\n")

print("Predicted classes (first 5):\n", test_preds[:5])
print("Class probabilities (first 5):\n", test_probs[:5])

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
results["models"]["TabPFN"] = {
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
# plt.plot(pred_prob, true_prob, marker="o", color="blue", label="TabPFN")

# MULTICLASS CHANGE: plot a one-vs-rest calibration curve for each class.
for class_index, class_label in enumerate(class_labels):
    true_prob, pred_prob = calibration_curve(
        y_test_binarized[:, class_index], test_probs[:, class_index], n_bins=10, strategy="quantile"
    )
    plt.plot(pred_prob, true_prob, marker="o", label=f"TabPFN class {class_label}")

# Formatting the visual graph
plt.xlabel("Mean Predicted Probability")
plt.ylabel("Fraction of Positives (Actual Frequency)")
plt.title("TabPFN Calibration Curve")
plt.legend(loc="upper left")
plt.grid(True)

# Display the plot
plt.show()