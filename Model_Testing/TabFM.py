import sys
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
from tqdm import tqdm

# The tabfm checkout lives under kaggle_items/ and its modules import each other
# by their full path from the repo root, so the root has to be on sys.path.
# (The editable-install .pth still points at the pre-move location and is dead.)
sys.path.insert(0, "/Users/dhruvaravind/Desktop/Work/WoodWide")

# tabfm prefers its JAX backend and falls back to PyTorch only on ImportError.
# jax 0.10 needs numpy >= 2 and this env is on 1.24, so importing it raises an
# AttributeError the fallback doesn't catch. Poisoning the module entry turns
# that into the ImportError tabfm expects, forcing the PyTorch backend.
sys.modules["jax"] = None

import Model_Testing.kaggle_items.tabfm.tabfm as tabfm

BASE = "/Users/dhruvaravind/Desktop/Work/WoodWide/Model_Testing"
DIR = "Forest_Cover"
TARGET = "Cover_Type"
RUN_NAME = "forest"

# Local copy of the google/tabfm-1.0.0-pytorch weights. Without this, load()
# falls through to a huggingface_hub download.
WEIGHTS = f"{BASE}/kaggle_items/tabfm_weights"

# Loading the training and testing data
train = pd.read_csv(f"{BASE}/{DIR}/train.csv")
test = pd.read_csv(f"{BASE}/{DIR}/test.csv")

# Like TabPFN, TabFM is in-context: the training rows are the context, not
# something it fits parameters to. max_num_rows below caps what each ensemble
# member actually sees; this cap keeps the preprocessing pass bounded too.
TRAIN_SAMPLE_SIZE = 10000
if TRAIN_SAMPLE_SIZE is not None and len(train) > TRAIN_SAMPLE_SIZE:
    train = train.sample(n=TRAIN_SAMPLE_SIZE, random_state=42)

X_train = train.drop(columns=[TARGET])
y_train = train[TARGET]
X_test = test.drop(columns=[TARGET])
y_test = test[TARGET]

# Reading the checkpoint takes ~20s and is a one-time weight load rather
# than fitting, so the training timer starts after it. It stays inside the
# tracker because the resident weights are part of the memory footprint.
model = tabfm.tabfm_v1_0_0_pytorch.load(
    model_type="classification",
    checkpoint_path=WEIGHTS,
    device="mps",
)
# enable_nnls must be off whenever max_num_rows is set (the constructor
# rejects the combination), which also drops the NNLS-weighted blending
# from the "ensemble" preset.
clf = tabfm.TabFMClassifier.ensemble(
    model=model,
    max_num_rows=500,
    n_estimators=4,
    batch_size=8,
    enable_nnls=False,
)

print("Training...\n")
# MULTICLASS CHANGE: wraps fit + predict in a memray Tracker, matching the
# script convention (this import was previously unused -- no memory profile
# was being captured).

training_start = time.time()
clf.fit(X_train, y_train)

print("Testing...\n")
testing_start = time.time()
chunk_size = 5000
chunks = [X_test.iloc[i:i + chunk_size] for i in range(0, len(X_test), chunk_size)]

# BINARY-CLASSIFICATION ROLLBACK: uncomment for a two-class target.
# test_probs = np.concatenate(
#     [clf.predict_proba(chunk) for chunk in tqdm(chunks, desc="Testing", unit="chunk")],
#     axis=0,
# )[:, 1]
# test_preds = (test_probs >= 0.5).astype(int)

# MULTICLASS CHANGE: Forest_Cover has seven mutually exclusive classes.
# clf.classes_ is sorted (sklearn convention) and predict_proba columns
# are aligned to it, so argmax avoids a second forward pass through the
# ensemble.
test_probs = np.concatenate(
    [clf.predict_proba(chunk) for chunk in tqdm(chunks, desc="Testing", unit="chunk")],
    axis=0,
)
class_labels = np.asarray(clf.classes_)
test_preds = class_labels[np.argmax(test_probs, axis=1)]
y_test_binarized = label_binarize(y_test, classes=class_labels)

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

# MULTICLASS CHANGE: compute the unscaled multiclass Brier score.  Older
# scikit-learn releases only accept binary targets in brier_score_loss, so
# calculate the equivalent one-hot formulation directly.
y_test_one_hot = label_binarize(y_test, classes=class_labels)
brier = np.mean(np.sum((y_test_one_hot - test_probs) ** 2, axis=1))
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
results["models"]["TabFM"] = {
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
# plt.plot(pred_prob, true_prob, marker="o", color="blue", label="TabFM")

# MULTICLASS CHANGE: plot a one-vs-rest calibration curve for each class.
for class_index, class_label in enumerate(class_labels):
    true_prob, pred_prob = calibration_curve(
        y_test_binarized[:, class_index], test_probs[:, class_index], n_bins=10, strategy="quantile"
    )
    plt.plot(pred_prob, true_prob, marker="o", label=f"TabFM: class {class_label}")
    plt.plot(pred_prob, true_prob, marker="o", label=f"TabFM class {class_label}")

# Formatting the visual graph
plt.xlabel("Mean Predicted Probability")
plt.ylabel("Fraction of Positives (Actual Frequency)")
plt.title("TabFM Calibration Curve")
plt.legend(loc="upper left")
plt.grid(True)

# Display the plot
plt.show()
