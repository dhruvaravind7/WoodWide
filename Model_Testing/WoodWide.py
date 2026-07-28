import requests
import os
import time
import json
import io
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import addcopyfighandler

np.set_printoptions(precision=4, suppress=True)

from memray import Tracker
from sklearn.calibration import calibration_curve

from dotenv import load_dotenv
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
    matthews_corrcoef,
    multilabel_confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)

load_dotenv()

BASE = "/Users/dhruvaravind/Desktop/Work/WoodWide/Model_Testing"
DIR = "Forest_Cover"
TARGET = "Cover_Type"
RUN_NAME = "forest"

labels = pd.read_csv(f"{BASE}/{DIR}/test_labels.csv").squeeze()

api_key = os.getenv("WOODWIDE_API_KEY")
base_url = "https://api.woodwide.ai"
headers = {"Authorization": f"Bearer {api_key}"}

print("Uploading Dataset... \n")
prepare_response = requests.post(
    f"{base_url}/datasets/upload",
    headers={**headers, "Content-Type": "application/json"},
    json={
        "dataset_name": f"{DIR}",
        "file": {
            "filename": f"{BASE}/{DIR}/train.csv",
            "bytes": os.path.getsize(f"{BASE}/{DIR}/train.csv"),
            "content_type": "text/csv",
        },
    },
)
prepare = prepare_response.json()
print(prepare)
upload_url = prepare["upload"]["upload_url"]
dataset_version_id = prepare["version_id"]

with open(f"{BASE}/{DIR}/train.csv", "rb") as f:
    requests.put(upload_url, data=f, headers={"Content-Type": "text/csv"})

complete_response = requests.post(
    f"{base_url}/datasets/{dataset_version_id}/complete",
    headers=headers,
)

print(complete_response.json())
dataset_id = complete_response.json()["id"]

while True:
    dataset = requests.get(
        f"{base_url}/datasets/{dataset_id}", headers=headers
    ).json()
    if dataset["current_version"]["status"] == "ready":
        break
    time.sleep(5)

print("Dataset Uploaded!")
# Uploading a dataset and training a model
'''
# Upload training data
print("Uploading training data...")
with open(f"{BASE}/{DIR}/train.csv", "rb") as f:
    resp = requests.post(
        f"{base_url}/datasets",
        headers=headers,
        files={"file": (f"{BASE}/{DIR}/train.csv", f, "text/csv")},
        data={"dataset_name": "HIGGS Dataset"},
    )

dataset_id = resp.json()["dataset"]["id"]

print("Upload finished. Dataset ID:", dataset_id)
'''
# dataset_id = "ds_HZS4PM2T"
##########################################################################################################################################

print("Creating model...")
training_start = time.time()

# Train a prediction model
resp = requests.post(
    f"{base_url}/models/train",
    headers=headers,
    json={
        "model_name": f"{DIR} Model",
        "model_type": "prediction",
        "dataset_id": dataset_id,
        "label_column": TARGET,
    },
)

print(resp.status_code)
print(resp.json())
resp.raise_for_status()
model_id = resp.json()["id"]
print("Model Created. Model ID:", model_id)
print("Training model...")

# Wait for training
while True:
    model = requests.get(
        f"{base_url}/models/{model_id}", headers=headers
    ).json()
    if model["status"] == "ready":
        break
    elif model['status'] == "failed":
        print(model)
        break
    time.sleep(5)

print("Training finished. Model ID:", model_id)

#model_id = "mdl_TSCR8PKV"
##########################################################################################################################################
testing_start = time.time()
print("Running inference on new data...")

# Run inference on new data and get predictions
with Tracker(f"{DIR}/memory_files/{RUN_NAME}_ww_run.bin"):
    with open(f"{BASE}/{DIR}/test_features.csv", "rb") as f:
        resp = requests.post(
            f"{base_url}/models/{model_id}/infer",
            headers=headers,
            files={"file": (f"{BASE}/{DIR}/test_features.csv", f, "text/csv")},
            data={"output_type": "json"},
        )

    resp.raise_for_status()
    data = resp.json()['data']
    predictions = data["prediction"]

# MULTICLASS CHANGE: WoodWide's /infer response keys the probability column
# differently per task type -- "prediction_prob" (singular) for binary,
# "prediction_probs" (plural) for multiclass. Unlike XGBoost.py/Kumo.py,
# which get a full [n_rows, n_classes] matrix from predict_proba(), the
# multiclass column here only carries the probability of the PREDICTED
# class (top-1 confidence) -- the API does not expose a full per-class
# distribution. That rules out ROC-AUC/PR-AUC/log-loss and per-class
# calibration; the calibration curve below falls back to the standard
# top-label "confidence vs. accuracy" reliability diagram instead.
class_labels = np.sort(labels.unique())
is_multiclass = "prediction_probs" in data

if is_multiclass:
    confidence = np.asarray(data["prediction_probs"], dtype=float)

    print("Predicted classes (first 5):\n", predictions[:5])
    print("Top-label confidence (first 5):\n", confidence[:5])

    print(f"\nAccuracy Score:\n{accuracy_score(labels, predictions):.4f}\n")
    print(f"Balanced Accuracy Score:\n{balanced_accuracy_score(labels, predictions):.4f}\n")
    print(f"Hamming Loss:\n{hamming_loss(labels, predictions):.4f}\n")
    print(f"Macro F1 Score:\n{f1_score(labels, predictions, average='macro', zero_division=0):.4f}\n")
    print(f"Macro Precision Score:\n{precision_score(labels, predictions, average='macro', zero_division=0):.4f}\n")
    print(f"Macro Recall Score:\n{recall_score(labels, predictions, average='macro', zero_division=0):.4f}\n")
    print(f"Macro Jaccard Score:\n{jaccard_score(labels, predictions, average='macro', zero_division=0):.4f}\n")
    print(f"Matthews Correlation Coefficient:\n{matthews_corrcoef(labels, predictions):.4f}\n")
    print(f"Cohen's Kappa Score:\n{cohen_kappa_score(labels, predictions):.4f}\n")
    print("Classification Report:\n", classification_report(labels, predictions, digits=4, zero_division=0))
    print("Confusion Matrix:\n", confusion_matrix(labels, predictions, labels=class_labels), "\n")
    print("One-vs-Rest Confusion Matrices:\n", multilabel_confusion_matrix(labels, predictions, labels=class_labels), "\n")
else:
    pred_probs = [p if pred == 1 else 1 - p for pred, p in zip(predictions, data["prediction_prob"])]

    print("\nROC-AUC Score:\n", roc_auc_score(labels, pred_probs), "\n")
    print("PR-AUC Score:\n", average_precision_score(labels, pred_probs), "\n")
    print("Matthews Correlation Coefficient:\n", matthews_corrcoef(labels, predictions), "\n")
    print("Cohen's Kappa Score:\n", cohen_kappa_score(labels, predictions), "\n")
    print("Classification Report:\n", classification_report(labels, predictions, digits=4))
    print("Confusion Matrix:\n", confusion_matrix(labels, predictions), "\n")

print("Training time taken:", testing_start - training_start, " seconds\n")
print("Testing time taken:", time.time() - testing_start, " seconds\n")
print("Total time taken: ", time.time() - training_start, " seconds\n")

if is_multiclass:
    # MULTICLASS CHANGE: with only top-1 confidence available, "calibration"
    # here means confidence-vs-accuracy (Guo et al. reliability diagram),
    # not a per-class positive-rate curve.
    is_correct = (np.asarray(predictions) == np.asarray(labels)).astype(int)
    true_prob, pred_prob = calibration_curve(is_correct, confidence, n_bins=10, strategy="quantile")
    brier = brier_score_loss(is_correct, confidence)
    print(f"Top-Label Brier Score Loss: {brier:.4f}")
else:
    true_prob, pred_prob = calibration_curve(labels, pred_probs, n_bins=10, strategy="quantile")
    brier = brier_score_loss(labels, pred_probs)
    print(f"Brier Score Loss: {brier:.4f}")

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
        "task_type": "multiclass" if is_multiclass else "binary",
        "models": {},
    }

training_time = testing_start - training_start
testing_time = time.time() - testing_start
total_time = time.time() - training_start

report = classification_report(
    labels, predictions, digits=4, zero_division=0, output_dict=True
) if is_multiclass else classification_report(labels, predictions, digits=4, output_dict=True)
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
results["task_type"] = "multiclass" if is_multiclass else "binary"
results.setdefault("models", {})

if is_multiclass:
    results["models"]["WoodWide"] = {
        "run_name": RUN_NAME,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_test_rows": int(len(labels)),
        "class_labels": class_labels.tolist(),
        "metrics": {
            "accuracy": round(accuracy_score(labels, predictions), 4),
            "balanced_accuracy": round(balanced_accuracy_score(labels, predictions), 4),
            "hamming_loss": round(hamming_loss(labels, predictions), 4),
            "f1_macro": round(f1_score(labels, predictions, average="macro", zero_division=0), 4),
            "precision_macro": round(precision_score(labels, predictions, average="macro", zero_division=0), 4),
            "recall_macro": round(recall_score(labels, predictions, average="macro", zero_division=0), 4),
            "jaccard_macro": round(jaccard_score(labels, predictions, average="macro", zero_division=0), 4),
            "matthews_correlation_coefficient": round(matthews_corrcoef(labels, predictions), 4),
            "cohens_kappa": round(cohen_kappa_score(labels, predictions), 4),
            "top_label_brier_score_loss": round(brier, 4),
        },
        "classification_report": rounded_report,
        "confusion_matrix": confusion_matrix(labels, predictions, labels=class_labels).tolist(),
        "one_vs_rest_confusion_matrices": multilabel_confusion_matrix(labels, predictions, labels=class_labels).tolist(),
        "timing_seconds": {
            "training": round(training_time, 4),
            "testing": round(testing_time, 4),
            "total": round(total_time, 4),
        },
    }
else:
    results["models"]["WoodWide"] = {
        "run_name": RUN_NAME,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_test_rows": int(len(labels)),
        "metrics": {
            "roc_auc": round(roc_auc_score(labels, pred_probs), 4),
            "pr_auc": round(average_precision_score(labels, pred_probs), 4),
            "matthews_correlation_coefficient": round(matthews_corrcoef(labels, predictions), 4),
            "cohens_kappa": round(cohen_kappa_score(labels, predictions), 4),
            "brier_score_loss": round(brier, 4),
        },
        "classification_report": rounded_report,
        "confusion_matrix": confusion_matrix(labels, predictions).tolist(),
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

# Plot the model's actual calibration curve
calibration_label = "WoodWide (top-label confidence)" if is_multiclass else "WoodWide"
plt.plot(pred_prob, true_prob, marker="o", color="blue", label=calibration_label)

# Formatting the visual graph
plt.xlabel("Mean Confidence" if is_multiclass else "Mean Predicted Probability")
plt.ylabel("Fraction Correct (Accuracy)" if is_multiclass else "Fraction of Positives (Actual Frequency)")
plt.title("WoodWide Calibration Curve")
plt.legend(loc="upper left")
plt.grid(True)

# Display the plot
plt.show()

# response = requests.get(
#     url = f"https://api.woodwide.ai/jobs?limit=5",
#     headers = headers
# )
# response.raise_for_status()
# results_uri = response.json()["items"]

# for item in results_uri:
#     print(json.dumps(item, indent=2))
#     print()

# job_id = "job_5YXT36HE"

# response = requests.get(
#     url = f"https://api.woodwide.ai/jobs/{job_id}/results",
#     headers = headers
# )

# response.raise_for_status()
# results_uri = response.json()["inference_results_uri"]

# parquet_response = requests.get(results_uri)
# parquet_response.raise_for_status()

# df = pd.read_parquet(io.BytesIO(parquet_response.content))
# predictions = df["prediction"].tolist()
# pred_probs = [p if pred == 1 else 1 - p for pred, p in zip(predictions, df["prediction_prob"])]

# print("\nROC-AUC Score:\n", roc_auc_score(labels, pred_probs), "\n")
# print("PR-AUC Score:\n", average_precision_score(labels, pred_probs), "\n")
# print("Matthews Correlation Coefficient:\n", matthews_corrcoef(labels, predictions), "\n")
# print("Cohen's Kappa Score:\n", cohen_kappa_score(labels, predictions), "\n")
# print("Classification Report:\n", classification_report(labels, predictions))
# print("Confusion Matrix:\n", confusion_matrix(labels, predictions), "\n")

# print("Total time taken: ", 19.664893 - 15.293170, " seconds", "\n")