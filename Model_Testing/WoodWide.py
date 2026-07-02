import requests
import os
import time
import json
import io
import pandas as pd

from dotenv import load_dotenv
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, matthews_corrcoef, cohen_kappa_score, average_precision_score

load_dotenv()

labels = pd.read_csv("/Users/dhruvaravind/Desktop/Work/WoodWide/Model_Testing/bank_test_labels.csv").squeeze()
api_key = os.getenv("WOODWIDE_API_KEY")
base_url = "https://api.woodwide.ai"
headers = {"Authorization": f"Bearer {api_key}"}

# Uploading a dataset and training a model
'''
# Upload training data
print("Uploading training data...")
with open("/Users/dhruvaravind/Desktop/Work/WoodWide/Model_Testing/bank_train.csv", "rb") as f:
    resp = requests.post(
        f"{base_url}/datasets",
        headers=headers,
        files={"file": ("/Users/dhruvaravind/Desktop/Work/WoodWide/Model_Testing/bank_train.csv", f, "text/csv")},
        data={"dataset_name": "bank_churn_train"},
    )

dataset_id = resp.json()["dataset"]["id"]

print("Upload finished. Dataset ID:", dataset_id)
'''
dataset_id = "ds_HZS4PM2T"
##########################################################################################################################################

print("Creating model...")
training_start = time.time()

# Train a prediction model
resp = requests.post(
    f"{base_url}/models/train",
    headers=headers,
    json={
        "model_name": "bank_churn_model",
        "model_type": "prediction",
        "dataset_id": dataset_id,
        "label_column": "Exited",
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
    time.sleep(5)

print("Training finished. Model ID:", model_id)

##########################################################################################################################################

print("Running inference on new data...")

# Run inference on new data and get predictions
with open("/Users/dhruvaravind/Desktop/Work/WoodWide/Model_Testing/bank_test_no_labels.csv", "rb") as f:
    resp = requests.post(
        f"{base_url}/models/{model_id}/infer",
        headers=headers,
        files={"file": ("/Users/dhruvaravind/Desktop/Work/WoodWide/Model_Testing/bank_test_no_labels.csv", f, "text/csv")},
        data={"output_type": "json"},
    )

resp.raise_for_status()
data = resp.json()['data']
predictions = data["prediction"]
pred_probs = [p if pred == 1 else 1 - p for pred, p in zip(predictions, data["prediction_prob"])]

print("\nROC-AUC Score:\n", roc_auc_score(labels, pred_probs), "\n")
print("PR-AUC Score:\n", average_precision_score(labels, pred_probs), "\n")
print("Matthews Correlation Coefficient:\n", matthews_corrcoef(labels, predictions), "\n")
print("Cohen's Kappa Score:\n", cohen_kappa_score(labels, predictions), "\n")
print("Classification Report:\n", classification_report(labels, predictions))
print("Confusion Matrix:\n", confusion_matrix(labels, predictions), "\n")

print("Total time taken: ", 19.664893 - 15.293170, " seconds", "\n")

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