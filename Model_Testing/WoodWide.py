import requests
import os
import time
import pandas as pd

from dotenv import load_dotenv
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

load_dotenv()

labels = pd.read_csv("/Users/dhruvaravind/Desktop/Work/WoodWide/Model_Testing/bank_labels.csv").squeeze()
api_key = os.getenv("WOODWIDE_API_KEY")
base_url = "https://api.woodwide.ai"
headers = {"Authorization": f"Bearer {api_key}"}
'''
# Upload training data
with open("/Users/dhruvaravind/Desktop/Work/WoodWide/Model_Testing/customer_churn_dataset-training-master.csv", "rb") as f:
    resp = requests.post(
        f"{base_url}/datasets",
        headers=headers,
        files={"file": ("/Users/dhruvaravind/Desktop/Work/WoodWide/Model_Testing/customer_churn_dataset-training-master.csv", f, "text/csv")},
        data={"dataset_name": "customer_churn"},
    )
dataset_id = resp.json()["dataset"]["id"]

#dataset_id = "ds_9MUU4UW6"

start = time.time()

# Train a prediction model
resp = requests.post(
    f"{base_url}/models/train",
    headers=headers,
    json={
        "model_name": "customer_churn_model",
        "model_type": "prediction",
        "dataset_id": dataset_id,
        "label_column": "Churn",
    },
)
print("Status code:", resp.status_code)
print("Response JSON:", resp.json())
resp.raise_for_status()
model_id = resp.json()["id"]

# Wait for training
while True:
    model = requests.get(
        f"{base_url}/models/{model_id}", headers=headers
    ).json()
    if model["status"] == "ready":
        break
    time.sleep(5)
'''
# Run inference on new data
with open("/Users/dhruvaravind/Desktop/Work/WoodWide/Model_Testing/bank_test.csv", "rb") as f:
    resp = requests.post(
        f"{base_url}/models/{"mdl_ANAPSM7Y"}/infer",
        headers=headers,
        files={"file": ("/Users/dhruvaravind/Desktop/Work/WoodWide/Model_Testing/bank_test.csv", f, "text/csv")},
        data={"output_type": "json"},
    )

resp.raise_for_status()
data = resp.json()['data']
predictions = data["prediction"]
# prediction_prob is confidence of the predicted class; convert to P("yes") for ROC-AUC
probs = [p if pred == "yes" else 1 - p for pred, p in zip(predictions, data["prediction_prob"])]

print("Roc-AUC Score:", roc_auc_score(labels, probs))
print("Classification Report:\n", classification_report(labels, predictions))
print("Confusion Matrix:\n", confusion_matrix(labels, predictions))
#print("Total time taken: ", time.time() - start, " seconds")