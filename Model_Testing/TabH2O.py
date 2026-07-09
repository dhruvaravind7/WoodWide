import os
import numpy as np
import pandas as pd
import time
import requests

from dotenv import load_dotenv
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, matthews_corrcoef, cohen_kappa_score, average_precision_score

load_dotenv()

train = pd.read_csv("/Users/dhruvaravind/Desktop/Work/WoodWide/Model_Testing/Bank_Marketing_Dataset/marketing_train.csv")
test_features = pd.read_csv("/Users/dhruvaravind/Desktop/Work/WoodWide/Model_Testing/Bank_Marketing_Dataset/marketing_test_features.csv")
y_test = pd.read_csv("/Users/dhruvaravind/Desktop/Work/WoodWide/Model_Testing/Bank_Marketing_Dataset/marketing_test_labels.csv")
train_test = pd.concat([train, test_features], ignore_index=True)
train_test.to_csv("data.csv", index=False)

api_key = os.getenv('TABH20_API_KEY')
if not api_key:
    raise ValueError("TABH20_API_KEY not found in environment")

start_time = time.time()

with open("data.csv", "rb") as f:
    response = requests.post(
        "https://tabh2o.h2oai.com/api/v1/predict",
        headers={"Authorization": f"Bearer {api_key}"},
        files={"file": ("data.csv", f)},
        data={
            "target_column": "Subscribed",
            "task": "classification",
        },
    )
if not response.ok:
    print(f"Train request failed {response.status_code}: {response.text}")
response.raise_for_status()
result = response.json()["probabilities"]

test_probs = np.array([p[1] for p in result])
test_preds = (test_probs >= 0.5).astype(int)

# Prints the important metrics
print("\nROC-AUC Score:\n", roc_auc_score(y_test, test_probs), "\n")
print("PR-AUC Score:\n", average_precision_score(y_test, test_probs), "\n")
print("Matthews Correlation Coefficient:\n", matthews_corrcoef(y_test, test_preds), "\n")
print("Cohen's Kappa Score:\n", cohen_kappa_score(y_test, test_preds), "\n")
print("Classification Report:\n", classification_report(y_test, test_preds, digits=4))
print("Confusion Matrix:\n", confusion_matrix(y_test, test_preds), "\n")

print("Total time taken: ", time.time() - start_time, " seconds", "\n")