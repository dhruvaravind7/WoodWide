import os
import numpy as np
import pandas as pd
import time
import requests

from dotenv import load_dotenv
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

load_dotenv()

def get_data(split: bool = False):
    if split:
        training_data = pd.read_csv("/Users/dhruvaravind/Desktop/Work/WoodWide/Model_Testing/customer_churn_dataset-training-master.csv")
        testing_data = pd.read_csv("/Users/dhruvaravind/Desktop/Work/WoodWide/Model_Testing/customer_churn_dataset-testing-master.csv")
        
        val_size = len(training_data) * 0.2
        val_data = training_data.sample(n=int(val_size), random_state=42)
        training_data = training_data.drop(val_data.index)

        y_val_true = val_data["Churn"].copy()
        val_data["Churn"] = ""
        
        y_test_true = testing_data["Churn"].copy()
        testing_data["Churn"] = ""
        total_combined = pd.concat([training_data, val_data,testing_data], ignore_index=True)
        total_combined.to_csv("total_for_prediction.csv", index=False)

        train_combined = pd.concat([training_data, val_data], ignore_index = True)
        train_combined.to_csv("train_for_prediction.csv", index=False)

        return (y_test_true, y_val_true)
    else:
        full_data = pd.read_csv("/Users/dhruvaravind/Desktop/Work/WoodWide/Model_Testing/customer_churn_dataset-master.csv")

        test_size = len(full_data) * 0.2
        test_data = full_data.sample(n=int(test_size), random_state=42)
        train_data = full_data.drop(test_data.index)
        train_data = train_data.sample(n=100_000, random_state=42)

        y_test_true = test_data["Churn"].copy()
        test_data["Churn"] = ""

        val_size = len(train_data) * 0.2
        val_data = train_data.sample(n=int(val_size), random_state=42)
        train_data = train_data.drop(val_data.index)

        y_val_true = val_data["Churn"].copy()
        val_data["Churn"] = ""

        total_combined = pd.concat([train_data, val_data, test_data], ignore_index=True)
        total_combined.to_csv("total_for_prediction.csv", index=False)

        train_combined = pd.concat([train_data, val_data], ignore_index = True)
        train_combined.to_csv("train_for_prediction.csv", index=False)
        return (y_test_true, y_val_true)

y_test_true, y_val_true = get_data()

start_time = time.time()

api_key = os.getenv('TABH20_API_KEY')
if not api_key:
    raise ValueError("TABH20_API_KEY not found in environment")

with open("train_for_prediction.csv", "rb") as f:
    response = requests.post(
        "https://tabh2o.h2oai.com/api/v1/predict",
        headers={"Authorization": f"Bearer {api_key}"},
        files={"file": ("train_for_prediction.csv", f)},
        data={
            "target_column": "Churn",
            "task": "classification",
        },
    )
if not response.ok:
    print(f"Train request failed {response.status_code}: {response.text}")
response.raise_for_status()
result = response.json()["probabilities"]
val_pred_probs = np.array([p[1] for p in result])
val_preds = (val_pred_probs >= 0.5).astype(int)

'''with open("total_for_prediction.csv", "rb") as f:
    response = requests.post(
        "https://tabh2o.h2oai.com/api/v1/predict",
        headers={"Authorization": f"Bearer {api_key}"},
        files={"file": ("total_for_prediction.csv", f)},
        data={
            "target_column": "Churn",
            "task": "classification",
        },
    )
if not response.ok:
    print(f"Total request failed {response.status_code}: {response.text}")
response.raise_for_status()
result = response.json()["probabilities"]
all_pred_probs = np.array([p[1] for p in result])
test_pred_probs = all_pred_probs[len(y_val_true):]
test_preds = (test_pred_probs >= 0.5).astype(int)
'''
print(f"Validation ROC-AUC: {roc_auc_score(y_val_true, val_pred_probs)}\n")
print(f"Validation Classification Report: \n{classification_report(y_val_true, val_preds)}")
print(f"Confusion Matrix: \n{confusion_matrix(y_val_true, val_preds)}")
'''
print(f"Test ROC-AUC: {roc_auc_score(y_test_true, test_pred_probs)}\n")
print(f"Test Classification Report: \n{classification_report(y_test_true, test_preds)}")
print(f"Test Confusion Matrix: \n{confusion_matrix(y_test_true, test_preds)}\n")
'''
end_time = time.time()
print(f"Time taken: {end_time - start_time:.2f} seconds")
