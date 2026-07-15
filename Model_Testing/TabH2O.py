import os
import numpy as np
import pandas as pd
import time
import requests
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import addcopyfighandler

from memray import Tracker
from sklearn.calibration import calibration_curve
from dotenv import load_dotenv
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, matthews_corrcoef, cohen_kappa_score, average_precision_score, brier_score_loss

load_dotenv()

BASE = "/Users/dhruvaravind/Desktop/Work/WoodWide/Model_Testing/"

# Loading the training and testing data
train = pd.read_csv(f"{BASE}Bank_Marketing_Dataset/train.csv")
test = pd.read_csv(f"{BASE}Bank_Marketing_Dataset/test.csv")
test_features = pd.read_csv(f"{BASE}Bank_Marketing_Dataset/test_features.csv")
y_test = pd.read_csv(f"{BASE}Bank_Marketing_Dataset/test_labels.csv")

train_test = pd.concat([train, test_features], ignore_index=True)
train_test.to_csv("data.csv", index=False)

api_key = os.getenv('TABH20_API_KEY')
if not api_key:
    raise ValueError("TABH20_API_KEY not found in environment")

start_time = time.time()
with Tracker("Bank_Marketing_Dataset/memory_files/marketing_h2o_run.bin"):
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

true_prob, pred_prob = calibration_curve(y_test, test_probs, n_bins=10, strategy="quantile")
brier = brier_score_loss(y_test, test_probs)
print(f"Brier Score Loss: {brier:.4f}")
plt.figure(figsize=(8, 6))

# Plot the ideal baseline (perfect calibration)
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly Calibrated")

# Plot the model's actual calibration curve
plt.plot(pred_prob, true_prob, marker="o", color="blue", label="TabH2O")

# Formatting the visual graph
plt.xlabel("Mean Predicted Probability")
plt.ylabel("Fraction of Positives (Actual Frequency)")
plt.title("TabH2O Calibration Curve")
plt.legend(loc="upper left")
plt.grid(True)

# Display the plot
plt.show()