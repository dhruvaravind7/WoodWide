import os

# v2 is the only ungated checkpoint. v2.5/v2.6/v3 require a one-time license
# acceptance at https://ux.priorlabs.ai before the weights will download.
# Has to be set before tabpfn is imported.
os.environ.setdefault("TABPFN_MODEL_VERSION", "v2")

import numpy as np
import pandas as pd
import time
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import addcopyfighandler

from memray import Tracker
from sklearn.calibration import calibration_curve
from tabpfn import TabPFNClassifier
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, matthews_corrcoef, cohen_kappa_score, average_precision_score, brier_score_loss

BASE = "/Users/dhruvaravind/Desktop/Work/WoodWide/Model_Testing/"

# Loading the training and testing data
train = pd.read_csv(f"{BASE}Bank_Marketing_Dataset/train.csv")
test = pd.read_csv(f"{BASE}Bank_Marketing_Dataset/test.csv")

# TabPFN was pretrained on contexts of up to 10,000 rows. Past that it still runs
# (ignore_pretraining_limits), but it is outside the regime it was trained for.
TRAIN_SAMPLE_SIZE = 10000
if TRAIN_SAMPLE_SIZE is not None and len(train) > TRAIN_SAMPLE_SIZE:
    train = train.sample(n=TRAIN_SAMPLE_SIZE, random_state=42)

X_train = train.drop(columns=["Subscribed"])
y_train = train["Subscribed"]
X_test = test.drop(columns=["Subscribed"])
y_test = test["Subscribed"]

# Trains the model
with Tracker("Bank_Marketing_Dataset/memory_files/marketing_pfn_run.bin"):
    training_start = time.time()
    # "auto" resolves to mps locally and cuda on Kaggle. CPU is refused above 1,000
    # samples. TabPFN encodes the string columns itself, so no OrdinalEncoder needed.
    model = TabPFNClassifier(device="auto", ignore_pretraining_limits=True)
    model.fit(X_train, y_train)

    # Evaluate the model on the test set
    testing_start = time.time()
    test_probs = model.predict_proba(X_test)[:, 1]
    test_preds = (test_probs >= 0.5).astype(int)

# Prints the important metrics
print("\nROC-AUC Score:\n", roc_auc_score(y_test, test_probs), "\n")
print("PR-AUC Score:\n", average_precision_score(y_test, test_probs), "\n")
print("Matthews Correlation Coefficient:\n", matthews_corrcoef(y_test, test_preds), "\n")
print("Cohen's Kappa Score:\n", cohen_kappa_score(y_test, test_preds), "\n")
print("Classification Report:\n", classification_report(y_test, test_preds, digits=4))
print("Confusion Matrix:\n", confusion_matrix(y_test, test_preds), "\n")

print("Training time taken: ", testing_start - training_start, " seconds", "\n")
print("Testing time taken: ", time.time() - testing_start, " seconds", "\n")
print("Total time taken: ", time.time() - training_start, " seconds", "\n")

true_prob, pred_prob = calibration_curve(y_test, test_probs, n_bins=10, strategy="quantile")
brier = brier_score_loss(y_test, test_probs)
print(f"Brier Score Loss: {brier:.4f}")
plt.figure(figsize=(8, 6))

# Plot the ideal baseline (perfect calibration)
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly Calibrated")

# Plot the model's actual calibration curve
plt.plot(pred_prob, true_prob, marker="o", color="blue", label="TabPFN")

# Formatting the visual graph
plt.xlabel("Mean Predicted Probability")
plt.ylabel("Fraction of Positives (Actual Frequency)")
plt.title("TabPFN Calibration Curve")
plt.legend(loc="upper left")
plt.grid(True)

# Display the plot
plt.show()