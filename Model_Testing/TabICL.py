import pandas as pd
import numpy as np
import time
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import addcopyfighandler

from memray import Tracker
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, matthews_corrcoef, cohen_kappa_score, average_precision_score, brier_score_loss
from tabicl import TabICLClassifier

BASE = "/Users/dhruvaravind/Desktop/Work/WoodWide/Model_Testing"
DIR = "Bank_Churn_Dataset"
TARGET = "Exited"
RUN_NAME = "bank"

# Loading the training and testing data
train = pd.read_csv(f"{BASE}/{DIR}/train.csv")
test = pd.read_csv(f"{BASE}/{DIR}/test.csv")

TRAIN_SAMPLE_SIZE = 5000
if TRAIN_SAMPLE_SIZE is not None and len(train) > TRAIN_SAMPLE_SIZE:
    train = train.sample(n=TRAIN_SAMPLE_SIZE, random_state=42)

X_train = train.drop(columns=[f"{TARGET}"])
y_train = train[f"{TARGET}"]
X_test = test.drop(columns=[f"{TARGET}"])
y_test = test[f"{TARGET}"]

with Tracker(f"{DIR}/memory_files/{RUN_NAME}_icl_run.bin"):
    print("Training...\n")
    training_start = time.time()
    tabicl = TabICLClassifier()
    tabicl.fit(X_train, y_train)

    print("Testing...\n")
    testing_start = time.time()
    test_probs = tabicl.predict_proba(X_test)
    test_probs = test_probs[:, 1]
    test_preds = (test_probs >= 0.5).astype(int)

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
plt.plot(pred_prob, true_prob, marker="o", color="blue", label="TabICL")

# Formatting the visual graph
plt.xlabel("Mean Predicted Probability")
plt.ylabel("Fraction of Positives (Actual Frequency)")
plt.title("TahICL Calibration Curve")
plt.legend(loc="upper left")
plt.grid(True)

# Display the plot
plt.show()