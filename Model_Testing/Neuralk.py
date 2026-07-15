import os
import time
import numpy as np
import pandas as pd
import skrub
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import addcopyfighandler

from memray import Tracker
from sklearn.calibration import calibration_curve
from neuralk import SeldonClassifier
from dotenv import load_dotenv
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, matthews_corrcoef, cohen_kappa_score, average_precision_score, brier_score_loss
from sklearn.pipeline import make_pipeline

load_dotenv()
BASE = "/Users/dhruvaravind/Desktop/Work/WoodWide/Model_Testing/"

# Loading the training and testing data
train = pd.read_csv(f"{BASE}Heloc/train.csv")
test = pd.read_csv(f"{BASE}Heloc/test.csv")

X_train = train.drop(columns=["RiskPerformance"])
y_train = train["RiskPerformance"]
X_test = test.drop(columns=["RiskPerformance"])
y_test = test["RiskPerformance"]

# Creates the model pipeline
model = make_pipeline(
    skrub.TableVectorizer(),
    skrub.SquashingScaler(),
    SimpleImputer(),
    SeldonClassifier(api_key=os.getenv("NeuralkAI_API_KEY"))
)
# Training the model
with Tracker("Heloc/memory_files/heloc_neu_run.bin"):
    training_start = time.time()
    model.fit(X_train, y_train)

    # Testing the model
    testing_start = time.time()
    churn_probs = model.predict_proba(X_test)
    test_probs = churn_probs[:, 1]
    test_preds = (test_probs >= 0.5).astype(int)
    np.save("predictions.npy", churn_probs)

churn_probs = np.load("predictions.npy")
test_probs = churn_probs[:, 1]
test_preds = (test_probs >= 0.5).astype(int)
np.save("predictions.npy", churn_probs)


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
plt.plot(pred_prob, true_prob, marker="o", color="blue", label="Neuralk")

# Formatting the visual graph
plt.xlabel("Mean Predicted Probability")
plt.ylabel("Fraction of Positives (Actual Frequency)")
plt.title("Neuralk Calibration Curve")
plt.legend(loc="upper left")
plt.grid(True)

# Display the plot
plt.show()