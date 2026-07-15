import pandas as pd
import matplotlib
import time
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import addcopyfighandler

from memray import Tracker
from sklearn.preprocessing import OrdinalEncoder
from tabdpt import TabDPTClassifier
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, matthews_corrcoef, cohen_kappa_score, average_precision_score, brier_score_loss


BASE = "/Users/dhruvaravind/Desktop/Work/WoodWide/Model_Testing/"

# Loading the training and testing data
train_data = pd.read_csv(f"{BASE}Bank_Marketing_Dataset/train.csv")
test_data = pd.read_csv(f"{BASE}Bank_Marketing_Dataset/test.csv")


X_train = train_data.drop(columns=["Subscribed"])
y_train = train_data["Subscribed"]

# Loads the testing data

X_test = test_data.drop(columns=["Subscribed"])
y_test = test_data["Subscribed"]

num_cols = X_train.select_dtypes(include="number").columns.tolist()
cat_cols = X_train.select_dtypes(exclude="number").columns.tolist()

# 'handle_unknown' handles unseen categories in your test data safely
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

# Fit and transform the training features
X_train_encoded = X_train.copy()
X_train_encoded[cat_cols] = encoder.fit_transform(X_train[cat_cols])

X_test_encoded = X_test.copy()
X_test_encoded[cat_cols] = encoder.transform(X_test[cat_cols])

model = TabDPTClassifier(
    device="cpu"           # Use "mps" for Apple Silicon GPU acceleration, or "cpu"
)

with Tracker("Bank_Marketing_Dataset/memory_files/marketing_dpt_run.bin"):
    training_start = time.time()
    model.fit(X_train_encoded.to_numpy(dtype="float32"), y_train.to_numpy())

    testing_start = time.time()
    test_probs = model.predict_proba(X_test_encoded.to_numpy(dtype="float32"), context_size=2048)
    test_probs = test_probs[:, 1]
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
plt.plot(pred_prob, true_prob, marker="o", color="blue", label="TabDPT")

# Formatting the visual graph
plt.xlabel("Mean Predicted Probability")
plt.ylabel("Fraction of Positives (Actual Frequency)")
plt.title("TabDPT Calibration Curve")
plt.legend(loc="upper left")
plt.grid(True)

# Display the plot
plt.show()