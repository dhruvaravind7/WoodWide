import numpy as np
import pandas as pd
import time
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import addcopyfighandler

from memray import Tracker
from sklearn.calibration import calibration_curve

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, matthews_corrcoef, cohen_kappa_score, average_precision_score, brier_score_loss

BASE = "/Users/dhruvaravind/Desktop/Work/WoodWide/Model_Testing/"

# Loads the training data
train_data = pd.read_csv(f"{BASE}Bank_Marketing_Dataset/train.csv")
X_train = train_data.drop(columns=["Subscribed"])
y_train = train_data["Subscribed"].astype(int)

# Loads the testing data
test_data = pd.read_csv(f"{BASE}Bank_Marketing_Dataset/test.csv")
X_test = test_data.drop(columns=["Subscribed"])
y_test = test_data["Subscribed"].astype(int)

# Automatically detects categorical and numerical columns based on dtype.
# Numeric dtypes go to the numerical pipeline; everything else (object,
# category, bool) is treated as categorical.
num_cols = X_train.select_dtypes(include="number").columns.tolist()
cat_cols = X_train.select_dtypes(exclude="number").columns.tolist()

# Preprocesses the data. Numerical columns are mean-imputed then scaled;
# categorical columns are most-frequent-imputed then one-hot encoded.
num_pipeline = Pipeline([
    ("impute", SimpleImputer(strategy="mean")),
    ("scale", RobustScaler())
])

cat_pipeline = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("encode", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_cols),
    ("cat", cat_pipeline, cat_cols)
])

# The pipeline that the model uses. It first preprocesses the data and then uses the model provided.
clf = Pipeline([
    ("preprocess", preprocessor),
    ("model", LogisticRegression(
        max_iter = 1000,
        class_weight="balanced")
    )
])

print("Training the model now...\n")
training_start= time.time()
with Tracker("Bank_Marketing_Dataset/memory_files/marketing_lr_run.bin"):
    # Trains the model using the training data
    clf.fit(X_train, y_train)

    testing_start = time.time()
    # Running the inference
    test_probs = clf.predict_proba(X_test)[:, 1]
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
plt.plot(pred_prob, true_prob, marker="o", color="blue", label="Logistic Regression")

# Formatting the visual graph
plt.xlabel("Mean Predicted Probability")
plt.ylabel("Fraction of Positives (Actual Frequency)")
plt.title("Logistic Regression Calibration Curve")
plt.legend(loc="upper left")
plt.grid(True)

# Display the plot
plt.show()