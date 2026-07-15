import os

# Install first, in its own cell:
#   !pip install autogluon.tabular[mitra] memray
#
# !pip install autogluon.tabular[mitra] memray
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

from memray import Tracker
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, matthews_corrcoef, cohen_kappa_score, average_precision_score, brier_score_loss

# Mitra's MitraClassifier normally calls Tab2D.from_pretrained(), which hands the
# value straight to hf_hub_download() -- it takes a HF repo id, NOT a local path,
# so there is no model_path-style escape hatch like TabPFN has. Flipping USE_HF
# off instead routes it through the local branch, which builds Tab2D from the
# defaults (dim=512, n_layers=12, n_heads=4, dim_output=10 -- these match the
# published config.json exactly) and torch.load()s the weights from state_dict.
# That branch wants a torch checkpoint, not safetensors, hence the .pt conversion.
import autogluon.tabular.models.mitra.sklearn_interface as si
si.USE_HF = False
from autogluon.tabular.models.mitra.sklearn_interface import MitraClassifier

KAGGLE_INPUT = "/kaggle/input/datasets/dhruvaravind"
WEIGHTS_DIR = f"{KAGGLE_INPUT}/mitra-classification-weights"
OUTPUT_DIR = "/kaggle/working"

# --- per-experiment: point these at the data dataset and name its label column ---
DATASET_DIR = "/kaggle/input/datasets/dhruvaravind/cardio-dataset-v2"
TARGET = "Disease"
RUN_NAME = "disease"


def load_mitra(device="cuda"):
    return MitraClassifier(
        state_dict=f"{WEIGHTS_DIR}/mitra_classifier.pt",
        device=device,
        fine_tune=False,
    )


clf = load_mitra(device="cuda")

train = pd.read_csv(f"{DATASET_DIR}/{RUN_NAME}_train.csv")
test = pd.read_csv(f"{DATASET_DIR}/{RUN_NAME}_test.csv")

# Mitra, like TabPFN, is an in-context learner with a bounded context window.
TRAIN_SAMPLE_SIZE = 10000
if TRAIN_SAMPLE_SIZE is not None and len(train) > TRAIN_SAMPLE_SIZE:
    train = train.sample(n=TRAIN_SAMPLE_SIZE, random_state=42)

X_train = train.drop(columns=[TARGET])
y_train = train[TARGET].astype(int)
X_test = test.drop(columns=[TARGET])
y_test = test[TARGET].astype(int)

# Mitra's preprocessor calls np.nanmean() straight on the feature array, so any
# string column blows up with "unsupported operand type(s) for /: 'str' and 'int'".
# TabPFN encodes categoricals internally and AutoGluon's TabularPredictor would
# normally do it upstream, but MitraClassifier is the bare sklearn interface and
# does neither. Ordinal (not one-hot) keeps the column count inside Mitra's limit.
cat_cols = X_train.select_dtypes(exclude="number").columns.tolist()
if cat_cols:
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X_train[cat_cols] = encoder.fit_transform(X_train[cat_cols])
    X_test[cat_cols] = encoder.transform(X_test[cat_cols])

X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)

print("Training data...")
training_start = time.time()
with Tracker(f"{OUTPUT_DIR}/{RUN_NAME}_mit_run.bin"):
    clf.fit(X_train, y_train)

    testing_start = time.time()
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

plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly Calibrated")
plt.plot(pred_prob, true_prob, marker="o", color="blue", label="Mitra")

plt.xlabel("Mean Predicted Probability")
plt.ylabel("Fraction of Positives (Actual Frequency)")
plt.title("Mitra Calibration Curve")
plt.legend(loc="upper left")
plt.grid(True)

plt.show()
