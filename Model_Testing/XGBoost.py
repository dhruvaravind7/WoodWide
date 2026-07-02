import numpy as np
import pandas as pd
import time

from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, matthews_corrcoef, cohen_kappa_score, average_precision_score

# Loading the training and testing data
train = pd.read_csv("/Users/dhruvaravind/Desktop/Work/WoodWide/Model_Testing/bank_train.csv")
test = pd.read_csv("/Users/dhruvaravind/Desktop/Work/WoodWide/Model_Testing/bank_test.csv")
X_train = train.drop(columns=["Exited"])
y_train = train["Exited"]
X_test = test.drop(columns=["Exited"])
y_test = test["Exited"]

# Labels which columns are categorical
cat_cols = ["Gender", "Geography"]
for col in cat_cols:
    X_train[col] = X_train[col].astype("category")
    X_test[col] = X_test[col].astype("category")

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

# The pipeline that the model uses. It first preprocesses the data and then uses the model provided.
clf = Pipeline([
    ("model", XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        enable_categorical=True,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42
    ))
])

# Trains the model using the training data
print("Training the model now...\n")
training_start = time.time()

clf.fit(X_train, y_train)

# Running the test-set tests
testing_start = time.time()
test_probs = clf.predict_proba(X_test)[:, 1]
test_preds = (test_probs >= 0.5).astype(int)

# Prints the important metrics
print("\nROC-AUC Score:\n", roc_auc_score(y_test, test_probs), "\n")
print("PR-AUC Score:\n", average_precision_score(y_test, test_probs), "\n")
print("Matthews Correlation Coefficient:\n", matthews_corrcoef(y_test, test_preds), "\n")
print("Cohen's Kappa Score:\n", cohen_kappa_score(y_test, test_preds), "\n")
print("Classification Report:\n", classification_report(y_test, test_preds))
print("Confusion Matrix:\n", confusion_matrix(y_test, test_preds), "\n")

print("Training time taken: ", testing_start - training_start, " seconds", "\n")
print("Testing time taken: ", time.time() - testing_start, " seconds", "\n")
print("Total time taken: ", time.time() - training_start, " seconds", "\n")