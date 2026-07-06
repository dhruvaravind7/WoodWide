import tabpfn_client
import os
import numpy as np
import pandas as pd
import time

from dotenv import load_dotenv
from tabpfn_client import TabPFNClassifier
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, matthews_corrcoef, cohen_kappa_score, average_precision_score

# Sets the access token for the TabPFN API
load_dotenv()
tabpfn_client.set_access_token(os.getenv("TABPFN_TOKEN"))

# Loads the training and testing data
train = pd.read_csv("/Users/dhruvaravind/Desktop/Work/WoodWide/Model_Testing/bank_train.csv")
test = pd.read_csv("/Users/dhruvaravind/Desktop/Work/WoodWide/Model_Testing/bank_test.csv")
X_train = train.drop(columns=["Exited"])
y_train = train["Exited"]
X_test = test.drop(columns=["Exited"])
y_test = test["Exited"]

# Trains the model
training_start = time.time()
model = TabPFNClassifier()
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
print("Classification Report:\n", classification_report(y_test, test_preds))
print("Confusion Matrix:\n", confusion_matrix(y_test, test_preds), "\n")

print("Training time taken: ", testing_start - training_start, " seconds", "\n")
print("Testing time taken: ", time.time() - testing_start, " seconds", "\n")
print("Total time taken: ", time.time() - training_start, " seconds", "\n")