import tabpfn_client
import os
import numpy as np
import pandas as pd
import time

from dotenv import load_dotenv
from tabpfn_client import TabPFNClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split


load_dotenv()
tabpfn_client.set_access_token(os.getenv("TABPFN_TOKEN"))
# Loads the data and splits it up into the the target column and the other columns
# full_data = pd.read_csv("/Users/dhruvaravind/Desktop/Work/WoodWide/Model_Testing/customer_churn_dataset-master.csv")
# X_full = full_data.drop(columns=["CustomerID", "Churn"])
# y_full = full_data["Churn"]

# # Splits the data into training, validation, and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=0.2, random_state=42, stratify=y_full)
# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

# # Stores the column names of the categorical and numerical columns
# cat_cols = ["Gender", "Subscription Type", "Contract Length"]
# num_cols = [col for col in X_train.columns if col not in cat_cols]

def get_data(split: bool = False):
    if split:
        training_data = pd.read_csv("/Users/dhruvaravind/Desktop/Work/WoodWide/Model_Testing/customer_churn_dataset-training-master.csv")
        testing_data = pd.read_csv("/Users/dhruvaravind/Desktop/Work/WoodWide/Model_Testing/customer_churn_dataset-testing-master.csv")

        X_train_full = training_data.drop(columns=["CustomerID", "Churn"])
        y_train_full = training_data["Churn"]
        X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full)
        X_test = testing_data.drop(columns=["CustomerID", "Churn"])
        y_test = testing_data["Churn"]
    else:
        full_data = pd.read_csv("/Users/dhruvaravind/Desktop/Work/WoodWide/Model_Testing/bank.csv")
        X_full = full_data.drop(columns=["Exited"])
        y_full = full_data["Exited"]

        # Splits the data into training, validation, and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=0.2, random_state=42, stratify=y_full)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)
    return X_train, X_val, X_test, y_train, y_val, y_test

X_train, X_val, X_test, y_train, y_val, y_test = get_data()

start_time = time.time()

model = TabPFNClassifier()
model.fit(X_train, y_train)

probs = model.predict_proba(X_val)[:, 1]
preds = (probs >= 0.5).astype(int)
print(f"Validation ROC-AUC: {roc_auc_score(y_val, probs)}\n")
print(f"Validation Classification Report: \n{classification_report(y_val, preds)}")
print(f"Confusion Matrix: \n{confusion_matrix(y_val, preds)}")

# Evaluate the model on the test set

test_probs = model.predict_proba(X_test)[:, 1]
test_preds = (test_probs >= 0.5).astype(int)
print(f"Test ROC-AUC: {roc_auc_score(y_test, test_probs)}\n")
print(f"Test Classification Report: \n{classification_report(y_test, test_preds)}")
print(f"Test Confusion Matrix: \n{confusion_matrix(y_test, test_preds)}\n")

end_time = time.time()
print(f"Total time taken: {end_time - start_time:.2f} seconds")