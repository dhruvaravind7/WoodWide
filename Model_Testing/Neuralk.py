from neuralk import SeldonClassifier
from dotenv import load_dotenv
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

import os
import time
import numpy as np
import pandas as pd
import skrub

load_dotenv()

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
        X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=0.09, random_state=42, stratify=y_full)
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = get_data()

# Stores the column names of the categorical and numerical columns
#cat_cols = ["Gender", "Subscription Type", "Contract Length"]
#cat_cols = ["gender", "payment_method", "city"]
# cat_cols = ["Gender", "Geography"]
# num_cols = [col for col in X_train.columns if col not in cat_cols]

start = time.time()

model = make_pipeline(
    skrub.TableVectorizer(),
    skrub.SquashingScaler(),
    SimpleImputer(),
    SeldonClassifier(api_key=os.getenv("NeuralkAI_API_KEY3"))
)

model.fit(X_train, y_train)
# val_preds = model.predict(X_val)
# val_probs = model.predict_proba(X_val)
#test_preds = model.predict(X_test)
test_probs = model.predict_proba(X_test)
churn_probs = test_probs[:, 1]
test_preds = (churn_probs >= 0.5).astype(int)
np.save("predictions.npy", test_probs)

total_time = time.time() - start

# print(f"Validation ROC-AUC: {roc_auc_score(y_val, val_probs)}\n")
# print(f"Validation Classification Report: \n{classification_report(y_val, val_preds)}")
# print(f"Confusion Matrix: \n{confusion_matrix(y_val, val_preds)}")

print(f"Test ROC-AUC: {roc_auc_score(y_test, churn_probs)}\n")
print(f"Test Classification Report: \n{classification_report(y_test, test_preds)}")
print(f"Test Confusion Matrix: \n{confusion_matrix(y_test, test_preds)}\n")

print(f"Total Training Time: {total_time:.2f} seconds")