import tabpfn_client
import os
import numpy as np
import pandas as pd
import time

from dotenv import load_dotenv
from tabpfn_client import TabPFNClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score


load_dotenv()
tabpfn_client.set_access_token(os.getenv("TABPFN_TOKEN"))

np_n_train = np.load("/Users/dhruvaravind/Desktop/Work/WoodWide/Model_Testing/bank/N_train.npy", allow_pickle=True)
np_c_train = np.load("/Users/dhruvaravind/Desktop/Work/WoodWide/Model_Testing/bank/C_train.npy", allow_pickle=True)
np_y_train = np.load("/Users/dhruvaravind/Desktop/Work/WoodWide/Model_Testing/bank/y_train.npy", allow_pickle=True)

# Loading the cross validation datasets
np_n_val = np.load("/Users/dhruvaravind/Desktop/Work/WoodWide/Model_Testing/bank/N_val.npy", allow_pickle=True)
np_c_val = np.load("/Users/dhruvaravind/Desktop/Work/WoodWide/Model_Testing/bank/C_val.npy", allow_pickle=True)
np_y_val = np.load("/Users/dhruvaravind/Desktop/Work/WoodWide/Model_Testing/bank/y_val.npy", allow_pickle=True)

# Loading the test datasets
np_n_test = np.load("/Users/dhruvaravind/Desktop/Work/WoodWide/Model_Testing/bank/N_test.npy", allow_pickle=True)
np_c_test = np.load("/Users/dhruvaravind/Desktop/Work/WoodWide/Model_Testing/bank/C_test.npy", allow_pickle=True)
np_y_test = np.load("/Users/dhruvaravind/Desktop/Work/WoodWide/Model_Testing/bank/y_test.npy", allow_pickle=True)

def make_df(n_arr, c_arr):
    n_df = pd.DataFrame(n_arr, columns=[f"num_{i}" for i in range(n_arr.shape[1])])
    c_df = pd.DataFrame(c_arr, columns=[f"col_{i}" for i in range(c_arr.shape[1])])
    return pd.concat([n_df, c_df], axis=1)

# Converts the trained numpy files into Dataframes. Concatenates the numerical and categorical tables into one input dataframe
X_train = make_df(np_n_train, np_c_train)
Y_train = pd.Series(np_y_train.ravel())

# Converts the trained numpy files into Dataframes. Concatenates the numerical and categorical tables into one input dataframe
X_val = make_df(np_n_val, np_c_val)
Y_val = pd.Series(np_y_val.ravel())

# Converts the trained numpy files into Dataframes. Concatenates the numerical and categorical tables into one input dataframe
X_test = make_df(np_n_test, np_c_test)
Y_test = pd.Series(np_y_test.ravel())

start_time = time.time()

model = TabPFNClassifier()
model.fit(X_train, Y_train)

preds = model.predict(X_val)
probs = model.predict_proba(X_val)[:, 1]
print(f"Validation ROC-AUC: {roc_auc_score(Y_val, probs)}\n")
print(f"Validation Classification Report: \n{classification_report(Y_val, preds)}")
print(f"Confusion Matrix: \n{confusion_matrix(Y_val, preds)}")

# Evaluate the model on the test set
test_preds = model.predict(X_test)
test_probs = model.predict_proba(X_test)[:, 1]
print(f"Test ROC-AUC: {roc_auc_score(Y_test, test_probs)}\n")
print(f"Test Classification Report: \n{classification_report(Y_test, test_preds)}")
print(f"Test Confusion Matrix: \n{confusion_matrix(Y_test, test_preds)}\n")

end_time = time.time()
print(f"Total time taken: {end_time - start_time:.2f} seconds")