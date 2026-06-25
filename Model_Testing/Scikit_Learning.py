import time
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

print("Loading datasets\n\n")
# Loading the training datasets
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

# print(x_train.head())
# print(y_train.head())

# Seperates the numerical and categorical columns
num_cols = [c for c in X_train.columns if c.startswith("num_")]
cat_cols = [c for c in X_train.columns if c.startswith("cat_")]

# Preprocesses the data since Scikit learning requires heavy feature engineering
# Numerical values are scaled used the RobustScalar
# Categorical values are changed to one hot encoding since scikit learning only works with numerical values.
preprocessor = ColumnTransformer([
    ("num", RobustScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])

# The pipeline that the model uses. It first preprocesses the data and then uses the model provided.
clf = Pipeline([
    ("preprocess", preprocessor),
    ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))
])

print("Training the model now...\n")
start_time = time.time()

# Trains the model using the training data
clf.fit(X_train, Y_train)

print("Testing on the validation set...\n")

# Cross validitaton set to check model capabilites before testing
# Necessary for hyperparameter tuning and to prevent training the model for the test set.
# AUC curves are a better way to evaluate the performance of a binary classifier.
probs = clf.predict_proba(X_val)[:, 1]
print(f"Validation AUC: {roc_auc_score(Y_val, probs)}")

# Accuracy, Recall, and F1-score are also important metrics to consider.
val_preds = clf.predict(X_val)
print(classification_report(Y_val, val_preds))
val_matrix = confusion_matrix(Y_val, val_preds)
print(f"Validation Confusion Matrix:\n{val_matrix}\n")

# Testing on the test set
test_preds = clf.predict(X_test)

# AUC curves are a better way to evaluate the performance of a binary classifier.
test_probs = clf.predict_proba(X_test)[:, 1]
print(f"Test AUC: {roc_auc_score(Y_test, test_probs)}")

# Accuracy, Recall, and F1-score are also important metrics to consider.
print(classification_report(Y_test, test_preds))
test_matrix = confusion_matrix(Y_test, test_preds)
print(f"Test Confusion Matrix:\n{test_matrix}\n")

# Measuring the time taken to train and test the model
end_time = time.time()
print(f"Total time taken: {end_time - start_time:.2f} seconds\n")