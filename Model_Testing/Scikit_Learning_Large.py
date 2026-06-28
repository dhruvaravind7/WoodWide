import numpy as np
import pandas as pd
import time

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

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
        full_data = pd.read_csv("/Users/dhruvaravind/Desktop/Work/WoodWide/Model_Testing/customer_churn_dataset-master.csv")
        X_full = full_data.drop(columns=["CustomerID", "Churn"])
        y_full = full_data["Churn"]

        # Splits the data into training, validation, and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=0.2, random_state=42, stratify=y_full)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)
    return X_train, X_val, X_test, y_train, y_val, y_test

# Chooses whether you want to use the train and test split given by Kaggle or combine the data together and split the data up randomly from that
# True means you use the Kaggle split. False means you combine the data and then split from that.
# The test set and training set from kaggle has different proportions of some of the variables, which is why working with False allow for better generalizability.
X_train, X_val, X_test, y_train, y_val, y_test = get_data()

# Stores the column names of the categorical and numerical columns
cat_cols = ["Gender", "Subscription Type", "Contract Length"]
num_cols = [col for col in X_train.columns if col not in cat_cols]

# Preprocesses the data
preprocessor = ColumnTransformer([
    ("num", RobustScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])

# The pipeline that the model uses. It first preprocesses the data and then uses the model provided.
clf = Pipeline([
    ("preprocess", preprocessor),
    ("model", RandomForestClassifier(n_estimators=100, class_weight="balanced"))
])

print("Training the model now...\n")
start_time = time.time()

# Trains the model using the training data
clf.fit(X_train, y_train)

print("Testing on the validation set...\n")

# Running the cross-validation tests
probs = clf.predict_proba(X_val)[:, 1]
val_preds = clf.predict(X_val)
val_matrix = confusion_matrix(y_val, val_preds)

print(f"Validation AUC: {roc_auc_score(y_val, probs)}")
print(classification_report(y_val, val_preds))
print(f"Validation Confusion Matrix:\n{val_matrix}\n")


# Running the test-set tests
test_probs = clf.predict_proba(X_test)[:, 1]
test_preds = clf.predict(X_test)
test_matrix = confusion_matrix(y_test, test_preds)

# Prints the important metrics
# AUC, Precision, Accuracy, Recall, F1 Score, Confusion Matrix
print(f"Test AUC: {roc_auc_score(y_test, test_probs)}")
print(classification_report(y_test, test_preds))
print(f"Test Confusion Matrix:\n{test_matrix}\n")

# Measuring the time taken to train and test the model
end_time = time.time()
print(f"Total time taken: {end_time - start_time:.2f} seconds\n")
