import numpy as np
import pandas as pd
import time

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, matthews_corrcoef, cohen_kappa_score, average_precision_score

# Loads the training data
train_data = pd.read_csv("/Users/dhruvaravind/Desktop/Work/WoodWide/Model_Testing/bank_train.csv")
X_train = train_data.drop(columns=["Exited"])
y_train = train_data["Exited"]

# Loads the testing data
test_data = pd.read_csv("/Users/dhruvaravind/Desktop/Work/WoodWide/Model_Testing/bank_test.csv")
X_test = test_data.drop(columns=["Exited"])
y_test = test_data["Exited"]

# Stores the column names of the categorical and numerical columns
cat_cols = ["Gender", "Geography"]
num_cols = [col for col in X_train.columns if col not in cat_cols]

# Preprocesses the data
preprocessor = ColumnTransformer([
    ("num", RobustScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])

# The pipeline that the model uses. It first preprocesses the data and then uses the model provided.
clf = Pipeline([
    ("preprocess", preprocessor),
    ("model", LogisticRegression(
        max_iter=1000,
        class_weight="balanced")
    )
])

print("Training the model now...\n")
training_start= time.time()

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
print("Classification Report:\n", classification_report(y_test, test_preds))
print("Confusion Matrix:\n", confusion_matrix(y_test, test_preds), "\n")

print("Training time taken: ", testing_start - training_start, " seconds", "\n")
print("Testing time taken: ", time.time() - testing_start, " seconds", "\n")
print("Total time taken: ", time.time() - training_start, " seconds", "\n")
