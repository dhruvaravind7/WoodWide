import os
import time
import numpy as np
import pandas as pd
import skrub

from neuralk import SeldonClassifier
from dotenv import load_dotenv
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, matthews_corrcoef, cohen_kappa_score, average_precision_score
from sklearn.pipeline import make_pipeline

load_dotenv()

# Loads the training and testing data
train = pd.read_csv("/Users/dhruvaravind/Desktop/Work/WoodWide/Model_Testing/bank_train.csv")
test = pd.read_csv("/Users/dhruvaravind/Desktop/Work/WoodWide/Model_Testing/bank_test.csv")
X_train = train.drop(columns=["Exited"])
y_train = train["Exited"]
X_test = test.drop(columns=["Exited"])
y_test = test["Exited"]

# Creates the model pipeline
model = make_pipeline(
    skrub.TableVectorizer(),
    skrub.SquashingScaler(),
    SimpleImputer(),
    SeldonClassifier(api_key=os.getenv("NeuralkAI_API_KEY"))
)
# Training the model
training_start = time.time()
model.fit(X_train, y_train)

# Testing the model
testing_start = time.time()
churn_probs = model.predict_proba(X_test)
test_probs = churn_probs[:, 1]
test_preds = (test_probs >= 0.5).astype(int)
np.save("predictions.npy", churn_probs)

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