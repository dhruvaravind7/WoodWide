import pandas as pd
import numpy as np
import time

from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, matthews_corrcoef, cohen_kappa_score, average_precision_score
from tabfm import TabFMClassifier, tabfm_v1_0_0_pytorch as tabfm_v1

model = tabfm_v1.load(device="mps")
clf = TabFMClassifier.ensemble(model=model, max_num_rows=500, n_estimators=4, batch_size=8, enable_nnls=False)

train = pd.read_csv("/Users/dhruvaravind/Desktop/Work/WoodWide/Model_Testing/Bank_Marketing_Dataset/marketing_train.csv")
test = pd.read_csv("/Users/dhruvaravind/Desktop/Work/WoodWide/Model_Testing/Bank_Marketing_Dataset/marketing_test.csv")
X_train = train.drop(columns=["Subscribed"])
y_train = train["Subscribed"]
X_test = test.drop(columns=["Subscribed"])
y_test = test["Subscribed"]


print("Training data...")
training_start = time.time()

clf.fit(X_train, y_train)

print("Testing data...")
chunk_size = 2000
testing_start = time.time()

chunks = [X_test.iloc[i:i + chunk_size] for i in range(0, len(X_test), chunk_size)]
test_probs = np.concatenate(
    [clf.predict_proba(chunk) for chunk in tqdm(chunks, desc="Testing", unit="chunk")],
    axis=0,
)
test_probs = test_probs[:, 1]
test_preds = (test_probs >= 0.5).astype(int)

#
print("\nROC-AUC Score:\n", roc_auc_score(y_test, test_probs), "\n")
print("PR-AUC Score:\n", average_precision_score(y_test, test_probs), "\n")
print("Matthews Correlation Coefficient:\n", matthews_corrcoef(y_test, test_preds), "\n")
print("Cohen's Kappa Score:\n", cohen_kappa_score(y_test, test_preds), "\n")
print("Classification Report:\n", classification_report(y_test, test_preds, digits=4))
print("Confusion Matrix:\n", confusion_matrix(y_test, test_preds), "\n") 

print("Training time taken: ", testing_start - training_start, " seconds", "\n")
print("Testing time taken: ", time.time() - testing_start, " seconds", "\n")
print("Total time taken: ", time.time() - training_start, " seconds", "\n")