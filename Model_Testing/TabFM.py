import pandas as pd
import numpy as np
import time

from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from tabfm import TabFMClassifier, tabfm_v1_0_0_pytorch as tabfm_v1

model = tabfm_v1.load(device="mps")
clf = TabFMClassifier(model=model, max_num_rows=500, n_estimators=8, batch_size=4)

full = pd.read_csv("/Users/dhruvaravind/Desktop/Work/WoodWide/Model_Testing/bank_small.csv")
X_full = full.drop(columns=["Exited"])
y_full = full["Exited"]

X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=0.05, random_state=42, stratify=y_full)

print("Training data...")
start = time.time()

clf.fit(X_train, y_train)

print("Testing data...")
chunk_size = 200
chunks = [X_test.iloc[i:i + chunk_size] for i in range(0, len(X_test), chunk_size)]
probs = np.concatenate(
    [clf.predict_proba(chunk) for chunk in tqdm(chunks, desc="Testing", unit="chunk")],
    axis=0,
)
preds = (probs[:, 1] >= 0.5).astype(int)

print("ROC AUC Score:", roc_auc_score(y_test, probs[:, 1]))
print("Classification Report: \n", classification_report(y_test, preds))
print("Confusion Matrix: \n", confusion_matrix(y_test, preds))
print("Total Time: ", time.time() - start, "seconds")