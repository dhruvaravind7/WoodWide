import time
import pandas as pd

from autogluon.tabular import TabularPredictor
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, matthews_corrcoef, cohen_kappa_score, average_precision_score

train_data = pd.read_csv("/Users/dhruvaravind/Desktop/Work/WoodWide/Model_Testing/bank_train.csv")
train_data = train_data.sample(frac=0.01, random_state=42).reset_index(drop=True)
test_data = pd.read_csv("/Users/dhruvaravind/Desktop/Work/WoodWide/Model_Testing/bank_test.csv")
y_test = test_data["Exited"]

training_start = time.time()
mitra_predictor = TabularPredictor(label='Exited')
mitra_predictor.fit(
    train_data=train_data,
    hyperparameters={
        'MITRA': {'fine_tune': False}
    },
    ag_args_fit={"ag.max_memory_usage_ratio": 1.3}
)

testing_start = time.time()
class_probabilities = mitra_predictor.predict_proba(test_data)
class_probabilities.to_csv("/Users/dhruvaravind/Desktop/Work/WoodWide/Model_Testing/mitra_test_probabilities.csv", index=False)
test_probs = class_probabilities.iloc[:, 1]
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