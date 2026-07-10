import pandas as pd
import time

from autogluon.tabular import TabularPredictor
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, matthews_corrcoef, cohen_kappa_score, average_precision_score

train_data = pd.read_csv("/Users/dhruvaravind/Desktop/Work/WoodWide/Model_Testing/Cardiovascular-Disease-dataset/disease_train.csv")
test_data = pd.read_csv("/Users/dhruvaravind/Desktop/Work/WoodWide/Model_Testing/Cardiovascular-Disease-dataset/disease_test.csv")
y_test = test_data['Disease']

training_start = time.time()
predictor = TabularPredictor(label="Disease", eval_metric="mcc").fit(
    train_data = train_data,
    ag_args_fit = {"num_gpus": 1},
)

testing_start = time.time()
test_probs = predictor.predict_proba(test_data)[1]
test_preds = (test_probs >= 0.5).astype(int)

# Prints the important metrics
print("\nROC-AUC Score:\n", roc_auc_score(y_test, test_probs), "\n")
print("PR-AUC Score:\n", average_precision_score(y_test, test_probs), "\n")
print("Matthews Correlation Coefficient:\n", matthews_corrcoef(y_test, test_preds), "\n")
print("Cohen's Kappa Score:\n", cohen_kappa_score(y_test, test_preds), "\n")
print("Classification Report:\n", classification_report(y_test, test_preds, digits=4))
print("Confusion Matrix:\n", confusion_matrix(y_test, test_preds), "\n") 

print("Training time taken: ", testing_start - training_start, " seconds", "\n")
print("Testing time taken: ", time.time() - testing_start, " seconds", "\n")
print("Total time taken: ", time.time() - training_start, " seconds", "\n")