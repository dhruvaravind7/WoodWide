import os
import time
import numpy as np
import pandas as pd
import kumoai.experimental.rfm as rfm

from dotenv import load_dotenv
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, matthews_corrcoef, cohen_kappa_score, average_precision_score

load_dotenv()
rfm.init(api_key=os.getenv("KUMO_API_KEY"))
TRAIN_DATASET_LENGTH = 132027

train = pd.read_csv("/Users/dhruvaravind/Desktop/Work/WoodWide/Model_Testing/bank_train.csv")
test = pd.read_csv("/Users/dhruvaravind/Desktop/Work/WoodWide/Model_Testing/bank_test.csv")
data = pd.concat([train, test], ignore_index=True)

data['Row_ID'] = range(1, len(data) + 1)
y_test = test['Exited']

graph = rfm.Graph.from_data({
    "churn_information": data
})
graph["churn_information"].primary_key = "Row_ID"
graph.validate()
model = rfm.KumoRFM(graph)
start_time = time.time()

pql_query = "PREDICT churn_information.Exited=1 FOR EACH churn_information.Row_ID"

with model.batch_mode(batch_size = 1000):
    prediction = model.predict(
        pql_query, 
        indices=data['Row_ID'].tolist()[TRAIN_DATASET_LENGTH:]
    )

np.save("predictions.npy", prediction[['TARGET_PRED', 'True_PROB']])
print(prediction[['TARGET_PRED', 'True_PROB']])
test_probs = prediction['True_PROB'].values
test_preds = (test_probs >= 0.5).astype(int)

# Prints the important metrics
print("\nROC-AUC Score:\n", roc_auc_score(y_test, test_probs), "\n")
print("PR-AUC Score:\n", average_precision_score(y_test, test_probs), "\n")
print("Matthews Correlation Coefficient:\n", matthews_corrcoef(y_test, test_preds), "\n")
print("Cohen's Kappa Score:\n", cohen_kappa_score(y_test, test_preds), "\n")
print("Classification Report:\n", classification_report(y_test, test_preds))
print("Confusion Matrix:\n", confusion_matrix(y_test, test_preds), "\n") 

print("Total time taken: ", time.time() - start_time, " seconds", "\n")