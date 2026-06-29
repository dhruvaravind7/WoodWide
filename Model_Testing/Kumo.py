import os
import time
import pandas as pd
import kumoai.experimental.rfm as rfm

from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

load_dotenv()
rfm.init(api_key=os.getenv("KUMO_API_KEY"))

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
        # full_data = pd.read_csv("/Users/dhruvaravind/Desktop/Work/WoodWide/Model_Testing/customer_churn_dataset-master.csv")
        full_data = pd.read_csv("/Users/dhruvaravind/Desktop/Work/WoodWide/Model_Testing/bank_serialized.csv")
        return full_data

data = get_data()

graph = rfm.Graph.from_data({
    "churn_information": data
})
graph["churn_information"].primary_key = "Row_ID"
graph.validate()
model = rfm.KumoRFM(graph)
start = time.time()

metrics = model.evaluate(
    "PREDICT churn_information.Exited=1 FOR EACH churn_information.Row_ID",
    metrics=['acc', 'precision', "recall", "f1", "auroc"]
)

metrics_dict = metrics.set_index("metric")["value"].to_dict()

print("AUROC:", metrics_dict["auroc"])
print("Accuracy:", metrics_dict["acc"])
print("F1:", metrics_dict["f1"])
print("Precision:", metrics_dict["precision"])
print("Recall:", metrics_dict["recall"])

print("Total Time: ", time.time() - start)