import os
import time
import numpy as np
import pandas as pd
import kumoai.experimental.rfm as rfm
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import addcopyfighandler

from dataclasses import replace
from kumoapi.model_plan import RunMode
from sklearn.calibration import calibration_curve
from memray import Tracker
from dotenv import load_dotenv
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, matthews_corrcoef, cohen_kappa_score, average_precision_score, brier_score_loss

load_dotenv()
rfm.init(api_key=os.getenv("KUMO_API_KEY"))

BASE = "/Users/dhruvaravind/Desktop/Work/WoodWide/Model_Testing/"

# Loading the training and testing data
train = pd.read_csv(f"{BASE}Bank_Marketing_Dataset/train.csv")
test = pd.read_csv(f"{BASE}Bank_Marketing_Dataset/test.csv")

data = pd.concat([train, test], ignore_index=True)
TRAIN_DATASET_LENGTH = len(train)

data['Row_ID'] = range(1, len(data) + 1)
y_test = test['Subscribed']

setup_start = time.perf_counter()

graph = rfm.Graph.from_data({
    "titanic_information": data
})
graph["titanic_information"].primary_key = "Row_ID"
graph.validate()
model = rfm.KumoRFM(graph)

setup_time = time.perf_counter() - setup_start

pql_query = "PREDICT titanic_information.Subscribed=1 FOR EACH titanic_information.Row_ID"
pred_indices = data['Row_ID'].tolist()[TRAIN_DATASET_LENGTH:]

# KumoRFM is pre-trained and never fits on our data, so it has no training step.
# Its analog is in-context learning: sampling labelled context examples out of
# the graph. predict() does that and then runs the forward pass, so unroll it
# into its two phases to time them separately.
query_def = model._parse_query(pql_query)
query_def = replace(query_def, for_each="FOR EACH", rfm_entity_ids=None)

icl_start = time.perf_counter()

task_table = model._get_task_table(
    query=query_def,
    indices=pred_indices,
    run_mode=RunMode.FAST,
)
task_table._query = query_def.to_string()

icl_time = time.perf_counter() - icl_start

inference_start = time.perf_counter()
with Tracker("Bank_Marketing_Dataset/memory_files/marketing_kum_run.bin"):
    with model.batch_mode(batch_size = 1000):
        prediction = model.predict_task(
            task_table,
            run_mode=RunMode.FAST,
            exclude_cols_dict=query_def.get_exclude_cols_dict(),
            top_k=query_def.top_k,
        )

    inference_time = time.perf_counter() - inference_start

    np.save("predictions.npy", prediction[['TARGET_PRED', 'True_PROB']])
    print(prediction[['TARGET_PRED', 'True_PROB']])
    test_probs = prediction['True_PROB'].values
    test_preds = (test_probs >= 0.5).astype(int)


# Prints the important metrics
print("\nROC-AUC Score:\n", roc_auc_score(y_test, test_probs), "\n")
print("PR-AUC Score:\n", average_precision_score(y_test, test_probs), "\n")
print("Matthews Correlation Coefficient:\n", matthews_corrcoef(y_test, test_preds), "\n")
print("Cohen's Kappa Score:\n", cohen_kappa_score(y_test, test_preds), "\n")
print("Classification Report:\n", classification_report(y_test, test_preds, digits=4))
print("Confusion Matrix:\n", confusion_matrix(y_test, test_preds), "\n") 

print(f"Graph setup time: {setup_time:.2f} seconds")
print(f"In-context learning time: {icl_time:.2f} seconds "
      f"({task_table.num_context_examples:,} context examples)")
print(f"Inference time: {inference_time:.2f} seconds "
      f"({task_table.num_prediction_examples:,} prediction examples)")
print(f"Total time taken: {setup_time + icl_time + inference_time:.2f} seconds\n")

true_prob, pred_prob = calibration_curve(y_test, test_probs, n_bins=10, strategy="quantile")
brier = brier_score_loss(y_test, test_probs)
print(f"Brier Score Loss: {brier:.4f}")
plt.figure(figsize=(8, 6))

# Plot the ideal baseline (perfect calibration)
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly Calibrated")

# Plot the model's actual calibration curve
plt.plot(pred_prob, true_prob, marker="o", color="blue", label="Kumo")

# Formatting the visual graph
plt.xlabel("Mean Predicted Probability")
plt.ylabel("Fraction of Positives (Actual Frequency)")
plt.title("Kumo Calibration Curve")
plt.legend(loc="upper left")
plt.grid(True)

# Display the plot
plt.show()