import os
import time
import numpy as np
import pandas as pd
import kumoai.experimental.rfm as rfm
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import addcopyfighandler

from sklearn.calibration import calibration_curve

from dotenv import load_dotenv
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, matthews_corrcoef, cohen_kappa_score, average_precision_score, brier_score_loss

load_dotenv()
rfm.init(api_key=os.getenv("KUMO_API_KEY"))

train = pd.read_csv("/Users/dhruvaravind/Desktop/Work/WoodWide/Model_Testing/Bank_Churn_Dataset/bank_train.csv")
test = pd.read_csv("/Users/dhruvaravind/Desktop/Work/WoodWide/Model_Testing/Bank_Churn_Dataset/bank_test.csv")
data = pd.concat([train, test], ignore_index=True)
TRAIN_DATASET_LENGTH = len(train)

data['Row_ID'] = range(1, len(data) + 1)
y_test = test['Exited']

graph = rfm.Graph.from_data({
    "titanic_information": data
})
graph["titanic_information"].primary_key = "Row_ID"
graph.validate()
model = rfm.KumoRFM(graph)
start_time = time.time()

pql_query = "PREDICT titanic_information.Exited=1 FOR EACH titanic_information.Row_ID"

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
print("Classification Report:\n", classification_report(y_test, test_preds, digits=4))
print("Confusion Matrix:\n", confusion_matrix(y_test, test_preds), "\n") 

print("Total time taken: ", time.time() - start_time, " seconds", "\n")

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