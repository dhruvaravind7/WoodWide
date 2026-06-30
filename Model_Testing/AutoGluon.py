import pandas as pd

from autogluon.tabular import TabularPredictor
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

data = pd.read_csv("/Users/dhruvaravind/Desktop/Work/WoodWide/Model_Testing/bank.csv")
X_train, X_test, y_train, y_test = train_test_split(data.drop(columns=["Exited"]), data["Exited"], test_size=0.2, random_state=42) # type: ignore[misc]

train_data = pd.concat([X_train, y_train], axis=1)
test_data = pd.concat([X_test, y_test], axis=1) # type: ignore[misc]

predictor = TabularPredictor(label="Exited", eval_metric="roc_auc").fit(train_data)
y_pred_proba = predictor.predict_proba(test_data)
predictor.evaluate(test_data, auxiliary_metrics=True)

auc = roc_auc_score(y_test, y_pred_proba[1])
print(f"ROC-AUC: {auc:.4f}")