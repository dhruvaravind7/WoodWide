import pandas as pd
import numpy as np
import openml

from tabarena_dataset import get_tabarena_dataset
from sklearn.model_selection import train_test_split

BASE = "/Users/dhruvaravind/Desktop/Work/WoodWide/Model_Testing"
TARGET = "RiskPerformance"

test = pd.read_csv(f"{BASE}/test.csv")
train = pd.read_csv(f"{BASE}/train.csv")


def encode_target(df):
    """Map the Good/Bad target to 1/0. Skips rows that are already encoded so
    re-running the script doesn't silently turn every label into 0."""
    if df[TARGET].dtype == object:
        df[TARGET] = (df[TARGET] == "Good").astype(int)
    return df


train = encode_target(train)
test = encode_target(test)

test_labels = test[TARGET]
test_features = test.drop(columns=[TARGET])

test.to_csv(f"{BASE}/test.csv", index=False)
train.to_csv(f"{BASE}/train.csv", index=False)
test_labels.to_csv(f"{BASE}/test_labels.csv", index=False)
test_features.to_csv(f"{BASE}/test_features.csv", index=False)
