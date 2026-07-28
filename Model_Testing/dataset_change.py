import json
import os

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

from ucimlrepo import fetch_ucirepo 

BASE = "/Users/dhruvaravind/Desktop/Work/WoodWide/Model_Testing"
DIR = "Forest_Cover"


# fetch dataset 
covertype = fetch_ucirepo(id=31) 
  
# data (as pandas dataframes) 
X = covertype.data.features 
y = covertype.data.targets 
  
df = pd.concat([X, y], axis=1)
df["Cover_Type"] = df["Cover_Type"] - 1

train, test = train_test_split(df, test_size=0.2, random_state=42)
test_features = test.drop(columns="Cover_Type")
test_features.to_csv(f"{BASE}/{DIR}/test_features.csv", index=False)
test_labels = test["Cover_Type"]
test_labels = test_labels - 1
test_labels.to_csv(f"{BASE}/{DIR}/test_labels.csv", index=False)
train.to_csv(f"{BASE}/{DIR}/train.csv", index=False)
test.to_csv(f"{BASE}/{DIR}/test.csv", index=False)