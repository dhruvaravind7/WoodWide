import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_csv("/Users/dhruvaravind/Desktop/Work/WoodWide/Model_Testing/bank_test.csv")
print(df["Exited"].value_counts())