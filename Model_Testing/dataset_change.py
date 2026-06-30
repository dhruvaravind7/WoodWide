import pandas as pd
import numpy as np

df = pd.read_csv("/Users/dhruvaravind/Desktop/Work/WoodWide/Model_Testing/bank.csv")
df = df.sample(frac=0.2, random_state=42, ignore_index=True)

df.to_csv("/Users/dhruvaravind/Desktop/Work/WoodWide/Model_Testing/bank_small.csv", index=False)