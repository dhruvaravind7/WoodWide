import pandas as pd
import numpy as np

df = pd.read_csv("/Users/dhruvaravind/Desktop/Work/WoodWide/Model_Testing/bank_serialized.csv")
# Separate the two groups
# df = pd.read_csv("/Users/dhruvaravind/Desktop/Work/WoodWide/Model_Testing/customer_churn_1M.csv")
# df["signup_date"] = pd.to_datetime(df["signup_date"])
col_to_move = df.pop('Row_ID')
df.insert(0, 'Row_ID', col_to_move)
# df["year"] = df["signup_date"].dt.year
# df["month"] = df["signup_date"].dt.month
# df["day"] = df["signup_date"].dt.day
# df.drop(columns=["signup_date", "customer_id"], inplace=True)
df.to_csv("/Users/dhruvaravind/Desktop/Work/WoodWide/Model_Testing/bank_serialized.csv", index=False)