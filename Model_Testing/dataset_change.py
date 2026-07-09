import pandas as pd
import numpy as np
import openml

from sklearn.model_selection import train_test_split

df = pd.read_csv("/Users/dhruvaravind/Desktop/Work/WoodWide/Model_Testing/Titanic_Dataset/titanic.csv")
df["Embarked"] = df["Embarked"].fillna("S")
df.to_csv("/Users/dhruvaravind/Desktop/Work/WoodWide/Model_Testing/Titanic_Dataset/titanic.csv", index=False)