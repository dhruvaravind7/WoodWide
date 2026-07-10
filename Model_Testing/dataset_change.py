import pandas as pd
import numpy as np
import openml

from sklearn.model_selection import train_test_split

df = pd.read_csv("/Users/dhruvaravind/Desktop/Work/WoodWide/Model_Testing/Titanic_Dataset/titanic.csv")

# Impute missing Age with the median age for each Pclass/Sex group,
# falling back to the overall median for any group with no known ages.
df['Age'] = df.groupby(['Pclass', 'Sex'])['Age'].transform(lambda x: x.fillna(x.median()))
df['Age'] = df['Age'].fillna(df['Age'].median())

df = df.drop(columns=["Name", "PassengerId"])
df['Cabin'] = df['Cabin'].notna().astype(int)

df['Ticket_Prefix'] = df['Ticket'].str.replace('.', '', regex=False)
df['Ticket_Prefix'] = df['Ticket_Prefix'].str.replace('/', '', regex=False)
df['Ticket_Prefix'] = df['Ticket_Prefix'].str.strip().str.split().str[0]
df['Ticket_Prefix'] = np.where(df['Ticket_Prefix'].str.isdigit(), 'X', df['Ticket_Prefix'])

# 3. Calculate Ticket Group Size (Count how many people share the same ticket)
ticket_counts = df['Ticket'].value_counts()
df['Ticket_Group_Size'] = df['Ticket'].map(ticket_counts)

# 4. Handle 'LINE' tickets (Set their fare to 0)
df['Fare'] = np.where(df['Ticket'] == 'LINE', 0.0, df['Fare'])

# 5. Calculate Individual Fare (Divide group fare by the number of people on the ticket)
df['Fare'] = df['Fare'] / df['Ticket_Group_Size']

# 6. Remove the prefix from the Ticket column, keeping just the ticket number
df['Ticket'] = df['Ticket'].str.split().str[-1]
df['Ticket'] = np.where(df['Ticket'] == 'LINE', 0, df['Ticket'])
df = df.drop(columns=["Ticket_Prefix", "Ticket_Group_Size"])

df = df[[col for col in df.columns if col != "Survived"] + ["Survived"]]
df.to_csv("/Users/dhruvaravind/Desktop/Work/WoodWide/Model_Testing/Titanic_Dataset/titanic_full.csv", index=False)

train, test = train_test_split(df, test_size=0.2, stratify=df["Survived"])
test_labels = test["Survived"]
test_features = test.drop(columns=["Survived"])

train.to_csv("/Users/dhruvaravind/Desktop/Work/WoodWide/Model_Testing/Titanic_Dataset/titanic_train.csv", index=False)
test.to_csv("/Users/dhruvaravind/Desktop/Work/WoodWide/Model_Testing/Titanic_Dataset/titanic_test.csv", index=False)
test_labels.to_csv("/Users/dhruvaravind/Desktop/Work/WoodWide/Model_Testing/Titanic_Dataset/titanic_test_labels.csv", index=False)
test_features.to_csv("/Users/dhruvaravind/Desktop/Work/WoodWide/Model_Testing/Titanic_Dataset/titanic_test_features.csv", index=False)