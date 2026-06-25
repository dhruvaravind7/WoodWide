from metaflow import FlowSpec, step
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

class ClassificationFlow(FlowSpec):

    # Loading the dataset
    @step
    def start(self):
        self.p_n_train = np.load("/Users/dhruvaravind/Desktop/Work/WoodWide/Model_Testing/bank/N_train.npy", allow_pickle=True)
        self.p_c_train = np.load("/Users/dhruvaravind/Desktop/Work/WoodWide/Model_Testing/bank/C_train.npy", allow_pickle=True)
        self.p_y_train = np.load("/Users/dhruvaravind/Desktop/Work/WoodWide/Model_Testing/bank/y_train.npy", allow_pickle=True)

        # Loading the cross validation datasets
        self.p_n_val = np.load("/Users/dhruvaravind/Desktop/Work/WoodWide/Model_Testing/bank/N_val.npy", allow_pickle=True)
        self.p_c_val = np.load("/Users/dhruvaravind/Desktop/Work/WoodWide/Model_Testing/bank/C_val.npy", allow_pickle=True)
        self.p_y_val = np.load("/Users/dhruvaravind/Desktop/Work/WoodWide/Model_Testing/bank/y_val.npy", allow_pickle=True)

        # Loading the test datasets
        self.p_n_test = np.load("/Users/dhruvaravind/Desktop/Work/WoodWide/Model_Testing/bank/N_test.npy", allow_pickle=True)
        self.p_c_test = np.load("/Users/dhruvaravind/Desktop/Work/WoodWide/Model_Testing/bank/C_test.npy", allow_pickle=True)
        self.p_y_test = np.load("/Users/dhruvaravind/Desktop/Work/WoodWide/Model_Testing/bank/y_test.npy", allow_pickle=True)

        self.next(self.process_data)

    # Processing the dataset
    @step
    def process_data(self):
        def make_df(n_arr, c_arr):
            n_df = pd.DataFrame(n_arr, columns=[f"num_{i}" for i in range(n_arr.shape[1])])
            c_df = pd.DataFrame(c_arr, columns=[f"col_{i}" for i in range(c_arr.shape[1])])
            return pd.concat([n_df, c_df], axis=1)
        
        self.X_train = make_df(self.p_n_train, self.p_c_train)
        self.Y_train = pd.Series(self.p_y_train.ravel())

        self.X_val = make_df(self.p_n_val, self.p_c_val)
        self.Y_val = pd.Series(self.p_y_val.ravel())

        self.X_test = make_df(self.p_n_test, self.p_c_test)
        self.Y_test = pd.Series(self.p_y_test.ravel())

        self.num_cols = [c for c in self.X_train.columns if c.startswith("num_")]
        self.cat_cols = [c for c in self.X_train.columns if c.startswith("cat_")]

        self.next(self.train_model)
    
    # Training the model
    @step
    def train_model(self):
        preprocesser = ColumnTransformer([
            ("num", RobustScaler(), self.num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), self.cat_cols)
        ])
        self.model = Pipeline([
            ("preprocess", preprocesser),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))
        ])
        self.model.fit(self.X_train, self.Y_train)
        self.next(self.evaluate)
    
    # Cross validation of the model
    @step
    def evaluate(self):
        Y_pred = self.model.predict(self.X_val)
        Y_prob = self.model.predict_proba(self.X_val)[:, 1]
        print(f"AUC: {roc_auc_score(self.Y_val, Y_prob)}")
        print(f"Classification Report:\n{classification_report(self.Y_val, Y_pred)}")
        print(f"Confusion Matrix:\n{confusion_matrix(self.Y_val, Y_pred)}")
        self.next(self.end)
    
    @step
    def end(self):
        print("Flow complete")

ClassificationFlow()