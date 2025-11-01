# Import packages
import pandas as pd
import numpy as np
import datetime as dt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import log_loss
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import precision_recall_curve, average_precision_score
import os

class ModelTrainer:
    def __init__(self, model_input_path, pen_var, tol_var, dual_var, 
                 c_var, fit_intercept_var, intercept_scaling_var, class_weight_var, random_state_var,
                 sol_var,max_iter_var,verbose_var,warm_start_var,n_jobs_var,l1_ratio_var):
        self.model_input_path = model_input_path
        self.pen_var = pen_var
        self.tol_var = tol_var
        self.dual_var = dual_var
        self.c_var = c_var
        self.fit_intercept_var = fit_intercept_var
        self.intercept_scaling_var = intercept_scaling_var
        self.class_weight_var = class_weight_var
        self.random_state_var = random_state_var
        self.sol_var = sol_var
        self.max_iter_var = max_iter_var
        self.verbose_var = verbose_var
        self.warm_start_var = warm_start_var
        self.n_jobs_var = n_jobs_var
        self.l1_ratio_var = l1_ratio_var
    
    def data_split(model_input_path):
        # Test train split
        df = pd.read_csv(model_input_path)
        X_train, X_test, y_train, y_test = train_test_split(df.drop('IsFraud', axis=1),df['IsFraud'], test_size=0.25, random_state=42)
        return X_train, X_test, y_train, y_test

    def model_trainer(X_train, y_train, pen_var, tol_var,dual_var,
                      c_var,fit_intercept_var,intercept_scaling_var, 
                      class_weight_var, random_state_var, sol_var,max_iter_var,
                      verbose_var, warm_start_var, n_jobs_var, l1_ratio_var):
        # Train the model
        LogReg = LogisticRegression(penalty=pen_var, 
                                    tol=tol_var,
                                    dual=dual_var,
                                    C=c_var,
                                    fit_intercept=fit_intercept_var, 
                                    intercept_scaling=intercept_scaling_var,
                                    class_weight=class_weight_var,
                                    random_state=random_state_var,
                                    solver=sol_var,
                                    max_iter=max_iter_var,
                                    verbose=verbose_var,
                                    warm_start=warm_start_var,
                                    n_jobs=n_jobs_var,
                                    l1_ratio=l1_ratio_var
                                    )
        LogReg.fit(X_train, y_train)
        return LogReg
    
    def scoring(LogReg,X_train, X_test, y_train, y_test, ):
        # Scoring
        train_score = LogReg.score(X_train, y_train)
        print(f"Training Accuracy: {round(train_score*100)}%")
        test_score = LogReg.score(X_test, y_test)
        print(f"Testing Accuracy: {round(test_score*100)}%")
        # Predictions for X_test
        y_pred = LogReg.predict(X_test)
        # F1 Score
        f1_value = f1_score(y_true=y_test, y_pred=y_pred, average='weighted')
        print(f"F1 Score: {round(f1_value,2)}")
        # Recall score
        recall_value = recall_score(y_true=y_test, y_pred=y_pred, average='weighted')
        print(f"Recall Score: {round(recall_value,2)}")
        # Precision Score
        precision_value = precision_score(y_true=y_test, y_pred=y_pred, average='weighted')
        print(f"Precision Score: {round(precision_value,2)}")
        #define metrics
        y_pred_proba = LogReg.predict_proba(X_test)[::,1]
        auc = metrics.roc_auc_score(y_test, y_pred_proba)
        print(f"AUC Score: {round(auc,2)}")
        cm = confusion_matrix(y_test, y_pred, labels=LogReg.classes_)
        tf_pf = cm[0][0]
        print(f"Amount Predicted False where Actual False: {tf_pf}")
        tf_pt = cm[0][1]
        print(f"Amount Predicted True where Actual False: {tf_pt}")
        tt_pf = cm[1][0]
        print(f"Amount Predicted False where Actual True: {tt_pf}")
        tt_pt = cm[1][1]
        print(f"Amount Predicted True where Actual True: {tt_pt}")

        return train_score, test_score, f1_value, recall_value, precision_value, auc, tf_pf, tf_pt, tt_pf, tt_pt
