# %%
import os
# %%
# Import packages
import pandas as pd
import numpy as np
import datetime as dt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from datetime import datetime
# %%
df = pd.read_csv('ml_data/credit_card_fraud_dataset.csv')
df
# %%
# List of federal holidays 2023 & 2024
hols_2023 = ["2023-01-02", "2023-01-16", "2023-05-29", "2023-06-19", "2023-07-04", "2023-09-04", "2023-11-10", "2023-11-23", "2023-12-25"]
hols_2024 = ["2024-01-01", "2024-01-15", "2024-05-27", "2024-06-19", "2024-07-04", "2024-09-02", "2024-11-11", "2024-11-24", "2024-12-25"]

for i in range(len(hols_2023)):
    hols_2023[i] = datetime.strptime(hols_2023[i], '%Y-%m-%d')

for i in range(len(hols_2024)):
    hols_2024[i] = datetime.strptime(hols_2024[i], '%Y-%m-%d')
# %%
# Hot encode Location
city_encoded = pd.get_dummies(df["Location"], dtype=int)
df = pd.merge(df, city_encoded, right_index=True, left_index=True, how='inner')

# Hot encode MerchantID
# Result -> Accuracy went down
#merchant_encoded = pd.get_dummies(df['MerchantID'], drop_first=True, dtype=int,prefix='MerchID', prefix_sep='_', )
#df = pd.merge(df, merchant_encoded, right_index=True, left_index=True, how='inner')

# Hot encode Year
df["Date"] = df["TransactionDate"].astype(str).str.split(" ").str[0]
df["Year"] = df["Date"].astype(str).str.split("-").str[0]
df['Year'] = np.where(df["Year"]==2024, 1, 0)

# Hot encode month
#df["Month"] = df["Date"].astype(str).str.split("-").str[1]
#month_encoded = pd.get_dummies(df["Month"], dtype=int, drop_first=True, prefix='Month', prefix_sep='_' )
#df = pd.merge(df, month_encoded, right_index=True, left_index=True, how='inner')
df['Date'] = pd.to_datetime(df['Date']) 
df['DayOfWeek'] = df['Date'].dt.day_name()
dayofweek = pd.get_dummies(df['DayOfWeek'], dtype=int, prefix='WD', prefix_sep='_' ) 
df = pd.merge(df, dayofweek, right_index=True, left_index=True, how='inner')
# df["Day"] = df["Date"].astype(str).str.split("-").str[2]


# Change TransactionType to binary
df['TransactionType'] = np.where(df["TransactionType"]=='refund', 1, 0)

df
# %%
df['quarter'] = df['Date'].dt.quarter
# %%
# Encode Quarters
quarter = pd.get_dummies(df['quarter'], dtype=int, prefix='q', prefix_sep='_' )
df = pd.merge(df, quarter, right_index=True, left_index=True, how='inner') 
# %%
# Drop Unnecessary columns
df = df.drop(columns=['TransactionDate','TransactionID', 'MerchantID','Location', 'Date', 'Year', 'DayOfWeek','quarter'])
df
# %%
# Test train split
X_train, X_test, y_train, y_test = train_test_split(df.drop('IsFraud', axis=1),df['IsFraud'], test_size=0.25, random_state=42)
# %%
pen_var = 'l2'
tol_var = 1e-4
dual_var = False
c_var = 1.0
fit_intercept_var = False
intercept_scaling_var = 1
class_weight_var = 'balanced'
random_state_var = 42
sol_var = 'sag'
max_iter_var = 10000
multi_class_var = 'auto'
verbose_var = 0
warm_start_var = False
n_jobs_var = None
l1_ratio_var = None


# Train the model
LogReg = LogisticRegression(penalty = pen_var, 
                            tol = tol_var,
                            dual=dual_var,
                            C = c_var,
                            fit_intercept = fit_intercept_var, 
                            intercept_scaling = intercept_scaling_var,
                            class_weight = class_weight_var,
                            random_state= random_state_var,
                            solver=sol_var,
                            max_iter=max_iter_var,
                            multi_class= multi_class_var,
                            verbose=verbose_var,
                            warm_start=warm_start_var,
                            n_jobs=n_jobs_var,
                            l1_ratio=l1_ratio_var
                             )
LogReg.fit(X_train, y_train)
# %%
# Scoring
train_score = LogReg.score(X_train, y_train)
print(f"Training Accuracy: {round(train_score*100)}%")
test_score = LogReg.score(X_test, y_test)
print(f"Testing Accuracy: {round(test_score*100)}%")
# %%
# GRIDSEARCH CV
from sklearn.model_selection import GridSearchCV
# %%
# Instantiate the model
# Train the model
LogReg = LogisticRegression()

# Define the parameter grid
param_grid = {
'tol' : [1e-1, 1e-2, 1e-3, 1e-4],
'dual':[False],
'C': [1.0],
'fit_intercept' : [False], 
'intercept_scaling' : [1.0],
'class_weight' : ['balanced'],
'random_state' : [42],
'solver' : ['sag'],
'max_iter':[10000],
'multi_class':['auto'],
'verbose':[2],
'warm_start':[False],
'n_jobs':[-1],
'l1_ratio':[None]
}

logi_cv = GridSearchCV(
    LogReg,
    param_grid,
    cv=5,
    scoring='accuracy'
)
# %%
# Train the model
logi_cv.fit(X_train,y_train)
# %%
print("Best parameters: ", logi_cv.best_params_)
print("Best cross-validated accuracy: ", logi_cv.best_score_)
# %%
