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
from sklearn import metrics
import matplotlib.pyplot as plt
import os
from pipeline import Pipeline
from model_trainer import ModelTrainer

print(os.getcwd())
os.chdir('../../ml_data/')

def rec_findings(var_list):
    output=''

    for i in range(len(var_list)):
        
        if i != (len(var_list)-1):
            output = output + str(var_list[i]) + ','

        else:
            output = output + str(var_list[i])
        
    return output

"""

penalty{‘l1’, ‘l2’, ‘elasticnet’, None}, default=’l2’

        Specify the norm of the penalty:

            None: no penalty is added;

            'l2': add a L2 penalty term and it is the default choice;

            'l1': add a L1 penalty term;

            'elasticnet': both L1 and L2 penalty terms are added.

dualbool, default=False

    Dual (constrained) or primal (regularized, see also this equation) formulation. 
    Dual formulation is only implemented for l2 penalty with liblinear solver. 
    Prefer dual=False when n_samples > n_features.

tolfloat, default=1e-4

    Tolerance for stopping criteria.

Cfloat, default=1.0

    Inverse of regularization strength; must be a positive float. 
    Like in support vector machines, smaller values specify stronger regularization.

fit_interceptbool, default=True

    Specifies if a constant (a.k.a. bias or intercept) should be added to the decision function.

intercept_scalingfloat, default=1

    Useful only when the solver liblinear is used and self.fit_intercept is set to True. 
    In this case, x becomes [x, self.intercept_scaling], i.e. a “synthetic” feature with constant value equal 
    to intercept_scaling is appended to the instance vector. The intercept becomes intercept_scaling * synthetic_feature_weight.

random_stateint, RandomState instance, default=None

    Used when solver == ‘sag’, ‘saga’ or ‘liblinear’ to shuffle the data. See Glossary for details.
    solver{‘lbfgs’, ‘liblinear’, ‘newton-cg’, ‘newton-cholesky’, ‘sag’, ‘saga’}, default=’lbfgs’

    Algorithm to use in the optimization problem. Default is ‘lbfgs’. To choose a solver, you might want to consider the following aspects:

    For small datasets, ‘liblinear’ is a good choice, whereas ‘sag’ and ‘saga’ are faster for large ones;

    For multiclass problems, all solvers except ‘liblinear’ minimize the full multinomial loss;

    ‘liblinear’ can only handle binary classification by default. To apply a one-versus-rest scheme for the multiclass 
    setting one can wrap it with the OneVsRestClassifier.

    ‘newton-cholesky’ is a good choice for n_samples >> n_features * n_classes, especially with one-hot encoded categorical 
    features with rare categories. Be aware that the memory usage of this solver has a quadratic dependency on n_features * n_classes 
    because it explicitly computes the full Hessian matrix.
    
+-------------------+--------------------------------+------------------------+
| solver            | penalty                        | multinomial multiclass |
+-------------------+--------------------------------+------------------------+
| 'lbfgs'           | 'l2', None                     | yes                    |
+-------------------+--------------------------------+------------------------+
| 'liblinear'       | 'l1', 'l2'                     | no                     |
+-------------------+--------------------------------+------------------------+
| 'newton-cg'       | 'l2', None                     | yes                    |
+-------------------+--------------------------------+------------------------+
| 'newton-cholesky' | 'l2', None                     | yes                    |
+-------------------+--------------------------------+------------------------+
| 'sag'             | 'l2', None                     | yes                    |
+-------------------+--------------------------------+------------------------+
| 'saga'            | 'elasticnet', 'l1', 'l2', None | yes                    |
+-------------------+--------------------------------+------------------------+
    
"""
# Read the data and parameters
input_path = pd.read_csv('credit_card_fraud_dataset.csv')
input_fraud_city_path = pd.read_csv('fraud_rates_city.csv')
bandwidth_plus = 1.20
bandwidth_minus = 0.80
output_path = 'fraud_data_cleaned.csv'
model_input_path = output_path

penalties = ['l1', 'l2']

def model_run(penalty):
    Pipeline(input_path=input_path, 
            input_fraud_city_path=input_fraud_city_path,
            bandwidth_plus=bandwidth_plus,
            bandwidth_minus=bandwidth_minus,
            output_path=output_path)

    X_train, X_test, y_train, y_test = ModelTrainer.data_split(model_input_path=model_input_path)

    LogReg = ModelTrainer.model_trainer(X_train=X_train, y_train=y_train, 
                                        pen_var=pen_var, tol_var=tol_var,dual_var=dual_var,
                        c_var=c_var,fit_intercept_var=fit_intercept_var,intercept_scaling_var=intercept_scaling_var, 
                        class_weight_var=class_weight_var, random_state_var=random_state_var, sol_var=sol_var,
                        max_iter_var=max_iter_var,
                        verbose_var=verbose_var, warm_start_var=warm_start_var, n_jobs_var=n_jobs_var, l1_ratio_var=l1_ratio_var)

    train_score, test_score, f1_value, recall_value, precision_value, auc, tf_pf, tf_pt, tt_pf, tt_pt = ModelTrainer.scoring(LogReg=LogReg, X_train=X_train, X_test=X_test,y_train=y_train,y_test=y_test)

    # Write data
    var_list = [pen_var, tol_var, dual_var, c_var, fit_intercept_var, intercept_scaling_var, 
                class_weight_var, random_state_var, sol_var, max_iter_var,
                verbose_var, warm_start_var, n_jobs_var, l1_ratio_var, train_score, test_score, 
                f1_value, recall_value, precision_value, auc, tf_pf, tf_pt, tt_pf, tt_pt]

    rec_findings(var_list)

    with open('output.txt', 'ab') as f:
        output = rec_findings(var_list)
        f.write((output + '\n').encode('utf-8'))
        f.close()

penalties=['l2', None]

for penalty in penalties:

    if penalty == 'elasticnet':
        pen_var = penalty
        tol_var = 1e-4
        dual_var = False
        c_var = 1.0
        fit_intercept_var = False
        intercept_scaling_var = 1
        class_weight_var = 'balanced'
        random_state_var = 42
        sol_var = 'saga'
        max_iter_var = 1000000
        verbose_var = 0
        warm_start_var = False
        n_jobs_var = None
        l1_ratio_var = 0.2
        model_run(penalty)
    else:
        pen_var = penalty
        tol_var = 1e-4
        dual_var = False
        c_var = 1.0
        fit_intercept_var = False
        intercept_scaling_var = 1
        class_weight_var = 'balanced'
        random_state_var = 42
        sol_var = 'saga'
        max_iter_var = 1000000
        verbose_var = 0
        warm_start_var = False
        n_jobs_var = None
        l1_ratio_var = 0.5
        model_run(penalty)

