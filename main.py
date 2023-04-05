import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from preprocess import get_exp_based_df
from config import *
from auton_survival.estimators import SurvivalModel
from auton_survival.metrics import survival_regression_metric
from sklearn.model_selection import ParameterGrid
from auton_survival.datasets import load_dataset

times, support = load_dataset('SUPPORT')
#print(support)
df_x, df_y = get_exp_based_df(experiments[0])
x_tr, x_te, y_tr, y_te = train_test_split(
        df_x, df_y, test_size=0.2, random_state=0)
x_tr, x_val, y_tr, y_val = train_test_split(
        x_tr, y_tr, test_size=0.5, random_state=0)

print(df_x)

# Define parameters for tuning the model
param_grid = {'l2' : [1e-3, 1e-4]}
params = ParameterGrid(param_grid)

# Define the times for model evaluation
#times = np.quantile(y_tr['time'][y_tr['event']==1], np.linspace(0.1, 1, 10)).tolist()

# Perform hyperparameter tuning 
models = []
for param in params:
    model = SurvivalModel('cph', random_seed=2, l2=param['l2'])
    
    # The fit method is called to train the model
    #model.fit(x_tr, y_tr)

    # Obtain survival probabilities for validation set and compute the Integrated Brier Score 
    #predictions_val = model.predict_survival(x_val, times)
    #metric_val = survival_regression_metric('ibs', y_val, predictions_val, times, y_tr)
    #models.append([metric_val, model])
    
# Select the best model based on the mean metric value computed for the validation set
#metric_vals = [i[0] for i in models]
#first_min_idx = metric_vals.index(min(metric_vals))
#model = models[first_min_idx][1]

