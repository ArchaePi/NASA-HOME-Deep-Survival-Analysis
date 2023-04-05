import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from config import *
from auton_survival.estimators import SurvivalModel
from auton_survival.metrics import survival_regression_metric
from sklearn.model_selection import ParameterGrid
from auton_survival.datasets import load_dataset

times, support = load_dataset('SUPPORT')
#print(support)


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

