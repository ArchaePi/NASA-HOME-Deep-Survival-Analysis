import numpy as np
import pandas as pd
from auton_survival import datasets
from preprocess import stretch_capacity
from sklearn.model_selection import ParameterGrid
from auton_survival.models.dsm import DeepSurvivalMachines
from auton_survival.preprocessing import Preprocessor
from sklearn.model_selection import train_test_split
from auton_survival.estimators import SurvivalModel
from auton_survival.metrics import survival_regression_metric
#from estimators_demo_utils import plot_performance_metrics

outcomes_sup, features_sup = datasets.load_support()
#print(features_sup)
#print(outcomes_sup)

#time has to be decreasing, as in how much time is left before the event occurs
#and all events should be equal to one.
features, outcomes = stretch_capacity(['B0005'])
print(features)
print(outcomes)

cat_feats = ['sex', 'dzgroup', 'dzclass', 'income', 'race', 'ca']
#num_feats = ['age', 'num.co', 'meanbp', 'wblc', 'hrt', 'resp', 
#	     'temp', 'pafi', 'alb', 'bili', 'crea', 'sod', 'ph', 
#             'glucose', 'bun', 'urine', 'adlp', 'adls']

num_feats = ['Voltage_measured', 'Current_measured', 'Temperature_measured', 'Current_load', 'Voltage_load']

features = Preprocessor().fit_transform(features, cat_feats = [], num_feats=num_feats)

horizons = [0.25, 0.5, 0.75]
times_sup = np.quantile(outcomes_sup.time[outcomes_sup.event==1], horizons).tolist()
#times = np.quantile(outcomes.time[outcomes.event==1], horizons).tolist()


x, t, e = features.values, outcomes.time.values, outcomes.event.values

n = len(x)

tr_size = int(n*0.70)
vl_size = int(n*0.10)
te_size = int(n*0.20)

#x_train, x_test, x_val = x[:tr_size], x[-te_size:], x[tr_size:tr_size+vl_size]
#t_train, t_test, t_val = t[:tr_size], t[-te_size:], t[tr_size:tr_size+vl_size]
#e_train, e_test, e_val = e[:tr_size], e[-te_size:], e[tr_size:tr_size+vl_size]
x_tr, x_te, y_tr, y_te = train_test_split(features, outcomes, test_size=0.2, random_state=1)
x_tr, x_val, y_tr, y_val = train_test_split(x_tr, y_tr, test_size=0.25, random_state=1) 

# Define parameters for tuning the model
param_grid = {'l2' : [1e-3, 1e-4]}
params = ParameterGrid(param_grid)

# Define the times for model evaluation
times = np.quantile(y_tr['time'][y_tr['event']==1], np.linspace(0.1, 1, 10)).tolist()

# Perform hyperparameter tuning 
models = []
for param in params:
    model = SurvivalModel('cph', random_seed=2, l2=param['l2'])
    
    # The fit method is called to train the model
    model.fit(x_tr, y_tr)

    # Obtain survival probabilities for validation set and compute the Integrated Brier Score 
    #predictions_val = model.predict_survival(x_val, times)
    #metric_val = survival_regression_metric('ibs', y_val, predictions_val, times, y_tr)
    #models.append([metric_val, model])
    
# Select the best model based on the mean metric value computed for the validation set
#metric_vals = [i[0] for i in models]
#first_min_idx = metric_vals.index(min(metric_vals))
#model = models[first_min_idx][1]

predictions_te = model.predict_survival(x_te, times)

# Compute the Brier Score and time-dependent concordance index for the test set to assess model performance
results = dict()
results['Brier Score'] = survival_regression_metric('brs', outcomes=y_te, predictions=predictions_te, 
                                                    times=times )
results['Concordance Index'] = survival_regression_metric('ctd', outcomes=y_te, predictions=predictions_te, 
                                                    times=times )
print(results['Brier Score'])
print(results['Concordance Index'])