import numpy as np
import pandas as pd
from preproc import load_one_battery, sim_censoring, fixed_censoring
from auton_survival.datasets import load_dataset
from auton_survival.preprocessing import Preprocessor
from sklearn.model_selection import ParameterGrid
from auton_survival.models.dsm import DeepSurvivalMachines
from sksurv.metrics import concordance_index_ipcw, brier_score, cumulative_dynamic_auc
from plotmaster import plot_predictions_v_true


#outcomes, features = load_dataset('SUPPORT')
#
#cat_feats = ['sex', 'dzgroup', 'dzclass', 'income', 'race', 'ca']
#num_feats = ['age', 'num.co', 'meanbp', 'wblc', 'hrt', 'resp', 
#	     'temp', 'pafi', 'alb', 'bili', 'crea', 'sod', 'ph', 
#             'glucose', 'bun', 'urine', 'adlp', 'adls']
#
#features = Preprocessor().fit_transform(features, cat_feats=cat_feats, num_feats=num_feats)
#
#x, t, e = features.values, outcomes.time.values, outcomes.event.values
#
#n = len(x)
#
#tr_size = int(n*0.70)
#vl_size = int(n*0.10)
#te_size = int(n*0.20)
#
#x_train, x_test, x_val = x[:tr_size], x[-te_size:], x[tr_size:tr_size+vl_size]
#t_train, t_test, t_val = t[:tr_size], t[-te_size:], t[tr_size:tr_size+vl_size]
#e_train, e_test, e_val = e[:tr_size], e[-te_size:], e[tr_size:tr_size+vl_size]
def pull_time():
    data = np.loadtxt('avg.txt', delimiter=' ', dtype=str, max_rows=3)
    B0005_list = data[0][1:]
    B0006_list = data[1][1:]
    B0007_list = data[2][1:]

    B0005_list[0] = B0005_list[0].replace('[', '')
    B0005_list[-1] = B0005_list[-1].replace(']', '')

    B0006_list[0] = B0006_list[0].replace('[', '')
    B0006_list[-1] = B0006_list[-1].replace(']', '')

    B0007_list[0] = B0007_list[0].replace('[', '')
    B0007_list[-1] = B0007_list[-1].replace(']', '')

    B0005_list = [int(element.replace(',', '')) for element in B0005_list]
    B0006_list = [int(element.replace(',', '')) for element in B0006_list]
    B0007_list = [int(element.replace(',', '')) for element in B0007_list]

    return B0005_list, B0006_list, B0007_list

def create_events(cycles_left_list):
    events = []
    for index, cycles_left in enumerate(cycles_left_list):
        if cycles_left == 3000:
            events.append(0)
            cycles_left_list[index] = 100
        else:
            events.append(1)
            if cycles_left < 0:
                cycles_left_list[index] = 0

    return np.asarray(events, dtype = 'int64'), cycles_left_list

def remove_neg(time):
    for indx, t in enumerate(time):
        if t < 0:
            time[indx] = 0
    return time

def get_dsm_data(battery, dataset, max_length = 371):
    features, time = load_one_battery(battery, dataset, max_length = max_length)
    x = features.reshape(features.shape[0], features.shape[1] * features.shape[2])
    e, t = sim_censoring(time)
    #e, t = sim_censoring(time)

    return x, t, e

x_train, t_train, e_train = get_dsm_data('B0005', 'time images')
x_val, t_val, e_val = get_dsm_data('B0006', 'time images')
x_test, t_test, e_test = get_dsm_data('B0007', 'time images')

t = np.concatenate([t_train, t_val, t_test])
e = np.concatenate([e_train, e_val, e_test])

outcomes = pd.DataFrame({'event': e,
      'time': t})
features = pd.DataFrame({'t_train': t_train,
                         'e_train': e_train,
                         't_val': t_val,
                         'e_val': e_val,
                         't_test': t_test,
                         'e_test': e_test})
#print(outcomes)
#outcomes.to_csv('outcomes.csv', index=False)

horizons = [0.25, 0.5, 0.75]
times = np.quantile(outcomes.time[outcomes.event==1], horizons).tolist()
print(times)

param_grid = {'k' : [3],
              'distribution' : ['Weibull'],
              'learning_rate' : [ 1e-4],
              'layers' : [[100]]
             }
params = ParameterGrid(param_grid)

models = []
for param in params:
    model = DeepSurvivalMachines(k = param['k'],
                                 distribution = param['distribution'],
                                 layers = param['layers'])
    # The fit method is called to train the model
    model.fit(x_train, t_train, e_train, iters = 100, learning_rate = param['learning_rate'])
    models.append([[model.compute_nll(x_val, t_val, e_val), model]])
print(models)
best_model = min(models)
model = best_model[0][1]

out_risk = model.predict_risk(x_test, times)
#print(out_risk)
out_survival = model.predict_survival(x_test, times)
#print(out_survival)


#cis = []
brs = []

et_train = np.array([(e_train[i], t_train[i]) for i in range(len(e_train))],
                 dtype = [('e', bool), ('t', float)])
et_test = np.array([(e_test[i], t_test[i]) for i in range(len(e_test))],
                 dtype = [('e', bool), ('t', float)])
et_val = np.array([(e_val[i], t_val[i]) for i in range(len(e_val))],
                 dtype = [('e', bool), ('t', float)])

#print(et_train)
#for i, _ in enumerate(times):
    #cis.append(concordance_index_ipcw(et_train, et_test, out_risk[:, i], times[i])[0])
brs.append(brier_score(et_train, et_test, out_survival, times)[1])
roc_auc = []
for i, _ in enumerate(times):
    roc_auc.append(cumulative_dynamic_auc(et_train, et_test, out_risk[:, i], times[i])[0])
for horizon in enumerate(horizons):
    print(f"For {horizon[1]} quantile,")
    #print("TD Concordance Index:", cis[horizon[0]])
    print("Brier Score:", brs[0][horizon[0]])
    print("ROC AUC ", roc_auc[horizon[0]][0], "\n")

outcomes.to_csv('new_outcomes.csv', index=False)
features.to_csv('features.csv', index=False)