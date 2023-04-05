import os
from scipy.io import loadmat

import numpy as np
import pandas as pd


def preprocess():
    path = "c:/Users/archa/battery_dataset/5. Battery Data Set"
    datasets = os.listdir(path)
    bats = []

    for dataset in datasets:
        bat_file_names = list(os.listdir(f'{path}/{dataset}'))
        bat_paths = ([f'{path}/{dataset}/{x}' for x in bat_file_names if ('.mat' in x)])
        bats.extend(loadmat(bp) for bp in bat_paths)

    params = ['Voltage_measured', 'Current_measured', 'Temperature_measured',
              'Current_load', 'Voltage_load', 'Time', 'Capacity', ]
    
    cols = params.copy()
    cols.insert(0, 'Battery')
    
    df = pd.DataFrame(columns = cols)
 
    for bat in bats:
        bat_preproc = np.empty((len(params)+1, 0)).tolist()
        bat_name = list(bat.keys())[-1]
        cycle = bat[bat_name]['cycle']
        data = cycle[0][0]['data']
       

        for i in range(len(data[0])):
            oper_type = cycle[0][0]['type'][0][i]
            
            if oper_type == 'discharge':
                bat_preproc[0].append(bat_name)
                for j in range(len(params)):
                    bat_preproc[j+1].append(data[0][i][params[j]][0][0][0])
                              
                    
        for row in range(len(bat_preproc[0])):
            df.loc[len(df.index)] = [bat_preproc[0][row], bat_preproc[1][row], bat_preproc[2][row], bat_preproc[3][row],
                                     bat_preproc[4][row], bat_preproc[5][row], bat_preproc[6][row], bat_preproc[7][row]]
        
    return df       

def get_battery_data(battery_list):
    df = preprocess()
    length = len(battery_list)
    df_list = []

    for i in range(length):
        df_list.append(df[df['Battery'] == battery_list[i]])
        

    return pd.concat(df_list).reset_index(drop=True)

