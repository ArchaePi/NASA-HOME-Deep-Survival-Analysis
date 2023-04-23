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

    # Should we use this time
    #bat = bats[0]
    #print(bat['B0005']['cycle'][0][0]['time'][0][0][0])
 
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
        

    #print(df_list['Voltage_measured'].shape)
    return pd.concat(df_list).reset_index(drop=True)


def get_capacity(df):
    cap = []
    for i in range(len(df.index)):
        if len(df['Capacity'][i]) != 0:
            cap.append(df['Capacity'][i][0])
        else:
            cap.append(0)

    return cap


def scale_data(data):
    norm_data = []
   
    for i in range(len(data)):
        math = (data[i] - min(data))/(max(data) - min(data))
        norm_data.append(math)

    return norm_data


def scale_features(df):
    columns = list(df.columns)
    columns.remove('Battery')
    columns.remove('Time')
    columns.remove('Capacity')

    num_rows = len(df.index)

    for col in columns:
        for i in range(num_rows):
            data_list = list(df[col][i])
            new_data_list = []
            for j in range(len(data_list)):
                math = (data_list[j] - min(data_list))/(max(data_list)-min(data_list))
                new_data_list.append(math)
            df[col][i] = new_data_list
    
    return df


def get_flatten_time(df):
    add_time = 0
    time = []

    for i in range(len(df.index)):
        arr = df['Time'][i]
        for j in range(len(arr)):
            time.append(arr[j] + add_time)
        add_time += max(arr)

    return time


def get_event_occurrence(time, cap):
    init_cap = cap[0]

    percent = 75
    threshold = init_cap/(100/percent)

    events = []
    for i in range(len(cap)):
        if cap[i] > threshold:
            events.append(0)
        else:
            events.append(1)

    data = {'event': events,
            'time': time}
    
    df_y = pd.DataFrame(data)

    return df_y


# Takes extremely long to run because there are 50285 times it needs to loop
def new_scale_features(all_features):
    all_features_scaled = np.empty((5, len(all_features[0]))).tolist()

    for i in range(len(all_features[0])):
        all_features_scaled[0].append((all_features[0][i] - min(all_features[0]))/(max(all_features[0]) - min(all_features[0])))
        all_features_scaled[1].append((all_features[1][i] - min(all_features[1]))/(max(all_features[1]) - min(all_features[1])))
        all_features_scaled[2].append((all_features[2][i] - min(all_features[2]))/(max(all_features[2]) - min(all_features[2])))
        all_features_scaled[3].append((all_features[3][i] - min(all_features[3]))/(max(all_features[3]) - min(all_features[3])))
        all_features_scaled[4].append((all_features[4][i] - min(all_features[4]))/(max(all_features[4]) - min(all_features[4])))

    return all_features_scaled

        
def stretch_capacity(battery_list):
    '''Flatten each feature in each cycle into one cycle.
    Extend the capacity by repeating the capacity value 
    measured for a given cycle n times, where n is the 
    length of the feature array within said cycle.'''
    
    df = get_battery_data(battery_list)
    columns = list(df.columns)
    columns.remove('Battery')
    columns.remove('Time')
    columns.remove('Capacity')

    cap = get_capacity(df)

    all_features = []
    extended_cap = []

    for col in columns:
        feature = []
        for i in range(len(df.index)):
            arr = df[col][i]
            
            if len(all_features) == 0:
                extended_cap.extend([cap[i]]*len(arr))

            for j in range(len(arr)):
                feature.append(arr[i])
                
        all_features.append(feature)

    time_array = get_flatten_time(df)

    data = {'Voltage_measured': all_features[0],
            'Current_measured':all_features[1],
            'Temperature_measured':all_features[2],
            'Current_load':all_features[3],
            'Voltage_load':all_features[4]}
    
    
    df_x = pd.DataFrame(data)

    df_y = get_event_occurrence(time_array, extended_cap)

    return df_x, df_y


def interpolate_capacity(df):
    '''Flatten each feature in each cycle into one cycle. 
    Extend the capacity with values interpolated from 
    each capacity value in each cycle'''

    # To DO


