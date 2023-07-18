import numpy as np
import pandas as pd
import os
import PIL
from preprocess import get_battery_data, get_capacity
from config import battery_list, THRESHOLD, colors
import matplotlib.pyplot as plt


def plot_capacity():
    for _, battery in enumerate(battery_list):
        battery_data = get_battery_data([battery])
        capacity = get_capacity(battery_data)

        plt.clf()
        plt.plot(capacity, '-', c = 'cornflowerblue')
        plt.ylabel('Capacity')
        plt.xlabel('Cycles')
        plt.title(battery + ' Capacity')
        plt.savefig('plots\\capacity plots\\' + battery)


def get_EOL_index(capacity):
    for index, cap in enumerate(capacity):
        if cap <= THRESHOLD:
            return index
        
    return len(capacity)

def plot_capacity_EOL():
    for _, battery in enumerate(battery_list):
        battery_data = get_battery_data([battery])
        capacity = get_capacity(battery_data)

        plt.clf()
        plt.plot(capacity, '-', c = 'red')

        eol_index = get_EOL_index(capacity) 
        if eol_index < len(capacity):
            plt.plot(capacity[0:eol_index], '-', c = 'cornflowerblue')
        else:
             plt.plot(capacity, '-', c = 'cornflowerblue')

        plt.ylabel('Capacity')
        plt.xlabel('Cycles')
        plt.title(battery + ': True End-of-life (EOL)')
        plt.savefig('plots\\EOL capacity plots\\' + battery)



def plot_features():
    for _, battery in enumerate(battery_list):
        battery_data = get_battery_data([battery])
        columns = list(battery_data.columns)
        columns.remove('Battery')
        columns.remove('Time')
        columns.remove('Capacity')

        num_cycles = len(battery_data.index)
        for feature in columns:
            feat_name = feature.replace('_', ' ').title()

            plt.clf()
            plt.plot(battery_data[feature][int((5/100)*num_cycles) - 1], c = colors[0])
            plt.plot(battery_data[feature][int((25/100)*num_cycles) - 1], c = colors[1])
            plt.plot(battery_data[feature][int((50/100)*num_cycles) - 1], c = colors[2])
            plt.plot(battery_data[feature][int((75/100)*num_cycles) - 1], c = colors[3])
            plt.plot(battery_data[feature][num_cycles-1], c = colors[4])
            plt.xlabel('Time')
            plt.ylabel(feat_name)
            plt.title(battery + ': ' + feat_name)
            plt.legend(['5%', '25%', '50%', '75%', '100%'])
            plt.savefig('plots\\feature plots\\' + battery + '\\' + feature)
            
def plot_all():
    cap_color = 'cornflowerblue'
    for _, battery in enumerate(battery_list):
        figure, axis = plt.subplots(2, 3, figsize = (12, 7))
        figure.suptitle(battery + ': All Features & Capacity', fontsize= 18)
        row_switch = 0

        battery_data = get_battery_data([battery])
        capacity = get_capacity(battery_data)

        columns = list(battery_data.columns)
        columns.remove('Battery')
        columns.remove('Time')
        columns.remove('Capacity')

        num_cycles = len(battery_data.index)
        for i in range(6):
            if i < 3:
                feat_name = columns[i].replace('_', ' ').title()

                axis[row_switch, i].plot(battery_data[columns[i]][int((5/100)*num_cycles) - 1], c = colors[0])
                axis[row_switch, i].plot(battery_data[columns[i]][int((25/100)*num_cycles) - 1], c = colors[1])
                axis[row_switch, i].plot(battery_data[columns[i]][int((50/100)*num_cycles) - 1], c = colors[2])
                axis[row_switch, i].plot(battery_data[columns[i]][int((75/100)*num_cycles) - 1], c = colors[3])
                axis[row_switch, i].plot(battery_data[columns[i]][num_cycles - 1], c = colors[4])
                axis[row_switch, i].set_xlabel('Time', fontsize = 12)
                axis[row_switch, i].set_ylabel(feat_name, fontsize = 12)
                axis[row_switch, i].legend(['5%', '25%', '50%', '75%', '100%'])
                axis[row_switch, i].set_title(feat_name, fontsize = 15)
            else:
                row_switch = 1

                if i < 5:
                    feat_name = columns[i].replace('_', ' ').title()

                    axis[row_switch, (i-3)].plot(battery_data[columns[i]][int((5/100)*num_cycles) - 1], c = colors[0])
                    axis[row_switch, (i-3)].plot(battery_data[columns[i]][int((25/100)*num_cycles) - 1], c = colors[1])
                    axis[row_switch, (i-3)].plot(battery_data[columns[i]][int((50/100)*num_cycles) - 1], c = colors[2])
                    axis[row_switch, (i-3)].plot(battery_data[columns[i]][int((75/100)*num_cycles) - 1], c = colors[3])
                    axis[row_switch, (i-3)].plot(battery_data[columns[i]][num_cycles - 1], c = colors[4])
                    axis[row_switch, (i-3)].set_xlabel('Time', fontsize = 12)
                    axis[row_switch, (i-3)].set_ylabel(feat_name, fontsize = 12)
                    axis[row_switch, (i-3)].legend(['5%', '25%', '50%', '75%', '100%'])
                    axis[row_switch, (i-3)].set_title(feat_name, fontsize = 15)
                else:
                    axis[row_switch, (i-3)].plot(capacity, c = cap_color)
                    axis[row_switch, (i-3)].set_xlabel('Cycles', fontsize = 12)
                    axis[row_switch, (i-3)].set_ylabel('Capacity', fontsize = 12)
                    axis[row_switch, (i-3)].set_title('Capacity', fontsize = 15)

        plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.5,
                    hspace=0.5)
        plt.savefig('plots\\all plots\\' + battery)
        plt.close()
       

# should return the best metric
def plot_metrics(metrics, metric_name, save = False):
    pass

# should return the best metric
def plot_best(metrics, metric_name, save = False):
    pass

def plot_predictions_v_true(y_pred, y_true, title, save = False):
    plt.plot(y_true, '-', c = 'cornflowerblue')
    plt.plot(y_pred, '-', c = 'red')
    plt.legend(['TRUE', 'PREDICTION'])
    plt.xlabel('Datapoints')
    plt.ylabel('Cycles Remaining')
    plt.title(title)
    plt.show()

# QUESTIONS & TO DO:
# 1. Compare to other methods [Today]
# 2. Check for anomaly drops [ Tommorow]
# 3. Read up on each experiment [Today]
# 4. Ultilize the batteries capacity to understand the drops in capacity to reach EOL criteria [Tommorow]
# 5. Compare voltages at deeper levels
# 6. Plot metrics