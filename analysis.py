from preprocess import get_battery_data

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import *

df = get_battery_data(['B0005'])

def get_capacity(df):
    cap = []
    for i in range(len(df)):
        cap.append(df['Capacity'][i][0])

    return cap

def plot_feats(row=0):
    cap = get_capacity(df)
    fig, axes = plt.subplots(2, 3)

    x, y = 0, 0
    for i in range(5):
        axes[x, y].plot(df['Time'][0], df[params[i]][0], color = 'black')
        axes[x, y].plot(df['Time'][55], df[params[i]][55], color = 'grey')
        axes[x, y].plot(df['Time'][167], df[params[i]][167], color = 'red')


        axes[x, y].legend(['First', 'Middle', 'Last'])
        axes[x, y].set_xlabel('Time')
        axes[x, y].set_ylabel(params[i])

        if y == 2:
            x, y = 1, 0
        else:
            y += 1

    axes[1, 2].plot(cap, color = 'red')

    plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.5,
                    hspace=0.5)
    plt.show()

    

plot_feats(0)