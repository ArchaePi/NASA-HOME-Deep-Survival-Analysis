from preprocess import split_data, get_battery_data

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
        axes[x, y].plot(df['Time'][16], df[params[i]][16], color = 'cornflowerblue')
        axes[x, y].plot(df['Time'][50], df[params[i]][50], color = 'darkorange')
        axes[x, y].plot(df['Time'][84], df[params[i]][84], color = 'yellow')
        axes[x, y].plot(df['Time'][117], df[params[i]][117], color = 'purple')
        axes[x, y].plot(df['Time'][151], df[params[i]][151], color = 'yellowgreen')


        axes[x, y].legend(['cycle 10%', 'cycle 30%', 'cycle 50%', 'cycle 70%', 'cycle 90%'])
        axes[x, y].set_xlabel('Time')
        axes[x, y].set_ylabel(params[i])

        if y == 2:
            x, y = 1, 0
        else:
            y += 1

    axes[1, 2].plot(cap, color = 'blue')

    plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.5,
                    hspace=0.5)
    plt.show()

    

plot_feats(0)