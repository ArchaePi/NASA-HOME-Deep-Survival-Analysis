import numpy as np
from preprocess import get_battery_data, get_capacity
from config import battery_list, MAX_CYCLES, MAX_CAPACITY, THRESHOLD

def completion_message(battery, index):
    if (index + 1) != len(battery_list) and (index + 2) != len(battery_list):
        print(f'{battery} Complete! {len(battery_list) - (index + 1)} baterries remain.')
    elif (index + 2) == len(battery_list):
        print(f'{battery} Complete! {1} battery remains.')
    else:
        print(f'{battery} Complete! All batteries have been completed!')

def get_avg_drop(capacity):
    drops = []
    for index, cap in enumerate(capacity):
        if index == 0:
            drop = MAX_CAPACITY - cap
        else:
            drop = capacity[index - 1] - cap

        if drop > 0:
            drops.append(drop)

    return np.mean(drops)

def gen_avg_drop_labels():
    all_labels = []
    for index, battery in enumerate(battery_list):
        battery_data = get_battery_data([battery])
        capacity = get_capacity(battery_data)

        avg_drop = get_avg_drop(capacity)
        labels = []
        for _, cap in enumerate(capacity):
            label = int((cap - THRESHOLD)/avg_drop)
            labels.append(label)

        completion_message(battery, index)
        num_cycles = len(labels)
        nans = [np.nan] * (MAX_CYCLES - num_cycles)
        labels.extend(nans)
        all_labels.append([battery, labels, num_cycles])


    avg_drop_labels = np.vstack(np.asanyarray(all_labels, dtype=object))
    np.savetxt('labels\\average_drop_labels_neg.txt', avg_drop_labels, delimiter=" ", newline = "\n", fmt="%s")
