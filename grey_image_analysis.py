import numpy as np
from numpy import asarray
from numpy import savetxt
import pandas as pd
from PIL import Image

import preprocess as p


def get_color_array(array):
    '''Turns an array of integer or float values 
    into an array of integers with values 
    between 0 and 255'''

    old_min = min(array)
    old_max = max(array)

    new_min = 0
    new_max = 255

    old_range = (old_max - old_min)  
    new_range = (new_max - new_min)  

    color_array = []
    for i in range(len(array)):
        new_value = (((array[i] - old_min) * new_range)/ old_range) + new_min
        color_array.append(int(new_value))

    return color_array

def capacity_image(cap, file_name):
    height = len(cap)
    width = height

    cap = get_color_array(cap)
    arr = np.zeros((height, width), dtype=np.uint8)

    #cap_index = 0

    for i in range(height):
        for j in range(width):
            arr[i, j] = cap[i]
            #cap_index += 1

    img = Image.fromarray(arr)
    img.save(file_name)
    #img.show()

def get_cycle_array(df):
    columns = list(df.columns)
    columns.remove('Battery')
    columns.remove('Time')
    columns.remove('Capacity')

    cycle_arr = []
    for i in range(len(df.index)):
        feature_arr = []
        for feature in columns:
            feature_arr.append(get_color_array(df[feature][i]))
        
        cycle_arr.append(feature_arr)
      

    return cycle_arr

def feature_image(cycle_array, cycle_num, cap, file_name):
    height = len(cycle_array[cycle_num]) + 1
    width = len(cycle_array[cycle_num][0])

    color_cap = get_color_array(cap)
    arr = np.zeros((height, width), dtype=np.uint8)

    for i in range(height):
        for j in range(width):
            if i < height - 1:
                arr[i, j] = cycle_array[cycle_num][i][j]
            else:
                arr[i, j] = color_cap[cycle_num]

    img = Image.fromarray(arr)
    img.save(file_name)
    #img.show()

def get_grey_images():
    battery_list = ["B0005", "B0006", "B0007", "B0018",
                    "B0025", "B0026", "B0027", "B0028",
                    "B0029", "B0030", "B0031", "B0032",
                    "B0033", "B0034", "B0036",
                    "B0038", "B0039", "B0040",
                    "B0041", "B0042", "B0043", "B0044",
                    "B0045", "B0046", "B0047", "B0048",
                    "B0049", "B0050", "B0051", "B0052",
                    "B0053", "B0054", "B0055", "B0056"]
    
    count_per_battery = []
    
    for battery in battery_list:
        df = p.get_battery_data([battery])
        old_cap = p.get_capacity(df)
        cap = p.scale_data(old_cap)
        new_df = p.scale_features(df)

        cap_file_name = '\cap_' + battery + '.png'

        capacity_image(cap, '.\images\grey_images\\'+ battery + cap_file_name)
        cycle_array = get_cycle_array(new_df)
        count_per_battery.append(len(cycle_array))

        for i in range(len(cycle_array)):
            file_name = '\cycle_' + str(i) +'_' + battery + '.png'
            feature_image(cycle_array, i, cap, '.\images\grey_images\\' + battery + file_name)

    data = asarray(count_per_battery)
    savetxt('image_count.csv', data, delimiter=',')





count = get_grey_images()



