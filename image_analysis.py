import numpy as np
import pandas as pd
from PIL import Image

import preprocess as p

df = p.get_battery_data(['B0005'])
old_cap = p.get_capacity(df)
cap = p.scale_data(old_cap)
new_df = p.scale_features(df)

def get_color_array(array):
    old_min = 0
    old_max = 1

    new_min = 0
    new_max = 255

    old_range = (old_max - old_min)  
    new_range = (new_max - new_min)  

    color_array = []
    for i in range(len(array)):
        new_value = (((array[i] - old_min) * new_range)/ old_range) + new_min
        color_array.append(int(new_value))

    return color_array

def capacity_image(cap):
    height = 168
    width = 168

    cap = get_color_array(cap)
    arr = np.zeros((height, width, 3), dtype=np.uint8)

    cap_index = 0

    for i in range(height):
        for j in range(width):
            arr[i, j] = (0, 0, cap[i])
            cap_index += 1

    img = Image.fromarray(arr)
    img.save('cap_blue.png')
    img.show()

def find_max(df, feature):
    '''This function is used to find the size/length
      of the largest array within a battery's cycles. 
      This is for the purposes of padding the smaller
      arrays to the same size.'''
    max_length = -1 

    for i in range(len(df.index)):
       length = len(df[feature][i])

       if length > max_length:
           max_length = length
        
    return max_length

def pad_array(max_length, array):
    length = len(array)

    if length == max_length:
        return array
    
    for i in range(max_length - length):
       array.append(0)

    return array

def get_feature_array(df, feature):
    max_length = find_max(df, feature)
    feature_arr = []

    for i in range(len(df.index)):
       color_array = get_color_array(df[feature][i])
       pad_arr = pad_array(max_length, color_array)
       feature_arr.append(pad_arr)

    return feature_arr

def feature_image(df, feature, img_name):
    height = len(df.index)
    width = find_max(df, feature)

    feature_array = get_feature_array(df, feature)
    arr = np.zeros((height, width, 3), dtype=np.uint8)

    for i in range(height):
        for j in range(width):
            arr[i, j] = (feature_array[i][j], 0, 0)

    img = Image.fromarray(arr)
    img.save(img_name)
    img.show()

def bi_feature_image(df, cap, feature, img_name):
    height = len(df.index)
    width = find_max(df, feature)

    feature_array = get_feature_array(df, feature)
    cap = get_color_array(cap)
    arr = np.zeros((height, width, 3), dtype=np.uint8)

    for i in range(height):
        for j in range(width):
            arr[i, j] = (feature_array[i][j], 0, cap[i])

    img = Image.fromarray(arr)
    img.save(img_name)
    img.show()

