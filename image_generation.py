import numpy as np
from PIL import Image
from config import battery_list
import preprocess as p

# DEPRECATED
def rescale_color_array(color_array):
    max_val = 653
    val = color_array[len(color_array) - 1]

    if len(color_array) < max_val:
        diff = max_val - len(color_array)
        color_array.extend([val]*diff)

    return color_array

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

    for i in range(height):
        for j in range(width):
            arr[i, j] = cap[i]

    #img = Image.fromarray(arr)
    #img.save(file_name)
    #img.show()

def get_cycle_array(df):
    columns = list(df.columns)
    columns.remove('Battery')
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
    return height, width

def get_grey_images():
    count_per_battery = []
    img_stats = []
    
    for battery in battery_list:
        battery_data = p.get_battery_data([battery])
        cap = p.get_capacity(battery_data)

        
        cycle_array = get_cycle_array(battery_data)
        count_per_battery.append([battery, str(len(cycle_array))])


        for i in range(len(cycle_array)):
            file_name = '\cycle_' + str(i) +'_' + battery + '.png'
            height, width = feature_image(cycle_array, i, cap, 'images\\grey_images\\' + battery + file_name)
            img_stats.append([file_name, str(height), str(width), str(cap[i])])


    return np.asarray(img_stats), np.asarray(count_per_battery)





#stats, count = get_grey_images()
#
#stats = np.vstack(stats)
#count = np.vstack(count)
#np.savetxt('pre_scaled_stats.txt', stats, delimiter=" ", newline = "\n", fmt="%s")
#np.savetxt('pre_scaled_counts.txt', count, delimiter=" ", newline = "\n", fmt="%s")