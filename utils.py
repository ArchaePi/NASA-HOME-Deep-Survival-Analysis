import os
from config import battery_list

def mean_percentage_error(y_test, y_pred):
    mpe_sum = 0
    for true, pred in zip(y_test, y_pred):
        mpe_sum += ((true - pred)/true)

    mpe = mpe_sum/len(y_test)
    return mpe

def mean_absolute_percentage_error(y_test, y_pred):
    mape_sum = 0
    for true, pred in zip(y_test, y_pred):
        mape_sum += (abs((true - pred))/true)

    mape = mape_sum/len(y_test)
    return mape

def make_image_dir(parent_dir):
    # Example parent_dir: 'images\\new_folder_name\\'
    for _, battery in enumerate(battery_list):
        path = os.path.join(parent_dir, battery)
        os.mkdir(path)
