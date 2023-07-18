import re
import os
import random
from PIL import Image
from config import battery_list
import numpy as np
import preprocess as pp

# ------------ HELPER FUNCTIONS ------------ #
def sort_list(a_list):
  sorted_list = [None]* len(a_list)

  for element in a_list:
    index = int(re.split('_', element)[1])
    sorted_list[index] = element
    
  return sorted_list

def pad_image_array(image_array, max_cols, pad_val):
  rows, cols = image_array.shape[0], image_array.shape[1]
  padding = np.asarray([pad_val]*(max_cols - cols))

  padded_image_array = np.empty([rows, max_cols])

  for i in range(rows):
    padded_image_array[i] = np.append(image_array[i], padding)

  return padded_image_array

def clean_labels(orig_labels):
   orig_labels.pop(0)
   end_index = int(orig_labels.pop())

   labels = orig_labels[:end_index]

   labels = [label.translate({ord(i): None for i in '[],'})  for label in labels]
   labels = [int(label) for label in labels]
   return labels


# --------------- LOAD IMAGES --------------- #
def load_one_battery(battery, dataset, max_length = 371):
    # For the first experiment max length is 371
    # For all experiments max length is 653

    img_array = []
    labels = []

    dir_path = 'images//' + dataset + '//' + battery + '//'

    filenames = os.listdir(dir_path)
    sorted_filenames = sort_list(filenames)

    max_cycles_left = len(sorted_filenames) - 1

    for name in sorted_filenames:
        img = Image.open(dir_path + name)
        img = np.array(img).astype(np.float32) / 255.0
        
        if img.shape[1] < max_length:
            img = pad_image_array(img, max_length, -1)

        img_array.append(img)
        labels.append(max_cycles_left - int(re.split('_', name)[1]))
   
    return np.array(img_array), np.array(labels).reshape(-1)

def load_multi_battery(batteries, dataset, max_length = 653):
   # For the first experiment max length is 371
   # For all experiments max length is 653
   img_array = []
   labels = []

   for battery in batteries:
      dir_path = 'images//' + dataset + '//' + battery + '//'

      filenames = os.listdir(dir_path)
      sorted_filenames = sort_list(filenames)

      max_cycles_left = len(sorted_filenames) - 1

      for name in sorted_filenames:
         img = Image.open(dir_path + name)
         img = np.array(img).astype(np.float32) / 255.0
            
         if img.shape[1] < max_length:
            img = pad_image_array(img, max_length, -1)
    
         img_array.append(img)
         labels.append(max_cycles_left - int(re.split('_', name)[1]))

   return np.array(img_array), np.array(labels).reshape(-1)


# --------------- LOAD LABELS --------------- #
def load_labels(batteries, filename):
   all_labels = np.loadtxt(filename, delimiter=' ', dtype=str)
   
   loaded_labels = []
   for _, battery in enumerate(batteries):
      index = battery_list.index(battery)
      labels = clean_labels(list(all_labels[index]))
      
      loaded_labels.extend(labels)

   return np.array(loaded_labels).reshape(-1)


# --------------- PRESET LOAD --------------- #
def train_test(train_bats, test_bats, dataset, max_length = 653):
   if isinstance(train_bats, str):
      x_tr, y_train = load_one_battery(train_bats, dataset, max_length)
   else:
      x_tr, y_train = load_multi_battery(train_bats, dataset, max_length)
 
   if isinstance(test_bats, str):
      x_te, y_test = load_one_battery(test_bats, dataset, max_length)
   else:
      x_te, y_test = load_multi_battery(test_bats, dataset, max_length)
      

   X_train = x_tr.reshape(x_tr.shape[0], x_tr.shape[1] * x_tr.shape[2])
   X_test = x_te.reshape(x_te.shape[0], x_te.shape[1] * x_te.shape[2])
   
   return X_train, X_test, y_train, y_test

def train_test_deep(train_bats, test_bats, dataset, max_length = 653):
   if isinstance(train_bats, str):
      x_tr, y_train = load_one_battery(train_bats, dataset, max_length)
   else:
      x_tr, y_train = load_multi_battery(train_bats, dataset, max_length)
 
   if isinstance(test_bats, str):
      x_te, y_test = load_one_battery(test_bats, dataset, max_length)
   else:
      x_te, y_test = load_multi_battery(test_bats, dataset, max_length)
      

   X_train = x_tr.reshape(x_tr.shape[0], x_tr.shape[1], x_tr.shape[2], 1)
   X_test = x_te.reshape(x_te.shape[0], x_te.shape[1], x_te.shape[2], 1)
   
   return X_train, X_test, y_train, y_test


def sim_censoring(time):
   length = time.shape[0]
   max_time = 200
   e = []

   for i in range(length):
      k = random.randint(0, 1)
      if k == 0:
         e.append(0)
         time[i] = max_time
      else:
         e.append(1)

   return np.asarray(e, dtype = 'int64'), time

def fixed_censoring(time):
   num_cycles = 20
   max_time = 200
   e = []

   for i in range(time.shape[0]):
      if i < num_cycles:
         e.append(0)
         time[i] = max_time
      else:
         e.append(1)
         
   return np.asarray(e, dtype = 'int64'), time

# not yet modified
def train_test_dsm(train_bats, test_bats, dataset, max_length = 653):
   if isinstance(train_bats, str):
      x_tr, y_train = load_one_battery(train_bats, dataset, max_length)
   else:
      x_tr, y_train = load_multi_battery(train_bats, dataset, max_length)
 
   if isinstance(test_bats, str):
      x_te, y_test = load_one_battery(test_bats, dataset, max_length)
   else:
      x_te, y_test = load_multi_battery(test_bats, dataset, max_length)
      
    # need to modify shape maybe and need to add censoring
   X_train = x_tr.reshape(x_tr.shape[0], x_tr.shape[1] * x_tr.shape[2])
   X_test = x_te.reshape(x_te.shape[0], x_te.shape[1] * x_te.shape[2])
   
   return X_train, X_test, y_train, y_test



# ------ CROSS VALIDATION HELPER FUNCTION ------ #
def cross_validation(batteries, index):
    train = batteries.copy()
    test = train.pop(index)
    
    return train, test


# To Do:
# Modify the shape of the train_test_DSM() function and add censoring
# Challenge if padding should be -1 or -1.0 (int or float)



