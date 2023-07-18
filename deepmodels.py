import numpy as np
from preproc import  load_one_battery, load_multi_battery, load_labels
from config import *
from plotmaster import plot_predictions_v_true

import tensorflow as tf
from tensorflow import keras
from keras.metrics import MeanSquaredError, RootMeanSquaredError, MeanAbsoluteError
from keras import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, TimeDistributed, Conv1D, LSTM, MaxPool1D


def cnn_baseline(X_train, X_test, y_train, y_test):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation = 'relu', input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])))
    model.add(MaxPool2D((2,2)))

    model.add(Conv2D(64, (3, 3), activation = 'relu', padding='same'))
    model.add(MaxPool2D((2,2)))

    model.add(Flatten())
    model.add(Dense(64, activation = 'relu'))
    model.add(Dense(1, activation = 'linear'))

    model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics=['mean_squared_error', 'mean_absolute_error', tf.keras.metrics.RootMeanSquaredError(name='rmse')])
    model.summary()

    model.fit(X_train, y_train, epochs = 50, validation_data=(X_test, y_test))

    test_loss, test_mse, test_mae, test_rmse = model.evaluate(X_test, y_test)
    predictions = model.predict(X_test)
    predictions = predictions.reshape(-1)

    for index, pred in enumerate(predictions):
        print(pred, y_test[index])
    plot_predictions_v_true(predictions, y_test)
    print(f'Mean Squared Error: {round(test_mse, 3)}')
    print(f'Mean Absolute Error: {round(test_mae, 3)}')
    print(f'Root Mean Squared Error: {round(test_rmse, 3)}')

    return test_mse, test_mae, test_rmse

def cnn_improved(X_train, X_test, y_train, y_test):
    model = Sequential()
    model.add(Conv2D(32, (2, 2), activation = 'relu', input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])))
    model.add(MaxPool2D((2,2)))

    model.add(Conv2D(64, (2, 2), activation = 'relu', padding='same'))
    model.add(Conv2D(64, (2, 2), activation = 'relu', padding='same'))
    model.add(MaxPool2D((2,2)))

    model.add(Flatten())
    model.add(Dense(64, activation = 'relu'))
    model.add(Dense(1, activation = 'linear'))

    model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics=['mean_squared_error', 'mean_absolute_error', tf.keras.metrics.RootMeanSquaredError(name='rmse')])
    model.summary()

    model.fit(X_train, y_train, epochs = 20, validation_data=(X_test, y_test))

    test_loss, test_mse, test_mae, test_rmse = model.evaluate(X_test, y_test)
    predictions = model.predict(X_test)
    predictions = predictions.reshape(-1)
    plot_predictions_v_true(predictions, y_test, 'CNN MODEL')
    new_pred = [int(pred) for pred in predictions]
    
    #for i in range(len(new_pred)):
        #print(new_pred[i], '   ',  y_test[i])
    print(f'Mean Squared Error: {round(test_mse, 3)}')
    print(f'Mean Absolute Error: {round(test_mae, 3)}')
    print(f'Root Mean Squared Error: {round(test_rmse, 3)}')

    return test_mse, test_mae, test_rmse

def lstm_baseline(X_train, X_test, y_train, y_test):
    model = Sequential()
    model.add(TimeDistributed(Conv2D(32, (3, 3), activation = 'relu'), input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])))
    model.add(TimeDistributed(MaxPool2D((2,2))))

    model.add(TimeDistributed(Conv2D(64, (3, 3), activation = 'relu', padding='same')))
    model.add(TimeDistributed(MaxPool2D((2,2))))

    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(128))
    model.add(LSTM(64))
    model.add(Dense(64, activation = 'relu'))
    model.add(Dense(1, activation = 'linear'))

    model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics=['mean_squared_error', 'mean_absolute_error', tf.keras.metrics.RootMeanSquaredError(name='rmse')])
    model.summary()

    model.fit(X_train, y_train, epochs = 50, validation_data=(X_test, y_test))

    test_loss, test_mse, test_mae, test_rmse = model.evaluate(X_test, y_test)
    predictions = model.predict(X_test)
    predictions = predictions.reshape(-1)
    new_pred = [int(pred) for pred in predictions]
    
    for i in range(len(new_pred)):
        print(new_pred[i], '   ',  y_test[i])
    print(f'Mean Squared Error: {round(test_mse, 3)}')
    print(f'Mean Absolute Error: {round(test_mae, 3)}')
    print(f'Root Mean Squared Error: {round(test_rmse, 3)}')

    return test_mse, test_mae, test_rmse

def lstm_improved(X_train, X_test, y_train, y_test):
    model = Sequential()
    model.add(TimeDistributed(Conv1D(32, kernel_size = 2, activation = 'relu'), input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])))
    model.add(TimeDistributed(MaxPool1D(pool_size = 2)))

    model.add(TimeDistributed(Conv1D(64, kernel_size = 2, activation = 'relu', padding='same')))
    model.add(TimeDistributed(MaxPool1D(pool_size = 2)))

    model.add(TimeDistributed(Conv1D(128, kernel_size = 2, activation = 'relu', padding='same')))
    model.add(TimeDistributed(MaxPool1D(pool_size = 2)))

    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(128))
    model.add(Dense(64, activation = 'relu'))
    model.add(Dense(1, activation = 'linear'))

    model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics=['mean_squared_error', 'mean_absolute_error', tf.keras.metrics.RootMeanSquaredError(name='rmse')])
    model.summary()

    model.fit(X_train, y_train, epochs = 30, validation_data=(X_test, y_test))

    test_loss, test_mse, test_mae, test_rmse = model.evaluate(X_test, y_test)
    predictions = model.predict(X_test)
    predictions = predictions.reshape(-1)
    plot_predictions_v_true(predictions, y_test, 'CNN LSTM MODEL')
    #new_pred = [int(pred) for pred in predictions]
    
    #for i in range(len(new_pred)):
        #print(new_pred[i], '   ',  y_test[i])
    print(f'Mean Squared Error: {round(test_mse, 3)}')
    print(f'Mean Absolute Error: {round(test_mae, 3)}')
    print(f'Root Mean Squared Error: {round(test_rmse, 3)}')

    return test_mse, test_mae, test_rmse


x_tr, _ = load_multi_battery(['B0006', 'B0007', 'B0018'], 'grey_images', 371)
x_te, _ = load_one_battery('B0005', 'grey_images')

X_train = x_tr.reshape(x_tr.shape[0], x_tr.shape[1], x_tr.shape[2], 1)
X_test = x_te.reshape(x_te.shape[0], x_te.shape[1], x_te.shape[2], 1)

filename = 'labels\\average_drop_labels_zero.txt'
y_train = load_labels(['B0006', 'B0007', 'B0018'], filename)
y_test = load_labels(['B0005'], filename)

mse, mae, rmse = lstm_improved(X_train, X_test, y_train, y_test)
