import numpy as np
import preproc as pp
import plotmaster as plt
from config import ren_train_bats, ren_test_bats
# ---------- MODELS AND METRICS ---------- #
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from utils import mean_percentage_error


# ---------------- MODELS ---------------- #
def linear_regression(X_train, X_test, y_train, y_test, save = False):
    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    plt.plot_predictions_v_true(y_pred, y_test, 'Linear Regression', save = save)

    rmse = mean_squared_error(y_test, y_pred, squared = False)
    mpe = mean_percentage_error(y_test, y_pred)

    print(f'Root Mean Squared Error (RMSE): {round(rmse, 3)}')
    print(f'Mean Percentage Error (MPE): {round(mpe, 3)}')

    return rmse, mpe

def support_vector_regression(X_train, X_test, y_train, y_test, save = False):
    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    rmse_list = []
    mpe_list = []

    for k in kernels:
        model = SVR(kernel = k)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        plt.plot_predictions_v_true(y_pred, y_test, 'Support Vector Regression: ' + k.upper(), save = save)

        rmse = mean_squared_error(y_test, y_pred, squared = False)
        mpe = mean_percentage_error(y_test, y_pred)

        print(f'Root Mean Squared Error (RMSE) For The {k.upper()} Kernel: {round(rmse, 3)}')
        print(f'Mean Percentage Error (MPE) For The {k.upper()} Kernel: {round(mpe, 3)}')

        rmse_list.append(rmse)
        mpe_list.append(mpe)
        

    return rmse_list, mpe_list

def mlp_regressor(X_train, X_test, y_train, y_test, save = False):
    model = MLPRegressor(activation = 'relu', solver = 'adam', hidden_layer_sizes = (200, ), random_state = 1, max_iter = 1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    plt.plot_predictions_v_true(y_pred, y_test, 'MLP Regressor', save = save)

    rmse = mean_squared_error(y_test, y_pred, squared = False)
    mpe = mean_percentage_error(y_test, y_pred)

    print(f'Root Mean Squared Error (RMSE): {round(rmse, 3)}')
    print(f'Mean Percentage Error (MPE): {round(mpe, 3)}')

    return rmse, mpe


# Come back to fix run_all and run_all_cross_val 
# also, save all metrics to a text file and save plots as well
def run_all(X_train, X_test, y_train, y_test, save = False):
    rmse_all = []
    mse_all = []
    mae_all = []

    lr_rmse, lr_mse, lr_mae = linear_regression(X_train, X_test, y_train, y_test, save = save)
    mlp_rmse, mlp_mse, mlp_mae = mlp_regressor(X_train, X_test, y_train, y_test, save = save)
    svr_rmse, svr_mse, svr_mae = support_vector_regression(X_train, X_test, y_train, y_test, save = save)

    rmse_all.extend([lr_rmse, mlp_rmse])
    rmse_all.extend(svr_rmse)

    mse_all.extend([lr_mse, mlp_mse])
    mse_all.extend(svr_mse)

    mae_all.extend([lr_mae, mlp_mae])
    mae_all.extend(svr_mae)

    rmse_best, rmse_sig = plt.plot_metrics(rmse_all, 'rmse')
    mse_best, mse_sig = plt.plot_metrics(mse_all, 'mse')
    mae_best, mae_sig = plt.plot_metrics(mae_all, 'mae')

    # rmse_best for example should be a list including the model name, kernel, and numerical value rounded to three digits.
    # then these values need to be evaluated or saved to a text file as is.

    return rmse_best, mse_best, mae_best

def run_all_cross_val(battery_list, dataset):
    best_rmse = []
    best_mse = []
    best_mae = []

    for index, battery in enumerate(battery_list):
        train, test = pp.cross_validation(battery_list, index)

        X_train, X_test, y_train, y_test = pp.train_test(train, test, dataset)
        rmse, mse, mae = run_all(X_train, X_test, y_train, y_test, save = False)

        best_rmse.append(rmse)
        best_mse.append(mse)
        best_mae.append(mae)

    
    # probably should save the best scores
    plt.plot_best(best_rmse, 'rmse')
    plt.plot_best(best_mse, 'mse')
    plt.plot_best(best_mae, 'mae')

x_tr, _ = pp.load_multi_battery(['B0006', 'B0007', 'B0018'], 'time images', 371)
x_te, _ = pp.load_one_battery('B0005', 'time images')

X_train = x_tr.reshape(x_tr.shape[0], x_tr.shape[1] * x_tr.shape[2])
X_test = x_te.reshape(x_te.shape[0], x_te.shape[1] * x_te.shape[2])

filename = 'labels\\average_drop_labels_zero.txt'
y_train = pp.load_labels(['B0006', 'B0007', 'B0018'], filename)
y_test = pp.load_labels(['B0005'], filename)

rmse, mpe = support_vector_regression(X_train, X_test, y_train, y_test)
