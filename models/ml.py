import numpy as np

class Gym(object):
    pass


def calculate_residuals(ys, y_pred):
    return ys - y_pred


def calculate_se(ys, y_pred):
    res = calculate_residuals(ys, y_pred)
    return res * res

def mean_line(eps):
    return np.mean(eps) * np.ones(eps.shape)


def mse(ys: np.array, y_pred: np.array):
    n = ys.shape[2]
    return 1.0/n * np.sum((ys - y_pred) * (ys - y_pred))


def rmse(ys, y_pred):
    return np.sqrt(mse(ys, y_pred))


def mape(ys, y_pred):
    n = ys.shape[2]
    total = 0
    for t in range(ys.shape[2]):
        total += np.abs(1.0/ys[:, :, t] * (ys[:, :, t] - y_pred[:, :, t]))
    return 1.0/n * total


def aae(ys, y_pred):
    n = ys.shape[2]
    total = 0
    for t in range(ys.shape[2]):
        total += np.abs((ys[:, :, t] - y_pred[:, :, t]))
    return 1.0/n * total


def mpse(ys, y_pred):
    n = ys.shape[2]
    total = 0
    for t in range(ys.shape[2]):
        total += np.square(1.0/ys[:, :, t] * (ys[:, :, t] - y_pred[:, :, t]))
    return 1.0/n * total