import numpy as np
"""
This module defines utilities and classes to run experiments seemlessly.
"""


def calculate_residuals(ys, y_pred):
    return ys - y_pred


def mean_line(eps):
    return np.mean(eps) * np.ones(eps.shape)


def mse(ys: np.array, y_pred: np.array):
    n = ys.shape[2]
    return 1.0/n * np.sum((ys - y_pred) * (ys - y_pred))


def rmse(ys, y_pred):
    np.sqrt(mse(ys, y_pred))


def aae(ys, y_pred):
    n = ys.shape[2]
    total = 0
    for t in range(ys.shape[2]):
        total += np.abs(1.0/ys[:, :, t] * (ys[:, :, t] - y_pred[:, :, t]))
    return 1.0/n * total


def mpse(ys, y_pred):
    n = ys.shape[2]
    total = 0
    for t in range(ys.shape[2]):
        total += np.square(1.0/ys[:, :, t] * (ys[:, :, t] - y_pred[:, :, t]))
    return 1.0/n * total