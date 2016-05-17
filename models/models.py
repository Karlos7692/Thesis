import numpy as np
from scipy.linalg import pinv
from abc import abstractmethod, ABCMeta
from enum import IntEnum
import models.probability as prob

class Axis(IntEnum):
    rows = 0
    cols = 1
    time = 2


class Model(metaclass=ABCMeta):
    pass


"""
 Linear Dynamic Systems. LG-SSM Kalman Filter etc.
 Columns will be the data points respectively
"""

class LDS(Model):
    init_t = -1

    @classmethod
    def fit(cls, ys, us, initial_kalman_params, iters=5, use_last_to_init=False, debug_limit=10):
        pass

    @abstractmethod
    def predict(self, A, B, C, D, mu_t, u_t):
        pass

    @abstractmethod
    def predict_observable(self, C, D, state_pred, u_t):
        pass

    @abstractmethod
    def predict_state(self, A, B, mu_t, u_t):
        pass

    @abstractmethod
    def predict_covariance(self, A, V, Q):
        pass


"""
Monte Carlo Methods
"""

