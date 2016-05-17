import numpy as np
from abc import abstractmethod
from models.lds.kalman import KalmanFilter
import models.probability as prob
import matplotlib.pyplot as plt

BIG_NUMBER = 1000000000000

class SpaceGenerator(object):

    @abstractmethod
    def generate_data(self):
        pass


def oscillating_u(t):
        return np.array([[t % 2], [abs(t % 2 - 1)]])


class KalmanSpaceGenerator(KalmanFilter, SpaceGenerator):

    def __init__(self, init_params, init_mu, init_V, n_observations):
        self.n_obs = n_observations
        super(KalmanSpaceGenerator, self).__init__(init_params, init_mu, init_V, fixed=True)

    def generate_data(self):
        init = KalmanFilter.init_t
        for tm1 in range(init, self.n_obs-1):
            (A, B, C, D, Q, R) = self.parameters(tm1)
            (mu_tm1, V_t) = self.state(tm1)
            t = tm1 + 1

            # Get Noise
            q_t = prob.mvn_noise(Q)
            r_t = prob.mvn_noise(R)

            # Conditional Variable
            u_t = np.zeros((2,1))

            # Predict next state and add noise
            mu_t = self.predict_state(A, B, mu_tm1, u_t) + q_t
            y_t = self.predict_observable(C, D, mu_t, u_t) + r_t

            # Store Data
            self.update_state(t, mu_t, V_t)
            self.observe(t, y_t, u_t)
        return self.ys, self.us


def line_gen(n):
    ys = np.empty(shape=(1, 1, n))
    us = np.zeros(shape=(2, 1, n))
    for i in range(1, n):
        ys[:, :, i] = np.array([[i]])
    return ys, us