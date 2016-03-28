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
            mu_t = self.predict_state(A, B, mu_tm1, u_t, tm1 == init)
            y_t = self.predict_observable(C, D, mu_t, u_t) + r_t

            # Store Data
            self.update_state(t, mu_t, V_t)
            self.observe(t, y_t, u_t)
        return self.y_0tT, self.u_0tT

def line_gen(n):
    ys = np.empty(shape=(1, 1, n))
    us = np.zeros(shape=(2, 1, n))
    for i in range(1, n):
        ys[:, :, i] = np.array([[i]])
    return ys, us

# Parameters
INIT_VAR = 100000
A = np.array([[1, 0.1],
              [0, 1]])
B = np.array([[0, 0],
              [0, 0]])
C = np.array([[1, 0]])
D = np.array([[0, 0]])
Q = np.array([[0.1, 0],
              [0, 0.1]])
R = np.array([[0.1]])
params = (A, B, C, D, Q, R)

# Initial Values
init_mu = np.array([[1],
                    [2]])
init_V = np.array([[INIT_VAR, 0],[0, INIT_VAR]])
init_state = (init_mu, init_V)

# Data Generator
space_gen = KalmanSpaceGenerator(params, init_mu, init_V, 100)
y_0tT, u_0tT = space_gen.generate_data()

# Kalman Filter
kf = KalmanFilter(params, init_mu, init_V)

# Assign data
(ys, us) = line_gen(100)
kf.y_0tT = y_0tT
kf.u_0tT = u_0tT

y_pred, ll, s = kf.kalman_smooth()

l1, = plt.plot(y_0tT.flatten(), label='Kalman Space', color='blue')
l2, = plt.plot(y_pred.flatten(), label='Online Predictions', color='green')
l3, = plt.plot(y_0tT.flatten() - y_pred.flatten(), label='Error', color='red')
plt.legend(handles=[l1, l2])
plt.show()