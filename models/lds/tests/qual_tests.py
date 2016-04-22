import numpy as np
from models.lds.datagen import KalmanSpaceGenerator, line_gen
from models.lds.kalman import KalmanFilter
from matplotlib import pyplot as plt

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
kf.ys = y_0tT
kf.us = u_0tT

y_pred, ll, s = kf.smooth_filter()

l1, = plt.plot(y_0tT.flatten(), label='Kalman Space', color='blue')
l2, = plt.plot(y_pred.flatten(), label='Online Predictions', color='green')
l3, = plt.plot(y_0tT.flatten() - y_pred.flatten(), label='Error', color='red')
plt.legend(handles=[l1, l2])
plt.show()