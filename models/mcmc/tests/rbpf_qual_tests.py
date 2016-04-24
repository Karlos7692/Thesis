import numpy as np
from typing import List
from models.mcmc.pf import KalmanParticle, ParticleFilter
from matplotlib import pyplot as plt
import sys

# System Miss-specification noise
SYSTEM_MISSPEC_NOISE = 0.00000001
EXOGENOUS_MISSPEC_NOISE = 0.25
OBS_MISSPEC_NOISE = 0.01
SSV_MISSPEC_NOISE = 0.01
OSV_MISSPEC_NOISE = 0.001

# State noise
INIT_STATE_MISSPEC_VAR = 0.5
INIT_PRIOR = 100000

# System Noise
SS_VAR_X = 1
SS_VAR_V = 1
SS_VAR_C = 0.1
OBS_VAR = 0.01


def exogenous_movement(n):
    return np.zeros((3, 1, n))


# TODO Create datagen model external to lds and mcmc
def quadratic(a, b, c, n):
    ys = np.zeros((1, 1, n))
    for t in range(n):
        ys[:, :, t] = np.array([[a * t**2 + b*t + c + gauss(OBS_VAR)]])
    return ys


def gauss(sigma):
    return np.random.normal(0, sigma)


def kalman_particle_factory(a, b, c) -> List[KalmanParticle]:
    A = np.array([[1 + gauss(SYSTEM_MISSPEC_NOISE), 1 + gauss(SYSTEM_MISSPEC_NOISE), c + gauss(SYSTEM_MISSPEC_NOISE)],
                  [2*a + gauss(SYSTEM_MISSPEC_NOISE), 0 + gauss(SYSTEM_MISSPEC_NOISE), b + gauss(SYSTEM_MISSPEC_NOISE)],
                  [0, 0, 1]])
    B = np.array([[1 + gauss(EXOGENOUS_MISSPEC_NOISE), 0 + gauss(EXOGENOUS_MISSPEC_NOISE), 0 + gauss(EXOGENOUS_MISSPEC_NOISE)],
                  [0 + gauss(EXOGENOUS_MISSPEC_NOISE), 1 + gauss(EXOGENOUS_MISSPEC_NOISE), 0 + gauss(EXOGENOUS_MISSPEC_NOISE)],
                  [0, 0, 0]])
    C = np.array([[1 + gauss(OBS_MISSPEC_NOISE), 0 + gauss(OBS_MISSPEC_NOISE), 0 + gauss(OBS_MISSPEC_NOISE)]])
    D = np.array([[0 + gauss(EXOGENOUS_MISSPEC_NOISE), 0 + gauss(EXOGENOUS_MISSPEC_NOISE), 0 + gauss(EXOGENOUS_MISSPEC_NOISE)]])

    Q = np.array([[SS_VAR_X, 0, 0 ],
                  [0, SS_VAR_V, 0],
                  [0, 0, SS_VAR_C]])

    R = np.array([[OBS_VAR + gauss(OSV_MISSPEC_NOISE)]])

    init_mu = np.array([[c + gauss(INIT_STATE_MISSPEC_VAR)],
                        [2 * a + b + gauss(INIT_STATE_MISSPEC_VAR)],
                        [1]])
    init_V = np.array([[INIT_PRIOR, 0, 0],
                       [0, INIT_PRIOR, 0],
                       [0, 0, INIT_PRIOR]])

    return KalmanParticle((A, B, C, D, Q, R), init_mu, init_V)


a = 0.3
b = 0.2
c = 5
n_points = 300
N = 1000
ys = quadratic(a, b, c, n_points)
us = exogenous_movement(n_points)

# Kalman Filter, Linear System
kf = kalman_particle_factory(a, b, c)
kf.ys = ys
kf.us = us

#Row Blackwellized PF
rbpf_init_particles = []
for i in range(N):
    rbpf_init_particles.append(kalman_particle_factory(a, b, c))

rbpf = ParticleFilter(init_particles=rbpf_init_particles)
rbpf_y_preds = np.zeros((1, 1, 300))
print("Running RBPF... Printing (t)")
for t in range(n_points):
    print('Processing point {t}.... '.format(t=t))
    rbpf_y_preds[:, :, t] = rbpf.predict(us[:, :, t])
    rbpf.observe(ys[:, :, t], us[:, :, t])

print("Running Kalman Filter")
(kf_y_preds, ll) = kf.filter(likelihood=True)

kf_preds_data = kf_y_preds.flatten()
rbpf_y_data = rbpf_y_preds.flatten()

plt.plot(list(range(n_points)), ys.flatten())
plt.plot(list(range(n_points)), kf_preds_data, color='red')
plt.plot(list(range(n_points)), rbpf_y_data, color='green')
plt.show()

