import random
import numpy as np
from abc import abstractmethod
from models.lds.kalman import KalmanFilter, Axis
from typing import List


def uniform_wheel_resampling(densities, particles):
    n = len(densities)

    # Draw from uniform distribution
    beta = 0
    density_index = 0
    new_particles = []
    mw = max(densities)
    for i in range(n):
        beta += random.random() * 2.0 * mw
        while beta > densities[density_index]:
            beta -= densities[density_index]
            density_index = (density_index + 1) % n
        new_particles.append(particles[density_index])
    return new_particles


class Particle(object):

    @abstractmethod
    def particle_predict(self, u_t) -> np.array:
        pass

    @abstractmethod
    def measure_likelihood(self, y: np.array, u: np.array) -> float:
        pass


class KalmanParticle(Particle, KalmanFilter):

    def particle_predict(self, u_t) -> np.array:

        # Current time t (mus indexed at +1, for initial mu value)
        t = self.mus.shape[Axis.time] - 2
        (A, B, C, D, Q, R) = self.parameters(t)
        (mu, V) = self.state(t)

        mu_pred = self.predict_state(A, B, mu, u_t)
        return self.predict_observable(C, D, mu_pred, u_t)

    def measure_likelihood(self, y: np.array, u: np.array) -> float:

        # Time before update. (mus indexed at +1 for initial mu)
        tm1 = self.mus.shape[Axis.time] -2

        (A, B, C, D, Q, R) = self.parameters(tm1)
        (mu, V) = self.state(tm1)

        (y_pred, mu_pred) = self.predict(A, B, C, D, mu, u)
        V_pred = self.predict_covariance(A, V, Q)

        return KalmanFilter.update(self, tm1+1, mu_pred, V_pred, y, y_pred, compute_likelihood=True)


class ParticleFilter(object):

    def __init__(self, init_particles: List[Particle]):
        self.particles = init_particles
        self.EPS = np.finfo(float).eps

    def predict(self, u_t) -> np.array:
        return sum([p_i.particle_predict(u_t) for p_i in self.particles])/len(self.particles)

    def observe(self, y: np.array, u: np.array, resample_function=uniform_wheel_resampling):
        weights = self.__measure__(y, u)
        self.particles = resample_function(weights, self.particles)

    def __measure__(self, y: np.array, u: np.array):
        measurements = [p.measure_likelihood(y, u) for i, p in enumerate(self.particles)]
        if logp_identity(measurements):
            measurements = [np.exp(m) for m in measurements]
        return measurements


def logp_identity(measurements):
    return all([m < 0 for m in measurements])