import random
import numpy as np
from abc import abstractmethod
from models.lds.kalman import KalmanFilter, Axis
from typing import List


def unif_w_r(densities, particles):
    """
    Uniform wheel resampling. Resample based on uniform distribution as a circular buffer
    :param densities: the pdfs of respective distribution
    :param particles: the particles associated with the densities
    :return:
    """
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
        new_particles.append(particles[density_index].copy())
    return new_particles


def logp_identity(measurements):
    return all([m < 0 for m in measurements])


class Particle(object):

    def __init__(self):
        self.label = ""

    def set_label(self, label):
        self.label = label
        return self

    @abstractmethod
    def project(self, n):
        pass

    @abstractmethod
    def particle_predict(self, u_t) -> np.array:
        pass

    @abstractmethod
    def measure_likelihood(self, y: np.array, u: np.array) -> float:
        pass

    @abstractmethod
    def observe(self, t, y_t, u_t):
        pass

    @abstractmethod
    def copy(self):
        pass


class KalmanParticle(Particle, KalmanFilter):

    def __init__(self, init_params, init_mu, init_V):
        Particle.__init__(self)
        KalmanFilter.__init__(self, init_params, init_mu, init_V)

    def project(self, n):
        return KalmanFilter.project(self, n)

    def observe(self, t, y_t, u_t):
        KalmanFilter.observe(self, t, y_t, u_t)

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

        # Update observation
        self.observe(tm1+1, y, u)

        # Update State
        return KalmanFilter.update(self, tm1+1, mu_pred, V_pred, y, y_pred, compute_likelihood=True)

    def copy(self):
        As = np.copy(self.As)
        Bs = np.copy(self.Bs)
        Cs = np.copy(self.Cs)
        Ds = np.copy(self.Ds)
        Qs = np.copy(self.Qs)
        Rs = np.copy(self.Rs)

        mus = np.copy(self.mus)
        Vs = np.copy(self.Vs)

        ys = np.copy(self.ys)
        us = np.copy(self.us)

        init_params = (As[:, :, 0], Bs[:, :, 0], Cs[:, :, 0], Ds[:, :, 0], Qs[:, :, 0], Rs[:, :, 0])
        init_mu = mus[:, :, 0]
        init_V = Vs[:, :, 0]

        particle = KalmanParticle(init_params, init_mu, init_V)
        particle.As = As
        particle.Bs = Bs
        particle.Cs = Cs
        particle.Ds = Ds
        particle.Qs = Qs
        particle.Rs = Rs

        particle.mus = mus
        particle.Vs = Vs

        particle.ys = ys
        particle.us = us

        particle.label = self.label
        return particle


class ParticleFilter(object):

    def __init__(self, init_particles: List[Particle], min_sample_space: int, resample_function=unif_w_r,
                 eta=1.0):
        self.particles = init_particles
        self.weights = [1.0/len(init_particles)] * len(init_particles)
        self.min_sample_space = min_sample_space
        self.resample_function = resample_function
        self.eta = eta

    def project(self, n):
        return sum([self.weights[i] * p_i.project(n) for i, p_i in enumerate(self.particles)])

    def predict(self, u_t) -> np.array:
        return sum([self.weights[i] * p_i.particle_predict(u_t) for i, p_i in enumerate(self.particles)])

    def observe(self, y: np.array, u: np.array):
        densities = self.measure(y, u)
        weights = [self.weights[i] * d for i, d in enumerate(densities)]
        w_norm = sum(weights)
        self.weights = [w/w_norm for w in weights]

        # Re-sample if the sample-efficiency has fallen below minimum sample space size
        if self.efficiency() < self.min_sample_space:
            print("Resampling...")
            self.particles = self.resample_function(self.weights, self.particles)
            self.weights = [1.0/len(self.particles)] * len(self.particles)

    def measure(self, y: np.array, u: np.array):
        densities = [p.measure_likelihood(y, u) for i, p in enumerate(self.particles)]
        if logp_identity(densities):
            densities = [np.exp(m * 1/self.eta) for m in densities]
        return densities

    def draw(self):
        return self.resample_function(self.weights, self.particles)

    def efficiency(self):
        return 1/sum([w ** 2 for w in self.weights])
