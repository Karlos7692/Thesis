from abc import abstractmethod
import random
from models.lds.kalman import KalmanFilter

class Particle(object):
    pass

class KalmanParticle(Particle, KalmanFilter):
    pass


class ParticleFilter(object):

    def __init__(self, init_particles, init_weights):
        self.particles = init_particles
        self.weights = init_weights

    def resample(self, likelihood_function, y, resample_function=uniform_wheel_resampling):
        measurements = self.__measure__(likelihood_function, y)


    def __measure__(self, likelihood_function, y):
        return [likelihood_function(p, y) for p in self.particles]


def uniform_wheel_resampling(measurements, particles, normalized=False):
    n = len(measurements)

    m_sum = 1
    if not normalized:
        m_sum = sum(measurements)

    # Get the uniform densities
    densities = [m/m_sum for m in measurements]

    # Draw from uniform distribution
    beta = random.uniform(0, )
    for i in range(n):
        pass


