from models.mcmc.pf import Particle, ParticleFilter
from models import probability as prob
from matplotlib import pyplot as plt
import random
import numpy as np

# Four landmarks
world = np.array([[90, 20],
                 [25, 30],
                 [60, 100],
                 [13, 12]])

world_size = 100

robot_actual_location = np.array([76, 31])

SENSE_NOISE = 4


class RobotParticle(Particle):

    def __init__(self, x, y):
        super().__init__()
        self.location = np.array([x, y])
        self.sense_noise = SENSE_NOISE

    def dist(self, point):
        return np.linalg.norm(self.location - point)

    def measure_likelihood(self, y: np.array, u: np.array) -> float:

        # Measure distance of particle from landmarks.
        belief_dists = np.array([self.dist(obs) for obs in world])
        return sum([prob.mvn_likelihood(belief_dists[i], y[i], self.sense_noise) for i in range(len(y))])

    def particle_predict(self, u_t) -> np.array:
        return self.location

    def copy(self):
        return RobotParticle(self.location[0], self.location[1])

    def __str__(self):
        return '[x={x}, y={y}]'.format(x=self.location[0], y=self.location[1])


def p_to_point(particle):
    return particle.location[0], particle.location[1]


def ps_to_points(particles):
    return [p_to_point(p)[0] for p in particles], [p_to_point(p)[1] for p in particles]


def dist(landmark):
    return np.linalg.norm(robot_actual_location - landmark)


def sense():
        return np.array([dist(landmark) + np.random.normal(0.0, SENSE_NOISE) for landmark in world])

N = 3000
init_particles = [RobotParticle(random.uniform(0, world_size), random.uniform(0, world_size)) for i in range(N)]
(x, y) = ps_to_points(init_particles)
plt.scatter(x, y)
z = [point[0] for point in world]
z2 = [point[1] for point in world]
plt.scatter(z, z2, c='red')
plt.scatter([robot_actual_location[0]], [robot_actual_location[1]], c='purple')
plt.show()
pf = ParticleFilter(init_particles, 5)
pf.observe(sense(), None)
pf_paricles = pf.draw()
(x, y) = ps_to_points(pf_paricles)
plt.scatter(x, y, c='blue')
plt.scatter(z, z2, c='red')
plt.scatter([robot_actual_location[0]], [robot_actual_location[1]], c='purple')
plt.show()
print(pf.predict(None))
