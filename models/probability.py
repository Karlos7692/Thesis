import numpy as np
from scipy.stats import multivariate_normal
from scipy.linalg import inv


def mvn_likelihood(x, mu, Sigma):
    return multivariate_normal.logpdf(x=x, mean=mu, cov=Sigma)

def mvn_noise(Sigma):
    mu = np.zeros(shape=(Sigma.shape[0]))
    return multivariate_normal.rvs(mu, Sigma).reshape(Sigma.shape[0], 1)
