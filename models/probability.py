import numpy as np
from scipy.stats import multivariate_normal
from scipy.linalg import inv


def mvn_likelihood(x, mu, Sigma):
    # Work around for multivariate_normal logpdf, since it only accepts dimensions as arrays
    # Reshape arrays to 1 dim
    if mu.ndim != 1 and not isinstance(mu, float):
        (rows, cols) = mu.shape
        x = x.reshape((rows))
        mu = mu.reshape((rows))
    return multivariate_normal.logpdf(x=x, mean=mu, cov=Sigma, allow_singular=True)

def mvn_noise(Sigma):
    mu = np.zeros(shape=(Sigma.shape[0]))
    return multivariate_normal.rvs(mu, Sigma).reshape(Sigma.shape[0], 1)
