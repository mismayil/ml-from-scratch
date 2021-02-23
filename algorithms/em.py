from typing import Tuple, NamedTuple
import numpy as np
from scipy.special import logsumexp
from functions import log_density_gaussian


class GaussianMixture(NamedTuple):
    """Tuple holding a gaussian mixture"""
    mu: np.ndarray  # (K, d) array - each row corresponds to a gaussian component mean
    var: np.ndarray  # (K, ) array - each row corresponds to the variance of a component
    p: np.ndarray  # (K, ) array = each row corresponds to the weight of a component


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment

    """
    def _helper_fn(i, j):
        mask = X[i, :] > 0
        return np.log(mixture.p[j] + 1e-16) + log_density_gaussian(X[i, mask], mixture.mu[j, mask], mixture.var[j])

    n = X.shape[0]
    K = len(mixture.p)
    log_posterior = np.zeros((n, K))

    for i in range(n):
        for j in range(K):
            log_posterior[i, j] = _helper_fn(i, j)

    likelihoods = np.reshape(logsumexp(log_posterior, axis=1), (-1, 1))
    log_posterior = log_posterior - likelihoods
    return np.exp(log_posterior), np.sum(likelihoods)


def mstep(X: np.ndarray, post: np.ndarray, mixture: GaussianMixture,
          min_variance: float = .25) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        post: (n, K) array holding the soft counts
            for all components for all examples
        mixture: the current gaussian mixture
        min_variance: the minimum variance for each gaussian

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    k = post.shape[1]
    n = X.shape[0]
    d = X.shape[1]

    varj = np.zeros((k,))
    mu = np.zeros((k, d))
    sum_pji = np.sum(np.transpose(post), axis=1, keepdims=True)
    pj = np.reshape(sum_pji / X.shape[0], (-1,))

    for j in range(k):
        for i in range(d):
            num = 0
            denom = 0
            for u in range(n):
                if X[u, i] > 0:
                    num += post[u, j] * X[u, i]
                    denom += post[u, j]
            mu[j, i] = num / denom if denom >= 1 else mixture.mu[j, i]

    for j in range(k):
        num = 0
        denom = 0
        for i in range(n):
            mask = X[i, :] > 0
            num += post[i, j] * np.sum(np.power(X[i, mask] - mu[j, mask], 2), keepdims=True)
            denom += len(X[i, mask]) * post[i, j]
        varj[j] = max(np.asscalar(num) / np.asscalar(denom), min_variance)

    return GaussianMixture(mu, varj, pj)


def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
    prev_LL = None
    LL = None
    while (prev_LL is None or ((LL - prev_LL) > (1e-6 * np.abs(LL)))):
        prev_LL = LL
        post, LL = estep(X, mixture)
        mixture = mstep(X, post, mixture)

    return mixture, post, LL
