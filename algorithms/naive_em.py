"""Mixture model using Naive Bayes"""
from typing import Tuple, NamedTuple
import numpy as np
from functions import density_gaussian


class GaussianMixture(NamedTuple):
    """Tuple holding a gaussian mixture"""
    mu: np.ndarray  # (K, d) array - each row corresponds to a gaussian component mean
    var: np.ndarray  # (K, ) array - each row corresponds to the variance of a component
    p: np.ndarray  # (K, ) array = each row corresponds to the weight of a component


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment
    """
    n = X.shape[0]
    K = len(mixture.p)
    posterior = np.zeros((n, K))

    for i in range(n):
        for j in range(K):
            posterior[i, j] = mixture.p[j] * density_gaussian(X[i], mixture.mu[j], mixture.var[j])

    likelihoods = np.sum(posterior, axis=1, keepdims=True)
    posterior = posterior / likelihoods
    return posterior, np.sum(np.log(likelihoods))


def mstep(X: np.ndarray, post: np.ndarray) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    post_nk = np.transpose(post)
    sum_pji = np.sum(post_nk, axis=1, keepdims=True)
    mu = np.dot(post_nk, X) / sum_pji
    pj = np.reshape(sum_pji / X.shape[0], (-1,))
    k = post.shape[1]
    varj = np.zeros((k,))
    d = X.shape[1]

    for j in range(k):
        varj[j] = np.asscalar(np.dot(post_nk[j], np.sum(np.power(X - mu[j], 2), axis=1, keepdims=True))) / (d * np.asscalar(sum_pji[j]))

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
    while (prev_LL is None or LL - prev_LL > 1e-6 * np.abs(LL)):
        prev_LL = LL
        post, LL = estep(X, mixture)
        mixture = mstep(X, post)

    return mixture, post, LL
