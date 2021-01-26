import numpy as np


def density_gaussian(x: np.ndarray, mu: np.ndarray, var: float):
    """Compute Multivariate gaussian

    Args:
        x (np.ndarray): Data of size (d, 1)
        mu (np.ndarray): Means of size (d, 1)
        var (float): Variance
    """
    d = len(x)
    cov_matrix = var * np.eye(d)
    return (1 / (np.power(2 * np.pi, d/2) * np.sqrt(np.linalg.det(cov_matrix)))) * np.exp((-1/2) * np.dot(np.dot(np.transpose(x-mu), np.linalg.inv(cov_matrix)), x-mu))


def log_density_gaussian(x: np.ndarray, mu: np.ndarray, var: float):
    d = len(x)
    cov_matrix = var * np.eye(d)
    inverse_cov = np.linalg.inv(cov_matrix)
    return (-d/2)*(np.log(2 * np.pi) + np.log(var)) - np.dot(np.dot(np.transpose(x-mu), inverse_cov), x-mu)/2


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return x > 0


def softmax(x, axis=1):
    return np.exp(x) / np.sum(np.exp(x), axis=axis)


def cross_entropy(x, target):
    return (-1/target.shape[0]) * np.sum(target * np.log(x))


def tanh(x):
    return np.tanh(x)


def tanh_derivative(x):
    return 1-np.tanh(x)**2
