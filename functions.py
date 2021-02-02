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
    return np.exp(x) / np.sum(np.exp(x), axis=axis, keepdims=True)


def cross_entropy(x, target):
    return (-1/target.shape[0]) * np.sum(target * np.log(x))


def tanh(x):
    return np.tanh(x)


def tanh_derivative(x):
    return 1-np.tanh(x)**2


def compute_pca(data, n_components=2):
    m, n = data.shape

    # mean center the data
    data -= data.mean(axis=0)
    # calculate the covariance matrix
    cov_matrix = np.cov(data, rowvar=False)
    # calculate eigenvectors & eigenvalues of the covariance matrix
    # use 'eigh' rather than 'eig' since R is symmetric,
    # the performance gain is substantial
    evals, evecs = np.linalg.eigh(cov_matrix)
    # sort eigenvalue in decreasing order
    # this returns the corresponding indices of evals and evecs
    idx = np.argsort(evals)[::-1]

    evecs = evecs[:, idx]
    # sort eigenvectors according to same index
    evals = evals[idx]
    # select the first n eigenvectors (n is desired dimension
    # of rescaled data array, or dims_rescaled_data)
    evecs = evecs[:, :n_components]

    return np.dot(evecs.T, data.T).T