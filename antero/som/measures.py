import numpy as np

from scipy.stats import ks_2samp


def umatrix(weights: np.ndarray, d: float = 1) -> np.ndarray:
    """
    Generate u-matrix from SOM weights.

    :param weights: SOM weight matrix with shape (..., 1, features)
    :param d: maximum distance to considered neighbors
    :return: u-matrix
    """
    map_shape = weights.shape[:-2]
    distances = np.zeros(map_shape)
    indices = np.indices(map_shape)

    for i in zip(*indices.reshape(indices.shape[0], -1)):
        diff = indices - np.array(i).reshape((-1,) + tuple(1 for _ in map_shape))
        ix_d = np.linalg.norm(diff, axis=0)
        locs = np.where(np.logical_and(ix_d > 0, ix_d <= d))
        dist = np.linalg.norm(weights[locs] - weights[i])
        distances[i] = np.mean(dist)

    return distances


def topographic_error(data: np.ndarray, weights: np.ndarray, neighbor_radius: float = 1):
    """
    Measure the topographic error of a SOM.
    E_t = mean(err(x_i)), where err is 1 if the two best matching units are not adjacent

    :param data: data samples
    :param weights: SOM weight matrix with shape (..., 1, features)
    :param neighbor_radius: specifies the neighborhood condition
    :return: topographic error
    """
    n_data = data.shape[0]
    map_shape = weights.shape[:-2]

    # Distances from data points to units
    diff = weights - data
    dist = np.sum(diff ** 2, axis=-1, keepdims=True)

    # Indices to best and second best matching units as [bmus, 2nd bmus]
    bmus = np.argsort(dist.reshape(-1, n_data), axis=0)[:2, :]
    b_ix = np.array(
        np.unravel_index(bmus.ravel(), map_shape)
    ).reshape(len(map_shape), -1)

    # Check distances between bmus
    errors = np.array([
        1 if np.linalg.norm(b_ix[:, i] - b_ix[:, i+n_data]) > neighbor_radius
        else 0
        for i in range(n_data)
    ])

    return np.mean(errors)


def embedding_accuracy(data: np.ndarray, weights: np.ndarray, alpha: float = 0.05):
    """
    Map embedding accuracy. Test whether the weights have a similar distribution to data.
    Uses scipy.ks_2samp (two-sided, two-sample test) to determine similarity.

    :param data: samples
    :param weights: SOM weight matrix with shape (..., 1, features)
    :param alpha: confidence interval
    :return: embedding accuracy
    """
    features = data.shape[-1]
    w = weights.reshape(-1, features)

    pvals = np.array([ks_2samp(data[:, f], w[:, f])[1] for f in range(features)])
    return np.mean(pvals > alpha)
