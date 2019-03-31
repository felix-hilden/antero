import numpy as np
import seaborn as sns

from matplotlib import pyplot as plt

from antero.som.measures import umatrix as _umatrix


def _gather_indices(indices: np.ndarray, shape: tuple):
    """
    Count occurrence of indices.

    :param indices: indices to array
    :param shape: shape of original array
    :return: array with index counts
    """
    heat = np.zeros(shape)
    np.add.at(heat, tuple(indices), 1)
    return heat


def _gather_indices_with_labels(indices: np.ndarray, labels: np.ndarray, shape: tuple):
    """
    Count occurrence of indices per label.

    :param indices: indices to array
    :param labels: true labels
    :param shape: shape of original array
    :return: array with index counts
    """
    heats = np.zeros((labels.max() + 1,) + shape)
    for i in range(labels.max() + 1):
        heats[i] = _gather_indices(indices[..., np.where(labels == i)], shape)
    return heats


def heatmap(indices: np.ndarray, shape: tuple, labels: np.ndarray = None):
    """
    Produce heatmaps indicating where samples land on a map.

    :param indices: SOM projections
    :param shape: shape of the map
    :param labels: optional, heatmaps are produced for every label separately
    :return:
    """
    if labels is None:
        plt.figure()
        sns.heatmap(_gather_indices(indices, shape), vmin=0, cmap='magma')
    else:
        heats = _gather_indices_with_labels(indices, labels, shape)
        for i in range(heats.shape[0]):
            plt.figure()
            sns.heatmap(heats[i], vmin=0, cmap='magma')
            plt.pause(0.1)


def umatrix(weights: np.ndarray, d: float = 1):
    """
    Plot U-matrix.

    :param weights: SOM weights
    :param d: size of neighbourhood
    :return: None
    """
    plt.figure()
    plt.imshow(_umatrix(weights, d), cmap='binary')
