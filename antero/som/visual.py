import numpy as np
import seaborn as sns

from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

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
        plt.title('Heatmap')
        sns.heatmap(_gather_indices(indices, shape), vmin=0, cmap='magma')
    else:
        heats = _gather_indices_with_labels(indices, labels, shape)
        for i in range(heats.shape[0]):
            plt.figure()
            plt.title('Heatmap %d' % i)
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
    plt.title('U-matrix')
    plt.imshow(_umatrix(weights, d), cmap='binary')


def class_pies(indices: np.ndarray, shape: tuple, labels: np.ndarray):
    """
    Plot self-organising map as a set of pie charts in terms of labels at each node.

    :param indices: SOM projections
    :param shape: shape of the map
    :param labels: true class labels
    :return: None
    """
    heats = _gather_indices_with_labels(indices, labels, shape)

    plt.figure(figsize=(7, 7))
    plt.suptitle('Class pies')
    grid = GridSpec(*shape)
    for y in range(shape[0]):
        for x in range(shape[1]):
            plt.subplot(grid[y, x])
            p, _ = plt.pie(heats[:, y, x], radius=1.4)
    plt.legend(p, np.arange(0, labels.max()+1), ncol=1)
