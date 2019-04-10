import numpy as np
import seaborn as sns

from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

from antero.som import _BaseSOM
from antero.som.measures import umatrix as _umatrix


def _gather_indices(indices: np.ndarray, shape: tuple) -> np.ndarray:
    """
    Count occurrence of indices.

    :param indices: indices to array
    :param shape: shape of original array
    :return: array with index counts
    """
    heat = np.zeros(shape)
    np.add.at(heat, tuple(indices), 1)
    return heat


def _gather_indices_with_labels(indices: np.ndarray, labels: np.ndarray, shape: tuple) -> np.ndarray:
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


def heatmap(som: _BaseSOM, x: np.ndarray, y: np.ndarray = None) -> None:
    """
    Produce heatmaps indicating where samples land on a map.

    :param som: self-organising map instance
    :param x: data samples
    :param y: optional, heatmaps are produced for every label separately
    :return: None
    """
    indices = som.project(x)

    if y is None:
        plt.figure()
        plt.title('Heatmap')
        sns.heatmap(_gather_indices(indices, som.shape), vmin=0, cmap='magma')
    else:
        heats = _gather_indices_with_labels(indices, y, som.shape)
        for i in range(heats.shape[0]):
            plt.figure()
            plt.title('Heatmap %d' % i)
            sns.heatmap(heats[i], vmin=0, cmap='magma')
            plt.pause(0.1)


def umatrix(som: _BaseSOM, d: float = 1) -> None:
    """
    Plot U-matrix.

    :param som: self-organising map instance
    :param d: size of neighbourhood
    :return: None
    """
    plt.figure()
    plt.title('U-matrix')
    plt.imshow(_umatrix(som, d), cmap='binary')


def class_pies(som: _BaseSOM, x: np.ndarray, y: np.ndarray) -> None:
    """
    Plot self-organising map as a set of pie charts in terms of labels at each node.
    Very inefficient for large maps as it produces a subplot for each node.

    :param som: self-organising map instance
    :param x: data samples
    :param y: true class labels
    :return: None
    """
    indices = som.project(x)
    heats = _gather_indices_with_labels(indices, y, som.shape)

    plt.figure(figsize=(7, 7))
    plt.suptitle('Class pies')
    grid = GridSpec(*som.shape)
    for iy in range(som.shape[0]):
        for ix in range(som.shape[1]):
            plt.subplot(grid[iy, ix])
            p, _ = plt.pie(heats[:, iy, ix], radius=1.4)
    plt.legend(p, np.arange(0, y.max()+1), ncol=1)
