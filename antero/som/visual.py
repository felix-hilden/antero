import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.cluster.hierarchy import linkage, dendrogram as _dendrogram

from antero.som import _BaseSOM
from antero.som.measures import umatrix as _umatrix
from antero.visual import heatmap as _heatmap


def heatmap(som: _BaseSOM, x: np.ndarray, y=None) -> None:
    """
    Produce heatmaps indicating where samples land on a map.

    :param som: self-organising map instance
    :param x: data samples
    :param y: labels, either pd.Series of pd.Categorical or np.ndarray of numerical labels
    :return: None
    """
    if y is None:
        heat = som.heatmap(x)
        plt.figure()
        plt.title('Heatmap (%d)' % heat.sum())
        sns.heatmap(heat, vmin=0, cmap='magma')
    else:
        if isinstance(y.dtype, pd.CategoricalDtype):
            title = 'Heatmap, ' + y.name + ': %s (%d)'
            names = y.cat.categories.values
            y = y.cat.codes.values
        else:
            title = 'Heatmap: %s n=%d'
            names = list(range(y.max() + 1))

        heats = som.heatmap(x, y)
        for i, name in enumerate(names):
            plt.figure()
            plt.title(title % (name, heats[i].sum()))
            sns.heatmap(heats[i], vmin=0, cmap='magma')
            plt.pause(0.1)


def labelmap(som: _BaseSOM, x: np.ndarray, y, ordinal: bool = False) -> None:
    """
    Produce a label map assigning a class to each node based on the most frequent class.

    :param som: self-organising map instance
    :param x: data samples
    :param y: labels, either pd.Series of pd.Categorical or np.ndarray of numerical labels
    :param ordinal: indicate ordinal data to choose a continuous color map
    :return: None
    """
    if isinstance(y.dtype, pd.CategoricalDtype):
        title = 'Label map: ' + y.name
        names = y.cat.categories.values
        y = y.cat.codes.values
        n_labs = y.max() + 1
    else:
        n_labs = y.max() + 1
        names = np.array([str(i) for i in range(n_labs)])
        title = 'Label map'

    heats = som.heatmap(x, y)
    labels = som.labelmap(x, y)

    y_ticks = [str(i) for i in range(labels.shape[0])]
    x_ticks = [str(i) for i in range(labels.shape[1])]

    norm = matplotlib.colors.BoundaryNorm(np.linspace(-0.5, n_labs-0.5, n_labs+1), n_labs)
    fmt = matplotlib.ticker.FuncFormatter(
        lambda z, pos: names[norm(z)] + ' (' + str(int(heats[norm(z)].sum())) + ')'
    )
    cmap = 'tab20' if not ordinal else 'copper'

    plt.figure()
    plt.suptitle(title)
    im, _ = _heatmap(
        labels, y_ticks, x_ticks, cmap=plt.get_cmap(cmap, n_labs), norm=norm,
        cbar_kw=dict(ticks=np.arange(n_labs), format=fmt),
        cbarlabel='Class label'
    )


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


def class_pies(som: _BaseSOM, x: np.ndarray, y) -> None:
    """
    Plot self-organising map as a set of pie charts in terms of labels at each node.
    Very inefficient for large maps as it produces a subplot for each node.

    :param som: self-organising map instance
    :param x: data samples
    :param y: labels, either pd.Series of pd.Categorical or np.ndarray of numerical labels
    :return: None
    """
    if isinstance(y.dtype, pd.CategoricalDtype):
        title = 'Class pies: ' + y.name
        names = y.cat.categories.values
        y = y.cat.codes.values
    else:
        names = np.arange(0, y.max()+1).astype(int).astype(str)
        title = 'Class pies'

    heats = som.heatmap(x, y)
    names = np.array([
        name + ' (%d)' % int(heats[i].sum()) for i, name in enumerate(names)
    ])

    plt.figure(figsize=(7, 7))
    plt.suptitle(title)
    grid = GridSpec(*som.shape)
    for iy in range(som.shape[0]):
        for ix in range(som.shape[1]):
            plt.subplot(grid[iy, ix])
            p, _ = plt.pie(heats[:, iy, ix], radius=1.4)
    plt.legend(p, names, ncol=1)


def class_image(som: _BaseSOM, x: np.ndarray, y: np.ndarray) -> None:
    """
    Create an RGB image of maximum of three classes of inputs.
    Scale brightness with amount of samples.

    :param som: self-organising map instance
    :param x: data samples
    :param y: true class labels
    :return: None
    """
    if y.max() > 2:
        raise ValueError('Maximum of three classes accepted!')

    heats = som.heatmap(x, y)
    scale = np.max(np.sum(heats, axis=0))

    # Insert heatmaps into red and blue channels
    image = np.zeros((*som.shape, 3))
    image[..., 0] = heats[0, ...] / scale
    image[..., 2] = heats[1, ...] / scale

    if y.max() == 2:
        image[..., 1] = heats[2, ...] / scale

    plt.figure()
    plt.title('Class image')
    plt.imshow(image)
