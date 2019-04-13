import numpy as np
import seaborn as sns
import matplotlib

from matplotlib import pyplot as plt
from antero.exceptions import ProgrammingError


def cat_heatmap(x: np.ndarray, cat, cmap: str = 'tab20', **kwargs) -> None:
    """
    Produce a label map assigning a class to each node based on the most frequent class.

    :param x: heatmap to draw
    :param cat: category names
    :param cmap: color map name
    :param kwargs: additional arguments to sns.heatmap
    :return: None
    """
    n_cat = len(cat)
    bounds = np.linspace(-0.5, n_cat-0.5, n_cat+1)
    norm = matplotlib.colors.BoundaryNorm(bounds, n_cat)
    fmt = matplotlib.ticker.FuncFormatter(
        lambda z, pos: cat[norm(z)]
    )

    for kw in ['cmap', 'vmax', 'cbar_kws']:
        if kw in kwargs:
            raise ProgrammingError('Keyword %s already provided by cat_heatmap!' % kw)

    sns.heatmap(
        x, cmap=plt.get_cmap(cmap, n_cat), vmax=n_cat, **kwargs,
        cbar_kws=dict(
            ticks=np.arange(n_cat), format=fmt,
            boundaries=bounds, drawedges=True
        ),
    )
