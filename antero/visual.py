import numpy as np
import matplotlib

from matplotlib import pyplot as plt


def heatmap(data, row_labels, col_labels, ax=None, cbar_kw=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Adapted from Matplotlib documentation:
    https://matplotlib.org/gallery/images_contours_and_fields/image_annotated_heatmap.html

    :param data: A 2D numpy array of shape (N,M)
    :param row_labels: A list or array of length N with the labels for the rows
    :param col_labels: A list or array of length M with the labels for the columns
    :param ax: A matplotlib.axes.Axes instance to which the heatmap is plotted.
        If not provided, use current axes or create a new one.
    :param cbar_kw: A dictionary with arguments to matplotlib.Figure.colorbar.
    :param cbarlabel: The label for the colorbar.

    Other arguments are directly passed on to the imshow call.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar_kw = cbar_kw if cbar_kw is not None else {}
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}", textcolors=("black", "white"), threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Adapted from Matplotlib documentation:
    https://matplotlib.org/gallery/images_contours_and_fields/image_annotated_heatmap.html

    :param im: The AxesImage to be labeled.
    :param data: Data used to annotate. If None, the image's data is used.
    :param valfmt: The format of the annotations inside the heatmap.
        This should either use the string format method, e.g.
        "$ {x:.2f}", or be a :class:`matplotlib.ticker.Formatter`.
    :param textcolors: A list or array of two color specifications. The first is
        used for values below a threshold, the second for those above.
    :param threshold: Value in data units according to which the colors from textcolors
        are applied. If None (the default) uses the middle of the colormap as separation.

    Other arguments are passed on to the created text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[im.norm(data[i, j]) > threshold])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts
