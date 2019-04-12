import numpy as np


def outliers(d: np.ndarray, change: float = 0.01) -> np.ndarray:
    """
    Detect outliers iteratively based on reduction in standard deviation.
    Remove data from distribution ends iteratively.

    :param d: data
    :param change: relative change in standard deviation for stopping iteration
    :return: boolean array indicating outliers
    """
    x = d.copy()
    x.sort()

    n = 0
    m = x.size

    while x[n:m].size > 1:
        std = x[n:m].std()
        std_l = x[n+1:m].std()
        std_r = x[n:m-1].std()

        std_n = std_l if std_l < std_r else std_r
        if std / std_n < 1 + change:
            break

        if std_l < std_r:
            n += 1
        else:
            m -= 1

    bound_l = x[n]
    bound_r = x[m-1]

    return np.logical_or(d < bound_l, d > bound_r)
