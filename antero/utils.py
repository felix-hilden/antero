from contextlib import contextmanager
from time import perf_counter

import numpy as np


def nthroot(base: np.ndarray, degree) -> np.ndarray:
    """
    Calculate the degreeth root of base.
    Workaround for numpy.power not accepting negative bases with fractional exponents.

    :param base: base
    :param degree: root degree
    :return: power
    """
    return np.sign(base) * np.abs(base) ** (1/degree)


@contextmanager
def timer(start: str = None, end: str = None):
    if start is not None:
        print(start)

    if start is None and end is None:
        end = 'Timer: %.3f'
    elif start is not None and end is None:
        end = start + ': %.3f'

    t = perf_counter()
    yield t, perf_counter
    print(end % (perf_counter() - t))

