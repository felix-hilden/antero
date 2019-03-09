from contextlib import contextmanager
from time import perf_counter


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
