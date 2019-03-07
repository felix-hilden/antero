from contextlib import contextmanager
from time import perf_counter


@contextmanager
def timer(msg: str = 'Timer: %.3f'):
    t = perf_counter()
    yield t, perf_counter
    print(msg % (perf_counter() - t))
