class PPError(Exception):
    """
    Base error for pputils.
    """


class ProgrammingError(PPError):
    """
    Error caused by incorrect use or sequence of routines.
    """
