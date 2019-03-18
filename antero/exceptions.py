class AnteroError(Exception):
    """
    Base error for Antero.
    """


class ProgrammingError(AnteroError):
    """
    Error caused by incorrect use or sequence of routines.
    """
