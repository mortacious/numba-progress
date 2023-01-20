from .progress import ProgressBar
from numba import types
from numba.extending import typeof_impl, as_numba_type

__all__ = ["progressbar"]


# Numba Native Implementation for the ProgressBar Class
class ProgressBarType(types.Type):
    def __init__(self):
        super().__init__(name='ProgressBar')


progressbar = ProgressBarType()


@typeof_impl.register(ProgressBar)
def typeof_index(val, c):
    return progressbar


as_numba_type.register(ProgressBar, progressbar)