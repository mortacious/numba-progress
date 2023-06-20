import numba as nb
from numba.extending import overload

__all__ = ['usleep', 'clock']


def _usleep(usec):
    raise NotImplementedError


@overload(_usleep, nogil=True, nopython=True, inline='always')
def _usleep_impl(usec):
    # c usleep function in numba
    import ctypes
    libc = ctypes.CDLL('libc.so.6')
    libc.usleep.argtypes = (ctypes.c_uint,)
    func_usleep = libc.usleep

    def usleep_impl(usec):
        func_usleep(usec)

    return usleep_impl


@nb.njit(nogil=True)
def usleep(usec):
    _usleep(usec)


def _clock():
    raise NotImplementedError


@overload(_clock, nogil=True, nopython=True, inline='always')
def _clock_impl():
    import ctypes
    libc = ctypes.CDLL('libc.so.6')
    libc.clock.argtypes = ()
    libc.clock.restype = ctypes.c_ulong
    func_clock = libc.clock
    CLOCKS_PER_SEC = 1_000_000

    def clock_impl():
        return func_clock() / CLOCKS_PER_SEC

    return clock_impl


@nb.njit(nogil=True)
def clock():
    _clock()
