import numba as nb


@nb.generated_jit(nogil=True, nopython=True)
def usleep(usec):
    # c usleep function in numba
    import ctypes
    libc = ctypes.CDLL('libc.so.6')
    libc.usleep.argtypes = (ctypes.c_uint,)
    func_usleep = libc.usleep

    def usleep_impl(usec):
        func_usleep(usec)

    return usleep_impl