from sleep import clock, usleep

import numba as nb

@nb.njit(nogil=True)
def numba_clock():
    c1 = clock()
    print("c1", c1)
    usleep(1000000)
    c2 = clock()
    print("c2", c2)
    print("time", c2-c1)

if __name__ == "__main__":
    numba_clock()