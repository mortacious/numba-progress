from sleep import usleep
import numba as nb
from numba_progress import ProgressBar


@nb.njit(nogil=True, parallel=True)
def numba_parallel_sleeper(num_iterations, sleep_us, progress_hook=None):
    for i in nb.prange(num_iterations):
        usleep(sleep_us)
        if progress_hook is not None:
            progress_hook.update(1)


if __name__ == "__main__":
    num_iterations = 30*8
    sleep_time_us = 250_000
    with ProgressBar(total=num_iterations, dynamic_ncols=True) as numba_progress:
        numba_parallel_sleeper(num_iterations, sleep_time_us, numba_progress)

