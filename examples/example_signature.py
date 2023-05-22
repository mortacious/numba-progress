# example code to use the progressbar with an explicit signature

from sleep import usleep
import numba as nb
from numba_progress import ProgressBar, ProgressBarType


@nb.njit(nb.void(nb.uint64, nb.uint64, ProgressBarType), nogil=True)
def numba_sleeper(num_iterations, sleep_us, progress_hook):
    for i in range(num_iterations):
        usleep(sleep_us)
        progress_hook.update(1)


if __name__ == "__main__":
    num_iterations = 30
    sleep_time_us = 250_000
    with ProgressBar(total=num_iterations, ncols=80) as numba_progress:
        numba_sleeper(num_iterations, sleep_time_us, numba_progress)