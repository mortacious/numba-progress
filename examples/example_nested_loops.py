from sleep import usleep
import numba as nb
from numba_progress import ProgressBar

@nb.njit(nogil=True)
def numba_sleeper(num_iterations, sleep_us, progress):
    for i in range(num_iterations):
        progress[0].update()
        for j in range(num_iterations):
            usleep(sleep_us)
            progress[1].update(1)
        # reset the second progress bar to 0
        progress[1].set(0)


if __name__ == "__main__":
    num_iterations = 30
    sleep_time_us = 25_000
    with ProgressBar(total=num_iterations, ncols=80) as numba_progress1, ProgressBar(total=num_iterations, ncols=80) as numba_progress2:
        # note: progressbar object must be passed as a tuple (a list will not work due to different treatment in numba)
        numba_sleeper(num_iterations, sleep_time_us, (numba_progress1, numba_progress2))