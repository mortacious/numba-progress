from numba import cuda
from numba_progress.progress_numba_cuda import ProgressBarCuda, update_progress
import cupy as cp

NUM_ITERATIONS = 10000
SLEEP_DUR = 10000000


@cuda.jit
def report_progress(progress_proxy):
    for i in range(NUM_ITERATIONS):
        update_progress(progress_proxy, 1)
        cuda.nanosleep(SLEEP_DUR)


if __name__ == "__main__":
    with ProgressBarCuda(total=NUM_ITERATIONS, dynamic_ncols=True) as cuda_progress:
        # Kernel launch, which runs asynchronously with respect to the host
        report_progress[1, 1](cuda_progress.proxy)
        cuda.synchronize()

