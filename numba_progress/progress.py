import numpy as np
import sys
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook
from threading import Thread, Event
from .atomic import atomic_add

__all__ = ['ProgressBar']


def is_notebook():
    """Determine if we're running within an IPython kernel

    >>> is_notebook()
    False
    """
    # http://stackoverflow.com/questions/34091701/determine-if-were-in-an-ipython-notebook-session
    if "IPython" not in sys.modules:  # IPython hasn't been imported
        return False
    from IPython import get_ipython

    # check for `kernel` attribute on the IPython instance
    return getattr(get_ipython(), "kernel", None) is not None


class ProgressBar(object):
    """
    Wraps the tqdm progress bar enabling it to be updated from within a numba nopython function.
    It works by spawning a separate thread that updates the tqdm progress bar based on an atomic counter which can be
    accessed within the numba function. The progress bar works with parallel as well as sequential numba functions.
    
    Note: As this Class contains python objects not useable or convertable into numba, it will be boxed as a
    proxy object, that only exposes the minimum subset of functionality to update the progress bar. Attempts
    to return or create a ProgressBar within a numba function will result in an error.

    Parameters
    ----------
    file: `io.TextIOWrapper` or `io.StringIO`, optional
        Specifies where to output the progress messages
        (default: sys.stdout). Uses `file.write(str)` and `file.flush()`
        methods.  For encoding, see `write_bytes`.
    update_interval: float, optional
        The interval in seconds used by the internal thread to check for updates [default: 0.1].
    notebook: bool, optional
        If set, forces or forbits the use of the notebook progress bar. By default the best progress bar will be
        determined automatically.
    dynamic_ncols: bool, optional
        If true, the number of columns (the width of the progress bar) is constantly adjusted. This improves the
        output of the notebook progress bar a lot.
    kwargs: dict-like, optional
        Addtional parameters passed to the tqdm class. See https://github.com/tqdm/tqdm for a documentation of
        the available parameters. Noteable exceptions are the parameters:
            - file is redefined above (see above)
            - iterable is not available because it would not make sense here
            - dynamic_ncols is defined above
    """
    def __init__(self, file=None, update_interval=0.1, notebook=None, dynamic_ncols=True, **kwargs):
        if file is None:
            file = sys.stdout
        self._last_value = 0

        if notebook is None:
            notebook = is_notebook()

        if notebook:
            self._tqdm = tqdm_notebook(iterable=None, dynamic_ncols=dynamic_ncols, file=file, **kwargs)
        else:
            self._tqdm = tqdm(iterable=None, dynamic_ncols=dynamic_ncols, file=file, **kwargs)

        self.hook = np.zeros(1, dtype=np.uint64)
        self._updater_thread = None
        self._exit_event = Event()
        self.update_interval = update_interval
        self._start()

    def _start(self):
        self._timer = Thread(target=self._update_function)
        self._timer.start()

    def close(self):
        self._exit_event.set()
        self._timer.join()
        self._update_tqdm()  # update to set the progressbar to it's final value in case the thread missed a loop
        self._tqdm.refresh()
        self._tqdm.close()

    @property
    def value(self):
        return self.hook[0]

    def update(self, n=1):
        atomic_add(self.hook, 0, n)
        self._update_tqdm()

    def _update_tqdm(self):
        value = self.value
        diff = value - self._last_value
        self._last_value = value
        self._tqdm.update(diff)

    def _update_function(self):
        """Background thread for updating the progress bar.
        """
        while not self._exit_event.is_set():
            self._update_tqdm()
            self._exit_event.wait(self.update_interval)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

