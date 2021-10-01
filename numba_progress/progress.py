import numba as nb
import numpy as np
import sys
from tqdm import tqdm
from threading import Thread, Event

from numba.core import types
from numba.experimental import structref
from numba.extending import overload_method
from .numba_atomic import atomic_add


__all__ = ['ProgressBar']


@structref.register
class ProgressProxyType(types.StructRef):
    def preprocess_fields(self, fields):
        # We don't want the struct to take Literal types.
        return tuple((name, types.unliteral(typ)) for name, typ in fields)


class _ProgressProxy(structref.StructRefProxy):
    def __new__(cls, hook=None):
        hook = np.zeros(1, dtype=np.uint64)
        return structref.StructRefProxy.__new__(cls, hook)

    def update(self, n=1):
        return _ProgressProxy_update(self, n)

    @property
    def value(self):
        return _ProgressProxy_value(self)


@nb.njit()
def _ProgressProxy_update(self, n=1):
    return self.update(n)


@nb.njit()
def _ProgressProxy_value(self):
    return self.hook[0]


structref.define_proxy(_ProgressProxy, ProgressProxyType,
                       ["hook"])


@overload_method(ProgressProxyType, "update", jit_options={"nogil": True})
def _ol_update(self, n=1):
    def _update_impl(self, n=1):
        atomic_add(self.hook, 0, n)
    return _update_impl


class ProgressBar(object):
    """
    Wraps the tqdm progress bar enabling it to be updated from within a numba nopython function.
    It works by spawning a separate thread that updates the tqdm progress bar based on an atomic counter which can be
    accessed within the numba function. The progress bar works with parallel as well as sequential numba functions.

    Parameters
    ----------
    file: `io.TextIOWrapper` or `io.StringIO`, optional
        Specifies where to output the progress messages
        (default: sys.stdout). Uses `file.write(str)` and `file.flush()`
        methods.  For encoding, see `write_bytes`.
    update_interval: float, optional
        The interval in seconds used by the internal thread to check for updates [default: 0.1].
    kwargs: dict-like, optional
        Addtional parameters passed to the tqdm class. See https://github.com/tqdm/tqdm for a documentation of
        the available parameters. Noteable exceptions are the parameters:
            - file is redefined above (see above)
            - iterable is not available because it would not make sense here
    """
    def __init__(self, file=None, update_interval=0.1, **kwargs):
        if file is None:
            file = sys.stdout
        self._last_value = 0
        self._tqdm = tqdm(iterable=None, file=file, **kwargs)
        self._numba_proxy = _ProgressProxy()
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
    def numba_proxy(self):
        """
        Returns the proxy object that can be used from within a numba function.
        """
        return self._numba_proxy

    def update(self, n=1):
        self._numba_proxy.update(n)
        self._update_tqdm()

    def _update_tqdm(self):
        value = self._numba_proxy.value
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
        return self._numba_proxy

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
