import numba as nb
import numpy as np
import sys
from tqdm import tqdm
from threading import Thread, Event

from numba.extending import overload_method, typeof_impl, as_numba_type, models, register_model, \
    make_attribute_wrapper, overload_attribute, unbox, NativeValue, box
from .numba_atomic import atomic_add
from numba import types
from numba.core import cgutils
from numba.core.boxing import unbox_array

__all__ = ['ProgressBar']


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


# Numba Native Implementation for the ProgressBar Class

class ProgressBarType(types.Type):
    def __init__(self):
        super().__init__(name='ProgressBar')


progressbar_type = ProgressBarType()


@typeof_impl.register(ProgressBar)
def typeof_index(val, c):
    return progressbar_type


as_numba_type.register(ProgressBar, progressbar_type)


@register_model(ProgressBarType)
class ProgressBarModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ('hook', types.Array(types.uint64, 1, 'C')),
        ]
        models.StructModel.__init__(self, dmm, fe_type, members)


# make the hook attribute accessible
make_attribute_wrapper(ProgressBarType, 'hook', 'hook')


@overload_attribute(ProgressBarType, 'value')
def get_value(progress_bar):
    def getter(progress_bar):
        return progress_bar.hook[0]
    return getter


@unbox(ProgressBarType)
def unbox_progressbar(typ, obj, c):
    """
    Convert a ProgressBar to it's native representation (proxy object)
    """
    hook_obj = c.pyapi.object_getattr_string(obj, 'hook')
    progress_bar = cgutils.create_struct_proxy(typ)(c.context, c.builder)
    progress_bar.hook = unbox_array(types.Array(types.uint64, 1, 'C'), hook_obj, c).value
    c.pyapi.decref(hook_obj)
    is_error = cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return NativeValue(progress_bar._getvalue(), is_error=is_error)


@box(ProgressBarType)
def box_progressbar(typ, val, c):
    raise TypeError("Native representation of ProgressBar cannot be converted back to a python object "
                    "as it contains internal python state.")


@overload_method(ProgressBarType, "update", jit_options={"nogil": True})
def _ol_update(self, n=1):
    """
    Numpy implementation of the update method.
    """
    if isinstance(self, ProgressBarType):
        def _update_impl(self, n=1):
            atomic_add(self.hook, 0, n)
        return _update_impl


