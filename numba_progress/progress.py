import numba as nb
import numpy as np
import sys
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook

from threading import Thread, Event

from numba.extending import overload_method, typeof_impl, as_numba_type, models, register_model, \
    make_attribute_wrapper, overload_attribute, unbox, NativeValue, box, lower_getattr, lower_setattr
from .numba_atomic import atomic_add, atomic_xchg
from numba import types
from numba.core import cgutils
from numba.core.boxing import unbox_array


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
    def n(self):
        return self.hook[0]
    
    def set(self, n=0):
        atomic_xchg(self.hook, 0, n)
        self._update_tqdm()

    def update(self, n=1):
        atomic_add(self.hook, 0, n)
        self._update_tqdm()

    def _update_tqdm(self):
        value = self.hook[0]
        #diff = value - self._last_value
        #self._last_value = value
        self._tqdm.n = value
        self._tqdm.refresh()
        #self._tqdm.update(diff)

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

class ProgressBarTypeImpl(types.Type):
    def __init__(self):
        super().__init__(name='ProgressBar')


# This is the numba type representation of the ProgressBar class to be used in signatures
ProgressBarType = ProgressBarTypeImpl()


@typeof_impl.register(ProgressBar)
def typeof_index(val, c):
    return ProgressBarType


as_numba_type.register(ProgressBar, ProgressBarType)


@register_model(ProgressBarTypeImpl)
class ProgressBarModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ('hook', types.Array(types.uint64, 1, 'C')),
        ]
        models.StructModel.__init__(self, dmm, fe_type, members)


# make the hook attribute accessible
make_attribute_wrapper(ProgressBarTypeImpl, 'hook', 'hook')



@overload_attribute(ProgressBarTypeImpl, 'n')
def get_value(progress_bar):
   def getter(progress_bar):
       return progress_bar.hook[0]
   return getter


@unbox(ProgressBarTypeImpl)
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


@box(ProgressBarTypeImpl)
def box_progressbar(typ, val, c):
    raise TypeError("Native representation of ProgressBar cannot be converted back to a python object "
                    "as it contains internal python state.")


@overload_method(ProgressBarTypeImpl, "update", jit_options={"nogil": True})
def _ol_update(self, n=1):
    """
    Numpy implementation of the update method.
    """
    if isinstance(self, ProgressBarTypeImpl):
        def _update_impl(self, n=1):
            atomic_add(self.hook, 0, n)
        return _update_impl
    
@overload_method(ProgressBarTypeImpl, "set", jit_options={"nogil": True})
def _ol_set(self, n=0):
    """
    Numpy implementation of the update method.
    """
    if isinstance(self, ProgressBarTypeImpl):
        def _set_impl(self, n=0):
            atomic_xchg(self.hook, 0, n)
        return _set_impl


