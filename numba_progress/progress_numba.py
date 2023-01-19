from .progress_base import ProgressBarBase
import numpy as np
import sys
from numba.extending import overload_method, typeof_impl, as_numba_type, models, register_model, \
    make_attribute_wrapper, overload_attribute, unbox, NativeValue, box
from .numba_atomic import atomic_add
from numba import types
from numba.core import cgutils
from numba.core.boxing import unbox_array

__all__ = ['ProgressBarNumba']


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


class ProgressBarNumba(ProgressBarBase):
    def _init_counter(self):
        self._counter = np.zeros(1, dtype=np.uint64)

    @property
    def counter(self):
        return self._counter

    def _update_counter(self, n):
        atomic_add(self._counter, 0, n)

#####################################################################################
# Lowering implementation to make the progress bar usable from within numba functions
#####################################################################################


# Numba Native Implementation for the ProgressBarNumba Class
class ProgressBarNumbaType(types.Type):
    def __init__(self):
        super().__init__(name='ProgressBarNumba')


progressbar_type = ProgressBarNumbaType()


@typeof_impl.register(ProgressBarNumba)
def typeof_index(val, c):
    return progressbar_type


as_numba_type.register(ProgressBarNumba, progressbar_type)


@register_model(ProgressBarNumbaType)
class ProgressBarNumbaModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ('counter', types.Array(types.uint64, 1, 'C')),
        ]
        models.StructModel.__init__(self, dmm, fe_type, members)


# make the counter attribute accessible
make_attribute_wrapper(ProgressBarNumbaType, 'counter', 'counter')


@overload_attribute(ProgressBarNumbaType, 'count')
def get_value(progress_bar):
    def getter(progress_bar):
        return progress_bar.counter[0]

    return getter


@unbox(ProgressBarNumbaType)
def unbox_progressbar(typ, obj, c):
    """
    Convert a ProgressBar to it's native representation (proxy object)
    """
    counter_obj = c.pyapi.object_getattr_string(obj, 'counter')
    progress_bar = cgutils.create_struct_proxy(typ)(c.context, c.builder)
    progress_bar.counter = unbox_array(types.Array(types.uint64, 1, 'C'), counter_obj, c).value
    c.pyapi.decref(counter_obj)
    is_error = cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return NativeValue(progress_bar._getvalue(), is_error=is_error)


@box(ProgressBarNumbaType)
def box_progressbar(typ, val, c):
    raise TypeError("Native representation of ProgressBarNumba cannot be converted back to a python object "
                    "as the python version contains internal python state.")


@overload_method(ProgressBarNumbaType, "update", jit_options={"nogil": True})
def _ol_update(self, n=1):
    """
    Numba nopython implementation of the update method.
    """
    if isinstance(self, ProgressBarNumbaType):
        def _update_impl(self, n=1):
            atomic_add(self.counter, 0, n)

        return _update_impl


