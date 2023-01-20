from numba.extending import overload_method, typeof_impl, as_numba_type, models, register_model, \
    make_attribute_wrapper, overload_attribute, unbox, NativeValue, box
from numba import types
from numba.core import cgutils
from numba.core.boxing import unbox_array
from .atomic import atomic_add
from .types import ProgressBarType


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
