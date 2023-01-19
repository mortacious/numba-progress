from .progress_base import ProgressBarBase
from functools import wraps
import numpy as np
try:
    from numba import cuda
    from numba.cuda import managed_array
    _has_managed = True
except ImportError:
    _has_managed = False
import platform
import warnings
from numba import types
from numba.core import cgutils
from numba.core.extending import (
    make_attribute_wrapper,
    models,
    register_model,
    type_callable,
    typeof_impl,
    lower_builtin
)
from numba.core.typing.templates import AttributeTemplate
from numba.cuda.cudadecl import registry as cuda_registry
from numba.cuda.cudaimpl import lower_attr as cuda_lower_attr, lower


class ProgressBarCuda(ProgressBarBase):
    @wraps(ProgressBarBase.__init__)
    def __init__(self, *args, **kwargs):
        if not _has_managed:
            raise TypeError("numba.cuda implementation has to support managed_array.")
        if platform.system() == "Windows":
            warnings.warn("The managed_array is considered experimental on Windows systems. Handle with care.")
        super().__init__(*args, **kwargs)

    def _init_counter(self):
        self._counter = cuda.managed_array(1, dtype=np.uint64)
        self._counter[0] = 0

    @property
    def counter(self):
        return self._counter

    def _update_counter(self, n):
        raise NotImplementedError("Updating the counter from host code is not supported")

    @property
    def proxy(self):
        return cuda.as_cuda_array(self._counter)

    # return this as an array passing custum types into cuda kernels is surprisingly difficult :)
    @property
    def __cuda_array_interface__(self):
        return self._counter.__cuda_array_interface__


# Numba.cuda Native Implementation for the ProgressBarCuda Class
class ProgressBarCudaType(types.Type):
    def __init__(self):
        super().__init__(name='ProgressBarCuda')


progressbar_type = ProgressBarCudaType()


@register_model(ProgressBarCudaType)
class ProgressBarCudaModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ("counter", types.Array(types.uint64, 1, 'C')),
        ]
        super().__init__(dmm, fe_type, members)


make_attribute_wrapper(ProgressBarCudaType, "counter", "counter")


@typeof_impl.register(ProgressBarCuda)
def typeof_progress_bar_cuda(val, c):
    return progressbar_type


@type_callable(ProgressBarCuda)
def type_progress_bar_cuda(context):
    def typer(array):
        if isinstance(array, types.Array) and array.dtype == types.uint64 and array.ndim == 1:
            return progressbar_type
    return typer


# constructor from an array
@lower_builtin(ProgressBarCuda, types.Array)
def impl_progressbar(context, builder, sig, args):
    typ = sig.return_type
    array = args[0]
    progressbar = cgutils.create_struct_proxy(typ)(context, builder)
    progressbar.counter = array
    return progressbar._getvalue()


@cuda.jit(device=True)
def update_progress(progress_proxy, n=1):
    progress_bar = ProgressBarCuda(progress_proxy)
    progress_bar.counter[0] += n

# @lower(update_progress, progressbar_type, types.uint64)
# def ptx_progressbar_update(context, builder, sig, args):
#     progress_typ = sig.args[0]
#     progress_val = args[0]
#
#     val = args[1]
#
#     ctor = cgutils.create_struct_proxy(progress_typ)
#     progressbar = ctor(context, builder, value=progress_val)
#
#     counter_array = progressbar.array
#     counter_array_typ = types.Array(types.uint64, 1, 'C')
#     idx = context.get_constant(types.int64, 0)
#     ptr = cgutils.get_item_pointer(context, builder, counter_array_typ, counter_array, idx,
#                                    wraparound=True)
#     return builder.atomic_rmw('add', ptr, val, 'monotonic')
