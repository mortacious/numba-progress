# source from https://github.com/KatanaGraph/katana/blob/master/python/katana/numba_support/numpy_atomic.py

# The 3-Clause BSD License
#
# Copyright 2018 The University of Texas at Austin
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


from numba import types
from numba.core import cgutils
from numba.core.typing.arraydecl import get_array_index_type
from numba.extending import lower_builtin, type_callable
from numba.np.arrayobj import basic_indexing, make_array, normalize_indices

__all__ = ["atomic_add", "atomic_sub", "atomic_max", "atomic_min"]


def atomic_rmw(context, builder, op, arrayty, val, ptr):
    assert arrayty.aligned  # We probably have to have aligned arrays.
    dataval = context.get_value_as_data(builder, arrayty.dtype, val)
    return builder.atomic_rmw(op, ptr, dataval, "monotonic")


def declare_atomic_array_op(iop, uop, fop):
    def decorator(func):
        @type_callable(func)
        def func_type(context):
            def typer(ary, idx, val):
                out = get_array_index_type(ary, idx)
                if out is not None:
                    res = out.result
                    if context.can_convert(val, res):
                        return res
                return None

            return typer

        _ = func_type

        @lower_builtin(func, types.Buffer, types.Any, types.Any)
        def func_impl(context, builder, sig, args):
            """
            array[a] = scalar_or_array
            array[a,..,b] = scalar_or_array
            """
            aryty, idxty, valty = sig.args
            ary, idx, val = args

            if isinstance(idxty, types.BaseTuple):
                index_types = idxty.types
                indices = cgutils.unpack_tuple(builder, idx, count=len(idxty))
            else:
                index_types = (idxty,)
                indices = (idx,)

            ary = make_array(aryty)(context, builder, ary)

            # First try basic indexing to see if a single array location is denoted.
            index_types, indices = normalize_indices(context, builder, index_types, indices)
            dataptr, shapes, _strides = basic_indexing(
                context, builder, aryty, ary, index_types, indices, boundscheck=context.enable_boundscheck,
            )
            if shapes:
                raise NotImplementedError("Complex shapes are not supported")

            # Store source value the given location
            val = context.cast(builder, val, valty, aryty.dtype)
            op = None
            if isinstance(aryty.dtype, types.Integer) and aryty.dtype.signed:
                op = iop
            elif isinstance(aryty.dtype, types.Integer) and not aryty.dtype.signed:
                op = uop
            elif isinstance(aryty.dtype, types.Float):
                op = fop
            if op is None:
                raise TypeError("Atomic operation not supported on " + str(aryty))
            return atomic_rmw(context, builder, op, aryty, val, dataptr)

        _ = func_impl

        return func

    return decorator


@declare_atomic_array_op("add", "add", "fadd")
def atomic_add(ary, i, v):
    """
    Atomically, perform `ary[i] += v` and return the previous value of `ary[i]`.

    i must be a simple index for a single element of ary. Broadcasting and vector operations are not supported.

    This should be used from numba compiled code.
    """
    orig = ary[i]
    ary[i] += v
    return orig


@declare_atomic_array_op("sub", "sub", "fsub")
def atomic_sub(ary, i, v):
    """
    Atomically, perform `ary[i] -= v` and return the previous value of `ary[i]`.

    i must be a simple index for a single element of ary. Broadcasting and vector operations are not supported.

    This should be used from numba compiled code.
    """
    orig = ary[i]
    ary[i] -= v
    return orig


@declare_atomic_array_op("max", "umax", None)
def atomic_max(ary, i, v):
    """
    Atomically, perform `ary[i] = max(ary[i], v)` and return the previous value of `ary[i]`.
    This operation does not support floating-point values.

    i must be a simple index for a single element of ary. Broadcasting and vector operations are not supported.

    This should be used from numba compiled code.
    """
    orig = ary[i]
    ary[i] = max(ary[i], v)
    return orig


@declare_atomic_array_op("min", "umin", None)
def atomic_min(ary, i, v):
    """
    Atomically, perform `ary[i] = min(ary[i], v)` and return the previous value of `ary[i]`.
    This operation does not support floating-point values.

    i must be a simple index for a single element of ary. Broadcasting and vector operations are not supported.

    This should be used from numba compiled code.
    """
    orig = ary[i]
    ary[i] = min(ary[i], v)
    return orig