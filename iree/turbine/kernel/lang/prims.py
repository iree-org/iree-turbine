from .. import ops

from .._support.tracing import (
    BaseContext,
    CompiledContext,
    custom_primitive_fn,
    eager_context,
)

__all__ = [
    "is_debug",
    "program_id",
    "constant",
    "exp2",
    "max",
    "sum",
    "dot",
    "for_loop",
    "load",
    "store",
    "broadcast",
    "broadcast_in_dim",
    "transpose",
    "to_dtype",
]


def is_debug() -> bool:
    """Returns whether we are currently executing a kernel in eager debug mode."""
    return BaseContext.current().eager


# Core language operations
program_id = ops.thread_program_id
to_dtype = ops.to_dtype

# Math Operations
exp2 = ops.exp2
constant = ops.vector_constant

# Reduction Operations
max = ops.vector_max
sum = ops.vector_sum
dot = ops.vector_dot

# Control Flow Operations
for_loop = ops.for_loop

# Memory Operations
load = ops.kernel_buffer_load
store = ops.kernel_buffer_store

# Shape Manipulation operations
broadcast = ops.vector_broadcast
broadcast_in_dim = ops.vector_broadcast_in_dim
transpose = ops.vector_transpose
