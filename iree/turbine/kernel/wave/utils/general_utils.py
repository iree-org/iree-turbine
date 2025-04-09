# Copyright 2024 The IREE Authors
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import functools
import glob
import iree.turbine.kernel.lang as tkl
import os
import sympy
import torch
from typing import Any, Callable, Optional


from ..._support.indexing import IndexExpr, IndexSequence, IndexSymbol
from ...lang.global_symbols import *
from ...ops.wave_ops import CustomOp, Read, Reduction, Write
from ..assumptions import Assumption
from ..constraints import (
    Constraint,
    DistributionConstraint,
    HardwareConstraint,
    TilingConstraint,
    WorkgroupConstraint,
)
from .symbol_utils import safe_subs, subs_idxc


# TODO: Monkey-patching f16 support, need to fix in iree.


def run_test(func: Callable[[], None]) -> Callable[[], None]:
    """Run a function as part of the test suite."""
    # Print func name before running
    print(f"{func.__name__}")
    func()
    # Print a separator between tests
    print("-----")
    return func


def get_default_scheduling_params() -> dict[IndexSymbol, Any]:
    """Return default scheduling params."""
    # TODO: get values based on get_default_arch()
    return {
        READ_SHARED_DELAY: 1,
        WRITE_SHARED_DELAY: 1,
        READ_GLOBAL_DELAY: 2,
        WRITE_GLOBAL_DELAY: 2,
        MMA_DELAY: 1,
        VALU_DELAY: 1,
        SHUFFLE_DELAY: 1,
        SHARED_MEMORY_UNITS: 4,
        GLOBAL_MEMORY_UNITS: 4,
        MMA_UNITS: 4,
        VALU_UNITS: 2,
        SHUFFLE_UNITS: 2,
    }


def delinearize_index(
    index: IndexExpr, shape: list[int | IndexExpr]
) -> list[IndexExpr]:
    """
    Delinearizes a 1D index into a multi-dimensional index
    based on the shapes provided. The returned array contains
    the multi-dimensional index.

    Assume the index is x and the shape is [5, 4, 3]. In this case,
    this function returns [x % 3, (x // 3) % 4, (x // 12) % 5].

    """
    nd_index = []
    product = 1
    for i, size in enumerate(reversed(shape)):
        if i == 0:
            nd_index.append(index % size)
        else:
            nd_index.append(sympy.floor(index / product) % size)
        product *= size
    return nd_index[::-1]


def get_hardware_vector_size(
    dim: IndexSymbol,
    hardware_constraint: HardwareConstraint,
    mma_indices: dict[IndexSymbol, int],
) -> int:
    """
    Given a hardware constraint, return the vector sizes for the given dimension.
    This could be a hardware specific vector size or a user specified vector size.
    """
    if mma_indices:
        vector_size = hardware_constraint.mma_matrix_shapes[mma_indices[dim]]
    else:
        vector_size = hardware_constraint.vector_shapes[dim]
    return vector_size


def get_hardware_vector_map(constraints: list[Constraint]) -> dict[IndexSymbol, int]:
    """
    Given a list of constraints, looks for hardware constraint and return a map
    containing dim's and their respective vector sizes.
    """
    vector_map = {}
    for c in constraints:
        if isinstance(c, HardwareConstraint):
            vector_map = c.vector_shapes
            break
    return vector_map


def remove_global_indexing(
    index: dict[IndexSymbol, IndexSequence], constraints: list[Constraint]
) -> dict[IndexSymbol, IndexSequence]:
    """
    This function takes the index sequence for a global read and removes all
    workgroup and induction level indexing. This is necessary for writes to shared memory
    that operate on promoted memory.
    """
    tiling_constraints = [c for c in constraints if isinstance(c, TilingConstraint)]
    workgroup_ids = [WORKGROUP_0, WORKGROUP_1, WORKGROUP_2]
    subs = {w: 0 for w in workgroup_ids}

    new_index = {key: safe_subs(index[key], subs) for key in index}
    for key in new_index:
        for constraint in tiling_constraints:
            new_dim = new_index[key]
            if sympy.sympify(new_dim.start).has(constraint.induction_var):
                new_dim = new_dim.subs({constraint.induction_var: 0})
                new_dim.start = new_dim.start - constraint.start
                new_index[key] = new_dim
    return new_index


def is_shared_mem_access(custom: "CustomOp") -> bool:
    return custom.memory_type.address_space == SHARED_ADDRESS_SPACE


def align_index_vars(
    index: dict[IndexSymbol, IndexSequence], constraints: list[Constraint]
) -> dict[IndexSymbol, IndexSequence]:
    """
    This function aligns index vars with Workgroup/Tiling constraints so it never
    need partial reads/writes.
    """
    key_subs = {
        c.dim: (c.work_bound)
        for c in constraints
        if isinstance(c, DistributionConstraint)
        and subs_idxc(c.dim) != subs_idxc(c.work_bound)
    }
    return {safe_subs(key, key_subs): index[key] for key in index}


def find_index_bounds(
    constraints: list[Constraint], index: dict[IndexExpr, IndexExpr]
) -> Optional[list[IndexExpr]]:
    bounds = []
    for constraint in constraints:
        if not isinstance(constraint, DistributionConstraint):
            continue

        dim = constraint.dim
        if dim not in index:
            continue

        work_size = constraint.work_bound
        if subs_idxc(work_size) == subs_idxc(dim):
            continue

        bounds.append(dim)

    if len(bounds) == 0:
        return None

    return bounds


def get_induction_variable(
    reduction: Reduction, constraints: list[Constraint]
) -> IndexSymbol:
    induction_var = None
    for constraint in constraints:
        if (
            isinstance(constraint, TilingConstraint)
            and reduction.axis == constraint.dim
        ):
            induction_var = constraint.induction_var
            break
    else:
        raise ValueError(f"Could not find induction variable for reduction {reduction}")
    return induction_var


def get_tiling_constraint(
    reduction: Reduction, constraints: list[Constraint]
) -> TilingConstraint:
    for constraint in constraints:
        if (
            isinstance(constraint, TilingConstraint)
            and reduction.axis == constraint.dim
        ):
            return constraint
    else:
        raise ValueError(f"Could not find tiling constraint for reduction {reduction}")


def get_hardware_constraint(constraints: list[Constraint]) -> HardwareConstraint:
    for constraint in constraints:
        if isinstance(constraint, HardwareConstraint):
            return constraint
    else:
        raise ValueError(f"Could not find hardware constraint in {constraints}")


def get_workgroup_constraints(
    constraints: list[Constraint],
) -> list[WorkgroupConstraint]:
    return [x for x in constraints if isinstance(x, WorkgroupConstraint)]


def ceildiv(a: int, b: int) -> int:
    return -(a // -b)


def all_equal(input_list: list[Any]) -> bool:
    if len(input_list) == 0:
        return True
    return all(elem == input_list[0] for elem in input_list)


def get_assumptions(constraints: list[Constraint]) -> list[Assumption]:
    assumptions: list[Assumption] = []
    for constraint in constraints:
        if isinstance(constraint, Assumption):
            assumptions.append(constraint)
    return assumptions


def evaluate_with_assumptions(constraints: list[Constraint], expr: IndexExpr) -> bool:
    """
    Evalutes whether the expression is true given the assumptions.
    To do this, we solve the assumptions and target expression and check
    that the result is in the assumptions.
    """
    facts = [subs_idxc(x.expr) for x in get_assumptions(constraints)]
    result = sympy.solve(facts + [expr])
    # Solve returns a false if the inequalities are not consistent.
    if isinstance(result, sympy.logic.boolalg.BooleanAtom):
        return False
    return True if any([result.equals(x) for x in facts]) else None


def _get_start_index(i: IndexSequence | IndexExpr) -> IndexExpr:
    if isinstance(i, IndexSequence):
        i = i.start

    return i


def _get_start_indices(
    src_indices: dict[IndexExpr, IndexSequence | IndexExpr],
) -> list[IndexExpr]:
    start_indices = []
    for dim_indexing in src_indices:
        i = _get_start_index(src_indices[dim_indexing])
        start_indices.append(i)

    return start_indices


def get_fastest_index(indices: dict[IndexExpr, IndexSequence]):
    """
    This function takes in indices of a Node, extract their sizes
    into a list, and then try do an argmax on it. In the case where
    there are multipled max_vals we pick the fastest/most minor one.
    """

    index_sizes = [subs_idxc(i.size) for i in indices.values()]
    # Find the maximum value
    max_size = max(index_sizes)
    # Find the fastest/most minor index of the maximum value.
    return max(i for i, size in enumerate(index_sizes) if size == max_size)


def get_largest_index_and_size(indices: dict[IndexExpr, IndexSequence]):
    """
    This function takes in indices of a Node, extract their sizes
    into a list, and then returns the dimension with the largest size.
    In case of ties, it picks the fastest changing dimension.
    """

    sorted_values = sorted(
        [
            # Call simplify_index to avoid comparing constants with sympy values.
            (i, dim, subs_idxc(index.size))
            # (i, dim, simplify_index(subs_idxc(index.size)))
            for i, (dim, index) in enumerate(indices.items())
        ],
        # x[0] is the index of the dimension.
        # x[2] is the size of the dimension.
        # We want to sort in descending order, first by size, then by index.
        # (pick the largest size with the largest index).
        key=lambda x: (-x[2], -x[0]),
    )
    return sorted_values[0][1:]


def partial(func, *args, **kwargs):
    """functools.partial but with function attributes copied to the partial function."""
    partial_func = functools.partial(func, *args, **kwargs)
    functools.update_wrapper(partial_func, func)
    return partial_func


TORCH_DTYPE_TO_WAVE = {
    torch.bfloat16: tkl.bf16,
    torch.float8_e5m2: tkl.f8e5m2,
    torch.float8_e5m2fnuz: tkl.f8e5m2fnuz,
    torch.float8_e4m3fn: tkl.f8e4m3fn,
    torch.float8_e4m3fnuz: tkl.f8e4m3fnuz,
    torch.float16: tkl.f16,
    torch.float32: tkl.f32,
    torch.float64: tkl.f64,
    torch.int16: tkl.i16,
    torch.int32: tkl.i32,
    torch.int64: tkl.i64,
    torch.bool: tkl.bool,
}

TORCH_DTYPE_RANGE = {
    torch.bfloat16: [-3.3895313892515355e38, 3.3895313892515355e38],
    torch.float8_e5m2: [-57344.0, 57344.0],
    torch.float8_e5m2fnuz: [-57344.0, 57344.0],
    torch.float8_e4m3fn: [-448.0, 448.0],
    torch.float8_e4m3fnuz: [-240.0, 240.0],
    torch.float16: [-65504.0, 65504.0],
    torch.float32: [-3.4028234663852886e38, 3.4028234663852886e38],
    torch.float64: [-1.7976931348623157e308, 1.7976931348623157e308],
    torch.int16: [-32768, 32767],
    torch.int32: [-2147483648, 2147483647],
    torch.int64: [-9223372036854775808, 9223372036854775807],
}


def torch_dtype_to_wave(torch_dtype: torch.dtype) -> Any:
    try:
        return TORCH_DTYPE_TO_WAVE[torch_dtype]
    except KeyError:
        raise ValueError(f"Unable to map torch dtype {torch_dtype} to Wave.")


def torch_dtype_range(torch_dtype: torch.dtype) -> Any:
    try:
        return TORCH_DTYPE_RANGE[torch_dtype]
    except KeyError:
        raise ValueError(f"Unable to retrieve torch dtype {torch_dtype} range.")


def is_shared_write(node: CustomOp) -> bool:
    return (
        isinstance(node, Write)
        and subs_idxc(node.memory_type.address_space) == SHARED_ADDRESS_SPACE
    )


def is_shared_read(node: CustomOp) -> bool:
    return (
        isinstance(node, Read)
        and subs_idxc(node.memory_type.address_space) == SHARED_ADDRESS_SPACE
    )


def is_gather(custom: CustomOp) -> bool:
    if not isinstance(custom, Read):
        return False
    assert custom.index, f"Read node {custom} does not have an index."
    return any(
        custom.index[x].size > 1
        for x in custom.memory_type.symbolic_shape[:-1]
        if x in custom.index
    )


def print_live_tensors():
    """
    Print all alive torch tensors in program.

    Use for debugging memory leaks.
    """
    import gc

    gc.collect()

    print("------ live tensors ---------")
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (
                hasattr(obj, "data") and torch.is_tensor(obj.data)
            ):
                print(hex(id(obj)), type(obj), obj.size())
        except:
            pass
    print("-----------------------------")


def remove_files_with_extension(directory, extension):
    pattern = os.path.join(directory, "*" + extension)
    files_to_remove = glob.glob(pattern)

    for file_path in files_to_remove:
        try:
            os.remove(file_path)
        except FileNotFoundError:
            print(f"File not found: {file_path}")
        except Exception as e:
            print(f"Error removing {file_path}: {e}")
