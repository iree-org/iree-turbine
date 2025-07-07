# Copyright 2025 The IREE Authors
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import TypeVar

import sympy

from ..._support.indexing import IndexingContext
from ...lang.wave_types import IndexMapping
from .general_utils import infer_dim
from .symbol_utils import IndexExpr, IndexSymbol, subs_idxc

K = TypeVar("K")  # Key type
V = TypeVar("V")  # Value type


def get_dict_with_updated_key(
    original_dict: dict[K, V], old_key: K, new_key: K
) -> dict[K, V]:
    """
    Update a key in a dictionary while preserving the original insertion order of values.

    Creates a new dictionary identical to the original except that the specified old key
    is replaced with the new key. All values and ordering remain unchanged.
    """
    if old_key not in original_dict:
        raise KeyError(f"Old key '{old_key}' not found in dictionary")
    if new_key in original_dict and new_key != old_key:
        raise KeyError(f"New key '{new_key}' already exists in dictionary")

    # Create a new dictionary with the same order but updated key
    new_dict = {}
    for key, value in original_dict.items():
        if key == old_key:
            new_dict[new_key] = value
        else:
            new_dict[key] = value

    return new_dict


def _simplify_sympy_expr(expr: IndexExpr) -> IndexExpr:
    """Apply custom sympy simplifications"""

    def check_mul(mul):
        ret = None
        for arg in mul.args:
            if arg.is_number:
                if arg < 0:
                    return None

                if ret is not None:
                    return None

                ret = arg
                continue

            if not (isinstance(arg, (sympy.floor, sympy.Mod)) or arg.is_integer):
                return None

            if not arg.is_nonnegative:
                return None

        return ret

    def transform_mod(expr):
        """Move constant outside of Mod expr

        Example:
        (floor(a) * 4 + 3) % 16 -> (floor(a) * 4) % 16 + 3
        """
        if not isinstance(expr, sympy.Mod):
            return None

        p, q = expr.args
        if not q.is_number or q < 0:
            return None

        if not isinstance(p, sympy.Add):
            return None

        c = None
        terms = []
        mult = None
        for arg in p.args:
            if arg.is_number:
                if c is not None:
                    return None

                c = arg
                continue

            if not isinstance(arg, sympy.Mul):
                return None

            m = check_mul(arg)
            if (m is None) or (q % m != 0):
                return None

            mult = m if (mult is None) or (m < mult) else mult
            terms.append(arg)

        if c >= mult:
            return None

        return (sum(terms) % q) + c

    def check_mul_rational(mul):
        ret = None
        for arg in mul.args:
            if isinstance(arg, sympy.Rational):
                if ret is not None:
                    return None

                if arg.p < 0 or arg.q < 0:
                    return None

                ret = arg
                continue

            if not (isinstance(arg, (sympy.floor, sympy.Mod)) or arg.is_integer):
                return None

            if not arg.is_nonnegative:
                return None

        return ret

    def transform_floor(expr):
        """Simplify rational addition inside floor expr

        Example:
        floor(floor(a)/3 + 1/6) -> floor(floor(a)/3)
        """
        if not isinstance(expr, sympy.floor):
            return None

        expr = expr.args[0]
        if not isinstance(expr, sympy.Add):
            return None

        c = None
        for arg in expr.args:
            if isinstance(arg, sympy.Rational):
                if c is not None:
                    return None

                c = arg

        if c is None:
            return None

        terms = []
        for arg in expr.args:
            if isinstance(arg, sympy.Rational):
                continue

            if not isinstance(arg, sympy.Mul):
                return None

            r = check_mul_rational(arg)
            if r is None or r.p != 1:
                return None

            if r <= c:
                return None

            terms.append(arg)

        return sympy.floor(sum(terms))

    expr = expr.replace(lambda e: transform_mod(e) is not None, transform_mod)
    expr = expr.replace(lambda e: transform_floor(e) is not None, transform_floor)
    return sympy.simplify(expr)


def approximate_difference(
    expr: IndexExpr, vars: list[IndexSymbol], elements_per_thread: int
) -> bool:
    """
    During the contiguity check, we take a unit step in the fastest changing
    dimension (j -> j + 1) and we compute f(j + 1) - f(j) to see if it is 1.
    In general, we will end up with expressions of the form
    g(x + eps) - g(x) where x = h(j) and eps is a rational of the form 1/q.
    We can use q to determine if the mapping is contiguous as follows

    if q is divisible by elements_per_thread (dimensions where we have not applied the unit step), or
    if eps is 1 (corresponds to the dimension where we have applied the unit step)
    then the mapping is contiguous.

    The mapping function f(j) will be non-linear in general, and so the difference
    of 1 will be transformed to different constant values based on the function.
    But, if we recover a value of 1, we can assume that the function preserves
    the difference.

    In this function we do a pre-order traversal of the expression to obtain
    the value of the constant eps.
    """
    if expr.is_number:
        return expr
    new_vars, new_exprs = sympy.cse(expr)
    new_expr = new_exprs[0] if new_vars else expr
    new_vars = [x[0] for x in new_vars] if new_vars else vars
    for arg in sympy.preorder_traversal(new_expr):
        if isinstance(arg, sympy.Add):
            if all([x in arg.args for x in new_vars]):
                constant = [x for x in arg.args if x not in new_vars][0]
                if not isinstance(constant, sympy.Rational):
                    return expr
                if constant.p != 1:
                    return expr
                if constant.q == 1:
                    return 1
                return 0 if constant.q % elements_per_thread == 0 else expr
    return expr


def check_is_mapping_contiguous(
    mapping: IndexMapping,
    symbolic_shape: tuple[IndexExpr, ...],
    index: tuple[IndexExpr, ...],
    elements_per_thread: int | IndexExpr,
    is_read: bool,
) -> bool:
    """Check if mapping can be lowered to contiguous vector ops instead of gathers/scatters"""
    elements_per_thread = subs_idxc(elements_per_thread)
    if elements_per_thread == 1:
        return True

    # TODO: Better dyn vals analysis.
    if mapping.num_dynamic_vals != 0:
        return False

    if is_read:
        assert (
            mapping.is_output_identity()
        ), "non-identity output mapping is not supported yet"
        symbolic_dims = [infer_dim(dim_size) for dim_size in symbolic_shape]
        index_mapping = mapping.map_input_indices(symbolic_dims)
    else:
        assert (
            mapping.is_input_identity()
        ), "non-identity input mapping is not supported yet"
        index_mapping = mapping.map_output_indices(symbolic_shape)

    index_mapping = tuple(subs_idxc(i) for i in index_mapping)
    iters = mapping.iters

    subs = [(sym, sym + int(i == len(iters) - 1)) for i, sym in enumerate(iters)]
    diff = [
        approximate_difference(
            index_mapping[i].subs(subs) - index_mapping[i],
            list(iters.keys())[-1:],
            elements_per_thread,
        )
        for i in range(len(index_mapping))
    ]

    expected_diff = [0] * len(index_mapping)
    expected_diff[-1] = 1

    return diff == expected_diff


def transform_index_on_mapping(
    mapping: IndexMapping,
    symbolic_shape: tuple[IndexExpr, ...],
    index: tuple[IndexExpr, ...],
) -> tuple[IndexExpr, ...]:
    """ "Transforms the index according to the specified mapping"""
    input_index_mapping = mapping.map_input_indices(symbolic_shape)
    idxc = IndexingContext.current()
    index_mapping = tuple(i.subs(idxc.subs) for i in input_index_mapping)
    iters = mapping.iters
    subs = [
        (sym, expr.start) for sym, expr in zip(iters.keys(), index.values())
    ] + list(idxc.subs.items())
    transformed_index = {
        key: m.subs(subs) for key, m in zip(symbolic_shape, index_mapping)
    }

    return transformed_index
