# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
from iree.turbine.kernel.lang import sym
from iree.turbine.kernel.wave.utils.symbol_utils import (
    delinearize_index,
)
from iree.turbine.kernel.wave.utils.mapping_utils import (
    _simplify_sympy_expr,
)
import sympy
import numpy as np

M = sym.M


def test_delinearize_index():
    shape = [5, 4, 3]
    nd_index = delinearize_index(M, shape)
    np_nd_index = np.unravel_index(23, shape)
    assert np.equal([x.subs({M: 23}) for x in nd_index], np_nd_index).all()


def test_custom_sympy_simplifications():
    a = sympy.Symbol("a", integer=True, nonnegative=True)
    mod_expr = (sympy.floor(a) * 4 + 3) % 16
    assert str(_simplify_sympy_expr(mod_expr)) == "4*(Mod(a, 4)) + 3"

    floor_expr = sympy.floor(sympy.floor(a) / 3 + sympy.sympify(1) / 6)
    assert str(_simplify_sympy_expr(floor_expr)) == "floor(a/3)"


@pytest.mark.skip("Too slow")
def test_fuzz_custom_sympy_simplifications_mod():
    x = sympy.Symbol("x", integer=True, nonnegative=True)
    a = sympy.Symbol("a")
    b = sympy.Symbol("b")
    c = sympy.Symbol("c")

    import random

    expr = (sympy.floor(x) * a + b) % c
    total = 0
    outer_num_iters = 1000
    for i in range(outer_num_iters):

        a_val = random.randint(2, 50)
        b_val = random.randint(1, a_val - 1)
        c_val = a_val * random.randint(1, 10)

        vals = [a_val, b_val, c_val]
        expr = expr.subs({a: vals[0], b: vals[1], c: vals[2]})
        expr = sympy.simplify(expr)

        expr2 = _simplify_sympy_expr(expr)

        if i % 50 == 0 and i > 0:
            print(f"{100*i/outer_num_iters}%")

        if expr == expr2:
            print("skip", vals)
            continue

        vals2 = vals + [0, 1]
        for j in range(100):
            val = vals2[j] if j < len(vals2) else random.randint(0, c_val * 2)
            if expr.subs({x: val}) != expr2.subs({x: val}):
                print(f"Failed: {vals}, {val}")

            assert expr.subs({x: val}) == expr2.subs({x: val})
            total += 1

    print(f"Sucess: {total} checks")


@pytest.mark.skip("Too slow")
def test_fuzz_custom_sympy_simplifications_floor():
    x = sympy.Symbol("x", integer=True, nonnegative=True)
    a = sympy.Symbol("a")
    b = sympy.Symbol("b")
    c = sympy.Symbol("c")
    d = sympy.Symbol("d")

    import random

    orig_expr = sympy.floor(sympy.floor(x) * a / b + c / d)

    def check_specific(*vals):
        expr1 = orig_expr.subs({a: vals[0], b: vals[1], c: vals[2], d: vals[3]})
        expr1 = sympy.simplify(expr1)

        expr2 = _simplify_sympy_expr(expr1)
        assert expr1.subs({x: vals[4]}) == expr2.subs({x: vals[4]})

    check_specific(10, 11, 6, 10, 6)
    check_specific(8, 5, 1, 5, 8)

    total = 0
    outer_num_iters = 500
    for i in range(outer_num_iters):
        while True:
            a_val = 1  # random.randint(1, 10)
            b_val = random.randint(1, 10)
            if b_val == a_val:
                b_val += 1

            c_val = random.randint(1, 10)
            d_val = random.randint(1, 10)
            if d_val == c_val:
                d_val += 1

            vals = [a_val, b_val, c_val, d_val]
            expr = orig_expr.subs({a: vals[0], b: vals[1], c: vals[2], d: vals[3]})
            expr = sympy.simplify(expr)

            expr2 = _simplify_sympy_expr(expr)
            if expr != expr2:
                break

        if i % 50 == 0 and i > 0:
            print(f"{100*i/outer_num_iters}%")

        vals2 = vals + [-1, 0, 1]
        for j in range(100):
            val = vals2[j] if j < len(vals2) else random.randint(0, c_val * 2)
            if expr.subs({x: val}) != expr2.subs({x: val}):
                print(f"Failed: {vals}, {val}")

            assert expr.subs({x: val}) == expr2.subs({x: val})
            total += 1

    print(f"Sucess: {total} checks")
