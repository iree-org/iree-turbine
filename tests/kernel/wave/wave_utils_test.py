# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from iree.turbine.kernel.lang import sym
from iree.turbine.kernel.wave.utils import delinearize_index, _simplify_sympy_expr
import sympy
import numpy as np

M = sym.M


def test_delinearize_index():
    shape = [5, 4, 3]
    nd_index = delinearize_index(M, shape)
    np_nd_index = np.unravel_index(23, shape)
    assert np.equal([x.subs({M: 23}) for x in nd_index], np_nd_index).all()


def test_custom_sympy_simplifications():
    mod_expr = sympy.sympify("(floor(a) * 4 + 3) % 16")
    assert str(_simplify_sympy_expr(mod_expr)) == "4*(Mod(floor(a), 4)) + 3"

    floor_expr = sympy.sympify("floor(floor(a)/3 + 1/6)")
    assert str(_simplify_sympy_expr(floor_expr)) == "floor(floor(a)/3)"
