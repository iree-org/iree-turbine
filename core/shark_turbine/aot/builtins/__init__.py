# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from .globals import *
from .jittable import jittable
from ..support.procedural import (
    AbstractBool,
    AbstractF32,
    AbstractF64,
    AbstractI32,
    AbstractI64,
    AbstractIndex,
    AbstractTensor,
    abstractify,
)

# Export the instantiated IREEEmitter as "IREE"
from ..support.procedural.iree_emitter import IREEEmitter as _IREEEmitter

IREE = _IREEEmitter()
del _IREEEmitter

__all__ = [
    "AbstractBool",
    "AbstractF32",
    "AbstractF64",
    "AbstractI32",
    "AbstractI64",
    "AbstractIndex",
    "AbstractTensor",
    "IREE",
    "abstractify",
    "export_global",
    "export_global_tree",
    "export_parameters",
    "jittable",
]
