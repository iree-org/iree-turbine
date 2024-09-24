# Copyright 2023 Nod Labs, Inc
# SPDX-FileCopyrightText: 2024 The IREE Authors
#
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
    "export_buffers",
    "jittable",
]
