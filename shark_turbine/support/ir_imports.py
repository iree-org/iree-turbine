# Copyright 2022 The IREE Authors
# Copyright 2023 Nod Labs, Inc
# SPDX-FileCopyrightText: 2024 The IREE Authors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Unifies all imports of iree.compiler.ir into one place."""

from iree.compiler.ir import (
    AsmState,
    Attribute,
    Block,
    BlockArgument,
    Context,
    DenseElementsAttr,
    DenseResourceElementsAttr,
    FlatSymbolRefAttr,
    FloatAttr,
    FunctionType,
    IndexType,
    InsertionPoint,
    IntegerAttr,
    Location,
    MLIRError,
    Module,
    OpResult,
    Operation,
    RankedTensorType,
    ShapedType,
    StringAttr,
    SymbolTable,
    Type as IrType,
    TypeAttr,
    UnitAttr,
    # Types.
    ComplexType,
    BF16Type,
    Float8E4M3FNUZType,
    F16Type,
    F32Type,
    F64Type,
    Float8E4M3FNType,
    Float8E4M3FNUZType,
    Float8E5M2Type,
    Float8E5M2FNUZType,
    IntegerType,
    RankedTensorType,
    Value,
)

from iree.compiler.passmanager import (
    PassManager,
)

from iree.compiler.dialects import (
    builtin as builtin_d,
    flow as flow_d,
    func as func_d,
    util as util_d,
    arith as arith_d,
    tensor as tensor_d,
)
