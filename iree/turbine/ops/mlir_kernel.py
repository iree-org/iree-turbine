# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import ClassVar, Type, cast, Optional, Callable, Dict, Sequence
import inspect
import textwrap
import logging
from dataclasses import dataclass

import torch
from jinja2 import Environment, BaseLoader

from iree.turbine.support.logging import get_logger
from iree.turbine.transforms.merger import Merger
from iree.turbine.support.conversions import TORCH_DTYPE_TO_IREE_TYPE
from iree.turbine.support.ir_imports import (
    Value,
    Operation,
    FlatSymbolRefAttr,
    FunctionType,
    IrType,
    RankedTensorType,
    MLIRError,
    TypeAttr,
)
from iree.turbine.runtime.op_reg import (
    CustomOp,
    KernelSelection,
    KernelBuilder,
    def_library,
)

logger = get_logger("turbine.ops")
_JINJA2_ENVIRONMENT: Optional[Environment] = None


def _get_jinja2_env() -> Environment:
    global _JINJA2_ENVIRONMENT
    if _JINJA2_ENVIRONMENT is None:
        _JINJA2_ENVIRONMENT = Environment(loader=BaseLoader())
    return _JINJA2_ENVIRONMENT


def call_function(target_function: Operation, *operands: Value) -> Sequence[Value]:
    target_symbol = FlatSymbolRefAttr.get(target_function.attributes["sym_name"].value)
    ftype = FunctionType(TypeAttr(target_function.attributes["function_type"]).value)
    return Operation.create(
        "util.call",
        results=ftype.results,
        operands=operands,
        attributes={
            "callee": target_symbol,
        },
    ).results


@dataclass
class _Dim:
    dynamic: bool
    name: str
    value: int | None = None

    def __str__(self) -> str:
        return ("dyn_" if self.dynamic else " ") + self.name

    def __repr__(self) -> str:
        return ("dyn_" if self.dynamic else " ") + self.name

    def __call__(self, val: int):
        if self.dynamic:
            raise ValueError(
                f"Cannot assign value to a dynamic dimension. Tried assigning value: {val} to {str(self)}"
            )
        return _Dim(self.dynamic, self.name, val)


@dataclass
class _Dtype:
    name: str
    value: torch.dtype | None = None

    def __call__(self, val: torch.dtype):
        return _Dtype(self.name, val)


class _StaticDimExpando:
    def __getattr__(self, n: str) -> "_Dim":
        return _Dim(False, n)


class _DynDimExpando:
    def __getattr__(self, n) -> "_Dim":
        return _Dim(True, n)


class _DtypeExpando:
    def __getattr__(self, n) -> "_Dtype":
        return _Dtype(n)


class MLIRTensor:
    """
    A class representing a symbolic tensor type. For example:

    There are 3 types of symbols that can be used to create the type:
        - DynDim: Dynamic Dimensions
        - StaticDim: Static Dimensions
        - Dtype: Static Data Type

    For example:
    ```
        D0 = DynDim.D0
        S0 = StaticDim.S0
        TY = Dtype.TY
        MLIRTensor[D0, S0, TY]
    ```

    The example above describes a 2-D tensor type, whose outer dimension is
    dynamic, the inner dimension is static and the element type is TY. The
    tensor type always assumes the last symbol passed is the data type, and all
    other symbols are StaticDim/DynDim.

    StaticDim and Dtype can also be statically assigned with a value:

    ```
        S0_64 = StaticDim.S0(64)
        TY_I1 = Dtype.TY(torch.bool)
    ```
    """

    shapes: ClassVar[tuple[_Dim, ...]]
    dtype: ClassVar[_Dtype]
    tensor: Optional[torch.Tensor]

    def __init__(self, t: Optional[torch.Tensor] = None):
        self.tensor = t

    def __class_getitem__(
        cls, shape_and_dtype: _Dtype | tuple[_Dim | _Dtype, ...]
    ) -> Type["MLIRTensor"]:
        """Syntax: `KernelBuffer[shape1, shape2, ..., shapeN, dtype]`"""

        if not isinstance(shape_and_dtype, tuple):
            shape_and_dtype = tuple([shape_and_dtype])

        if len(shape_and_dtype) < 1:
            raise TypeError(f"Expected at least 1 argument, got: {shape_and_dtype}")

        shape = shape_and_dtype[:-1]
        ty = shape_and_dtype[-1]

        if not all(isinstance(s, _Dim) for s in shape):
            raise TypeError(f"Expected shape to be a tuple of _Dim, got {shape}")
        if not isinstance(ty, _Dtype):
            raise TypeError(f"Expected dtype to be a _Dtype, got {ty}")

        shape = cast(tuple[_Dim, ...], shape)
        ty = cast(_Dtype, ty)

        class SubType(cls):
            shapes = shape
            dtype = ty

        return cast(Type["MLIRTensor"], SubType)


class MLIRSpec:
    """
    A class representing a MLIR Jinja2 template.

    The `subs` dictionary contains additional substitutions passed to the jinja
    template generator.

    Note that currently, no specialization is done on the extra substitutions
    passed, it is recommended to pass all specialization information through
    input/output tensor type description.
    """

    mlir: str
    subs: dict

    def __init__(self, mlir: str, subs: dict = {}):
        self.mlir = mlir
        self.subs = subs


def mlir_kernel(
    *,
    library: str,
    inputs: tuple[type[MLIRTensor], ...],
    results: tuple[type[MLIRTensor], ...],
):
    """
    A decorator that allows a user to inject inline mlir kernels directly into
    the model.

    The mlir_kernel decorator takes an input/output spec containing tensor type
    descriptions. These tensor type descriptions are used to automatically
    specialize the kernel and generate verifiers. The kernel is specialized on
    static dimensions and dtypes, and is not specialized on dynamic dimensions.

    The decorator takes a function, with the same number of input/output
    arguments as the input spec. These function argument names are used to
    generate type aliases in the given mlir_spec.

    Example:

    ```
    D0 = DynDim.D0
    S0 = StaticDim.S0
    TY = Dtype.TY
    @mlir_kernel(
        library="example"
        inputs=(MLIRTensor[D0, S0, TY],),
        results = (MLIRTensor[D0, S0, TY],)
    )
    def identity(input, result=None):
        mlir = \"""

        // The mlir spec must be wrapped around in a module. We might change this
        // in the future and not ask the user to write a util.func and module well.
        module {

        // Note that the function name MUST be the passed `kernel_name` alias.
        // Also note the type aliases passed for the input and result argument.
        util.func private @{{kernel_name}}(%input: !input) -> !result {

            // dtype aliases are generated as `<argname>_dtype`
            // dimension and dtype values are accessible as aliases. For dynamic
            // dimensions the alias is `?`.
            %c0 = arith.constant {{S0}} : !input_dtype

            util.return %input : !result
        }

        }

        \"""
    ```
    """

    kernel_lib = def_library(library)

    def fun(func: Callable[..., MLIRSpec]) -> Callable:
        sig = inspect.signature(func)
        params = sig.parameters
        args = list(params.keys())

        if len(args) != len(inputs) + len(results):
            raise TypeError(
                "Number of arguments to kernel should be same as the mlir_kernel spec"
            )

        input_args = args[: len(inputs)]
        result_args = args[len(inputs) :]

        # Create a dimension mapping to input dimensions

        @CustomOp.register(library=kernel_lib)
        class kernel(CustomOp):
            @property
            def signature(self) -> str:
                input_tensors = [f"Tensor {arg}" for arg in input_args]
                result_tensors = ["Tensor" for _ in result_args]
                return f'{func.__name__}({", ".join(input_tensors)}) -> ({", ".join(result_tensors)})'

            def select(self, sel: KernelSelection):
                # Create input descriptions.
                input_descs = [sel.arg_tensor(i) for i in range(len(input_args))]

                # Specialize static dimensions.
                for sym_ty, desc in zip(inputs, input_descs):
                    static_dims = [
                        i for i, dim in enumerate(sym_ty.shapes) if not dim.dynamic
                    ]
                    desc.specialize_dims(*static_dims)

                # Resolve shape and dtype symbols.
                dims = {
                    sym_dim.name: sym_dim.value
                    for sym_ty in inputs + results
                    for sym_dim in sym_ty.shapes
                    if sym_dim.value is not None
                }
                dtypes = {
                    sym_ty.dtype.name: sym_ty.dtype.value
                    for sym_ty in inputs + results
                    if sym_ty.dtype.value is not None
                }
                for sym_ty, ty in zip(inputs, input_descs):
                    if len(sym_ty.shapes) != len(ty.t.shape):
                        raise ValueError(
                            f"Mismatched input rank. Expected: {sym_ty.shapes}, got: {ty.t.shape}"
                        )
                    # Resolve shape symbols.
                    for sym_dim, dim in zip(sym_ty.shapes, ty.t.shape):
                        if sym_dim.name in dims:
                            if not sym_dim.dynamic and dims[sym_dim.name] != dim:
                                raise ValueError(
                                    f"Mismatched dim error. Expected value for {sym_dim.name}: {dims[sym_dim.name]}, got: {dim}"
                                )
                        else:
                            dims[sym_dim.name] = dim
                    # Resolve dtype symbols.
                    if sym_ty.dtype.name in dtypes:
                        if dtypes[sym_ty.dtype.name] != ty.t.dtype:
                            raise ValueError(
                                f"Mismatched dtype error. Expected value for {sym_ty.dtype.name}: {dtypes[sym_ty.dtype.name]}, got: {ty.t.dtype}"
                            )
                    else:
                        dtypes[sym_ty.dtype.name] = ty.t.dtype

                # Specialize static dimensions on return type.
                for sym_ty in results:
                    resolved_shape = [dims[dim.name] for dim in sym_ty.shapes]
                    resolved_dtype = dtypes[sym_ty.dtype.name]
                    desc = sel.return_new_tensor(
                        size=resolved_shape, dtype=resolved_dtype
                    )
                    static_dims = [
                        i for i, dim in enumerate(sym_ty.shapes) if not dim.dynamic
                    ]
                    desc.specialize_dims(*static_dims)

            def generate(self, ksel: KernelSelection, kb: KernelBuilder):
                # Create input descriptions and types.
                input_values = [kb.arg_value(i) for i in range(len(input_args))]
                input_types = [RankedTensorType(val.type) for val in input_values]

                # Resolve shape and dtype symbols.
                dims: Dict[str, int | None] = {
                    sym_dim.name: sym_dim.value
                    for sym_ty in inputs + results
                    for sym_dim in sym_ty.shapes
                    if sym_dim.value is not None
                }
                dtypes: Dict[str, IrType] = {
                    sym_ty.dtype.name: TORCH_DTYPE_TO_IREE_TYPE[sym_ty.dtype.value]()
                    for sym_ty in inputs + results
                    if sym_ty.dtype.value is not None
                }
                for sym_ty, ty in zip(inputs, input_types):
                    # Resolve shape symbols.
                    for sym_dim, dim in zip(sym_ty.shapes, ty.shape):
                        if sym_dim.dynamic:
                            # For dynamic dimensions, map the dim to None.
                            dim = None

                        if sym_dim.name in dims:
                            if dims[sym_dim.name] != dim:
                                raise ValueError(
                                    f"Mismatched dim error. Expected value for {sym_dim.name}: {dims[sym_dim.name]}, got: {dim}"
                                )
                        else:
                            dims[sym_dim.name] = dim
                    # Resolve dtype symbols.
                    if sym_ty.dtype.name in dtypes:
                        if dtypes[sym_ty.dtype.name] != ty.element_type:
                            raise ValueError(
                                f"Mismatched dtype error. Expected value for {sym_ty.dtype.name}: {dtypes[sym_ty.dtype.name]}, got: {ty.element_type}"
                            )
                    else:
                        dtypes[sym_ty.dtype.name] = ty.element_type

                # Generate kernel name.
                kernel_name = self._get_kernel_name(func.__name__, dims, dtypes)

                # Try to check if the symbol table already has a generated
                # kernel for this specialization.
                symbol_name = None
                try:
                    symbol_name = kb.symbol_table[kernel_name]
                except KeyError:
                    pass

                # If this kernel is not already generated, generate it using
                # the mlir spec.
                if symbol_name is None:
                    # Get the MLIR spec.
                    mlir_spec = func(*input_values, *([None] * len(result_args)))

                    # Insert type aliases to the mlir_spec.
                    mlir = self._get_type_aliases(dims, dtypes) + mlir_spec.mlir

                    # Generate the MLIR spec using jinja.
                    asm = (
                        _get_jinja2_env()
                        .from_string(mlir)
                        .render(
                            {
                                "kernel_name": kernel_name,
                                **self._get_additional_aliases(dims),
                                **mlir_spec.subs,
                            }
                        )
                    )
                    try:
                        module_op = Operation.parse(asm, context=kb.context)
                    except MLIRError as e:
                        lines = asm.splitlines()
                        lines_numbered = "\n".join(
                            [f"      {str(i+1):>5}: {l}" for i, l in enumerate(lines)]
                        )
                        raise RuntimeError(
                            f"Error parsing generated op template:"
                            f"\n{textwrap.indent(str(e), '  ')}"
                            f"\n{lines_numbered}"
                        )
                    op = module_op.operation

                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            "Generated kernel IR %s:\n%s", kernel_name, str(op)
                        )
                    merger = Merger(
                        op, kb.module_body.owner, target_symbol_table=kb.symbol_table
                    )
                    merger.merge()

                    symbol_name = kb.symbol_table[kernel_name]

                kb.yield_results(*call_function(symbol_name, *kb.arg_bindings))

            def _get_type_aliases(
                self, dims: Dict[str, Optional[int]], dtypes: Dict[str, IrType]
            ) -> str:
                aliases = ""
                for arg, sym_ty in zip(input_args, inputs):
                    mlir_shapes = [
                        "?" if dims[dim.name] is None else str(dims[dim.name])
                        for dim in sym_ty.shapes
                    ]
                    dtype = dtypes[sym_ty.dtype.name]
                    aliases += (
                        f'!{arg} = tensor<{"x".join(mlir_shapes + [str(dtype)])}>\n'
                    )
                    aliases += f"!{arg}_dtype = {dtype}\n"
                for arg, sym_ty in zip(result_args, results):
                    mlir_shapes = [
                        "?" if dims[dim.name] is None else str(dims[dim.name])
                        for dim in sym_ty.shapes
                    ]
                    dtype = dtypes[sym_ty.dtype.name]
                    aliases += (
                        f'!{arg} = tensor<{"x".join(mlir_shapes + [str(dtype)])}>\n'
                    )
                    aliases += f"!{arg}_dtype = {dtype}\n"
                return aliases

            def _get_kernel_name(
                self,
                prefix: str,
                dims: Dict[str, Optional[int]],
                dtypes: Dict[str, IrType],
            ) -> str:
                kernel_name = prefix

                # Add input args as suffix.
                kernel_name += "_"
                input_names = []
                for sym_ty in inputs:
                    input_dims = []
                    for sym_dim in sym_ty.shapes:
                        input_dim = sym_dim.name
                        if not sym_dim.dynamic:
                            input_dim += f"_{dims[sym_dim.name]}"
                        input_dims.append(input_dim)
                    input_name = (
                        "_".join(input_dims) + "_" + str(dtypes[sym_ty.dtype.name])
                    )
                    input_names.append(input_name)
                kernel_name += "_".join(input_names)

                # Add result args as suffix.
                result_names = []
                kernel_name += "_"
                for sym_ty in results:
                    result_dims = []
                    for sym_dim in sym_ty.shapes:
                        result_dim = sym_dim.name
                        if not sym_dim.dynamic:
                            result_dim += f"_{dims[sym_dim.name]}"
                        result_dims.append(result_dim)
                    result_name = (
                        "_".join(result_dims) + "_" + str(dtypes[sym_ty.dtype.name])
                    )
                    result_names.append(result_name)
                kernel_name += "_".join(result_names)

                return kernel_name

            def _get_additional_aliases(
                self, dims: Dict[str, Optional[int]]
            ) -> Dict[str, str]:
                return dict(
                    {
                        (dim, "?") if val is None else (dim, str(val))
                        for dim, val in dims.items()
                    }
                )

        return kernel

    return fun


StaticDim = _StaticDimExpando()
DynDim = _DynDimExpando()
Dtype = _DtypeExpando()

__all__ = ["StaticDim", "DynDim", "Dtype", "MLIRTensor", "mlir_kernel", "MLIRSpec"]
