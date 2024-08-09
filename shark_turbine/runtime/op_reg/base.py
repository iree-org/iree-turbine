# Copyright 2023 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Base classes for registering custom operations with the PyTorch
dispatcher.
"""

from typing import Any, Callable, List, Optional, Sequence, Type, Union, cast

from abc import ABC, abstractmethod
import functools
import logging
import re
import textwrap
import threading

import torch
from torch import Tensor

from ...support.ir_imports import (
    Block,
    Context,
    FunctionType,
    IndexType,
    InsertionPoint,
    IntegerAttr,
    Location,
    StringAttr,
    SymbolTable,
    IrType,
    Value,
    arith_d,
    builtin_d,
    func_d,
)

from ...support.logging import runtime_logger as logger

from ...support.conversions import (
    TORCH_DTYPE_TO_IREE_TYPE_ASM,
)

__all__ = [
    "ArgDescriptor",
    "AttrArg",
    "CustomOp",
    "FreeFuncKernelBuilder",
    "IntArg",
    "KernelBuilder",
    "KernelSelection",
    "TensorArg",
    "def_library",
]


###############################################################################
# Op library management
###############################################################################

_CONFIG_LOCK = threading.Lock()


def def_library(ns) -> torch.library.Library:
    """Creates a new 'DEF' library which contains custom ops.

    It is necessary to create such custom op libraries in this way since
    the library is registered with the compiler in such a way that it can
    operate over all known custom ops.
    """
    return torch.library.Library(ns, "DEF")


def default_dispatch_keys() -> list[str]:
    keys = ["CPU"]
    if torch.cuda.is_available():
        keys.append("CUDA")
    return keys


# All such custom kernels are registered in the 'turbine' library/namespace.
# We also allow extending existing libraries outside of this, but that is
# the non default case.
TURBINE_LIBRARY = def_library("turbine")

# Set of all programmatically registered op names in libraries we manage.
# This is used to detect name collisions eagerly and providing name uniqueing.
# Keys are (Library.ns, name)
DEFINED_OP_NAMES: set[tuple[str, str]] = set()

# Mapping of (Library.ns, name_spec) to an integer counter used to unique it.
UNIQUE_OP_NAME_COUNTER: dict[tuple[str, str], int] = {}


class CustomOp(ABC):
    """Users subclass this in order to register a turbine custom op."""

    @staticmethod
    def register(
        op_class: Optional[Type["CustomOp"]] = None,
        *,
        library: torch.library.Library = TURBINE_LIBRARY,
        dispatch_key: Union[str, Sequence[str], None] = None,
        register_meta: bool = True,
        register_impl: bool = True,
    ) -> Callable:
        """Class decorator for `CustomOp` implementations.

        The decorator will instantiate the class and then replace it with
        the callable operation that can be used to invoke the kernel.

        Typical usage:

        ```
        @CustomOp.register
        class identity(CustomOp):
          ...

        result = identity(torch.tensor(1, 2, 3))
        ```
        """
        if not op_class:
            return functools.partial(
                CustomOp.register,
                library=library,
                dispatch_key=dispatch_key,
                register_meta=register_meta,
                register_impl=register_impl,
            )
        instance = op_class(
            library=library,
            dispatch_key=dispatch_key,
            register_meta=register_meta,
            register_impl=register_impl,
        )
        return instance.op

    def __init__(
        self,
        *,
        library: torch.library.Library,
        dispatch_key: Union[str, Sequence[str], None],
        register_meta: bool,
        register_impl: bool,
    ):
        self.name = name = _define_signature_in_library(library, self.signature)
        self.library = library
        self.cache_key_base = f"{library.ns}.{library.kind}::{name}"
        self.op = _get_library_op(library, name)

        # The meta kernel can be provided by the selection machinery and
        # does not require a tie-in to the kernel generator, which layers
        # on top.
        if register_meta:
            library.impl(name, _get_meta_impl(self), "Meta")

        if register_impl:
            if dispatch_key is None:
                dispatch_key = default_dispatch_keys()
            elif isinstance(dispatch_key, str):
                dispatch_key = [dispatch_key]
            for k in dispatch_key:
                library.impl(name, _create_impl_trampoline(self), k)

        fq_name = f"{library.ns}.{name}"
        ALL_CUSTOM_OP_REGS[fq_name] = self

    @property
    @abstractmethod
    def signature(self) -> str:
        """PyTorch function signature.

        This is in the normal PyTorch kernel registration form. For example:

        ```
        my_op(Tensor t) -> Tensor
        ```

        The signature can have some special tokens in the name part:

        * "@UNIQUE@": Generates a name-specific numeric value and replaces it.
        """
        ...

    @abstractmethod
    def select(self, sel: "KernelSelection"):
        """Performs kernel selection.

        This method has three purposes:

          1. Selects which kernel specialization is needed based on
             arguments.
          2. Returns the meta tensor results of the operation, effectively
             completing the transfer function from argument types to
             result types.
          3. Sets additional metadata that the generate method can use.

        The `device="meta"` kernel implementation is composed completely by
        invoking `select`. For implementation devices, `select` is called
        for each invocation. The `generate` will be called subsequently if
        the kernel needs to be generated.
        """
        ...

    def eager_execute(self, *args):
        """When executing eagerly, allows the CustomOp to provide a direct Python
        implementation. For AOT/Graph modes, this will not be called.

        If the method returns NotImplemented, then a standalone kernel will be
        compiled and executed.

        This is commonly used for ops that have no significance to a single op
        execution in the PyTorch runtime (e.g. metadata ops), but could theoretically
        be used to perform any Python analog desired.
        """
        return NotImplemented

    @abstractmethod
    def generate(self, ksel: "KernelSelection", kb: "KernelBuilder"):
        """Generates a kernel based on the `KernelSelection`.

        This method should generate IR into the given `KernelBuilder`. It
        can do so by consulting any state set on the `KernelSelection`.
        Each `KernelSelection.args` corresponds to `KernelBuilder.args`.
        Unless if the argument was set as `ir_arity=0`, the argument
        will be a `Value`. Otherwise, it will be `None`. It is recommended
        to use `KernelBuilder.arg(n)` to access.

        Generation should conclude with a call to `KernelBuilder.yield_results`.
        """
        ...


# All instantiated CustomOp instances, keyed by fully qualified name. This is
# used by the AOT compiler to expand custom ops that were captured in a trace.
ALL_CUSTOM_OP_REGS: dict[str, CustomOp] = {}


class KernelSelection(ABC):
    """Represents a selected kernel based on a concrete signature.

    The `CustomOp.select` method must yield an instance of this, and
    it will be done for every invocation. At this point, the kernel
    has not yet been generated, but we have selected a generation
    strategy based on a concrete signature.

    This mechanism also serves as the means for servicing `meta`
    registrations because it implicitly computes everything needed
    (i.e. shapes, etc).
    """

    __slots__ = [
        "arg_descs",
        "inplace_tied_arg_descs",
        "op",
        "result_descs",
        "variant",
    ]

    def __init__(self, op: CustomOp, arg_arity: int):
        self.op = op
        self.arg_descs = cast(list[Optional[ArgDescriptor]], arg_arity * [None])
        self.inplace_tied_arg_descs: list[ArgDescriptor] = []
        self.result_descs: list[ArgDescriptor] = []
        self.variant: str = "default"

    def __repr__(self):
        lines = [
            "KernelSelection<",
            f"  op = '{self.op.name}',",
            f"  variant = '{self.variant}',",
            "  arg_descs = [",
        ]
        for arg_desc in self.arg_descs:
            lines.append(f"    {arg_desc},")
        lines.append("  ],")
        lines.append("  result_descs = [")
        for result_desc in self.result_descs:
            lines.append(f"    {result_desc},")
        lines.append("  ]")
        lines.append(">")
        return "\n".join(lines)

    def generate_meta_returns(self) -> Any:
        results = [d.generate_meta() for d in self.result_descs]
        arity = len(results)
        if arity == 1:
            return results[0]
        elif arity == 0:
            return None
        else:
            return tuple(results)

    @property
    def spec_key(self) -> str:
        try:
            arg_keys = ",".join(
                d.spec_key if d is not None else "None" for d in self.arg_descs
            )
            return_keys = ",".join(
                d.spec_key if d is not None else "None" for d in self.result_descs
            )
            return (
                f"{self.op.cache_key_base}::{self.variant}({arg_keys})->({return_keys})"
            )
        except Exception as e:
            raise AssertionError(
                f"Error generating spec_key from:\n{textwrap.indent(repr(self), '  ')}"
            ) from e

    @abstractmethod
    def arg_tensor(self, arg: int, *, inplace_tied: bool = False) -> "TensorArg":
        """Declares an argument to allow any ranked tensor and to specialize for each rank
        and dtype.

        Returns the argument descriptor, which can be used to further inspect or constrain
        the selection. It will default to allowing all dimensions to be dynamic.

        If inplace_tied is True, then this argument participates in in-place
        semantics. The kernel must yield the result-mutated after all normal
        results in the order declared.
        """
        ...

    @abstractmethod
    def arg_tensor_list(self, arg: int) -> "TensorListArg":
        """Declares an argument to accept a list of tensors which will be specialized
        for the list size and each rank/dtype.

        Returns the argument descriptor, which can be used to further inspect or constrain
        the selection. It will default to allowing all dimensions to be dynamic.
        """
        ...

    @abstractmethod
    def arg_int(self, arg: int) -> "IntArg":
        """Declares an argument to be an integer value that can take any value.

        Returns the argument descriptor, which can be used to further inspect or constrain
        the selection.
        """
        ...

    @abstractmethod
    def attr_str(self, arg: int) -> "AttrArg":
        """Declares an argument to be a string attribute.

        Such arguments are not materialized in the IR as Values but may be used to
        generate the IR. In AOT contexts, they must be derived from static values.
        """
        ...

    @abstractmethod
    def attr_int(self, arg: int) -> "AttrArg":
        """Declares an argument to be an integer attribute.

        Such arguments are not materialized in the IR as Values but may be used to
        generate the IR. In AOT contexts, they must be derived from static values.
        """
        ...

    @abstractmethod
    def attr_list_int(self, arg: int) -> "AttrArg":
        """Declares an argument to be a list<integer> attribute.

        Such arguments are not materialized in the IR as Values but may be used to
        generate the IR. In AOT contexts, they must be derived from static values.
        """
        ...

    @abstractmethod
    def attr_float(self, arg: int) -> "AttrArg":
        """Declares an argument to be a float attribute.

        Such arguments are not materialized in the IR as Values but may be used to
        generate the IR. In AOT contexts, they must be derived from static values.
        """
        ...

    @abstractmethod
    def attr_list_float(self, arg: int) -> "AttrArg":
        """Declares an argument to be a list<float> attribute.

        Such arguments are not materialized in the IR as Values but may be used to
        generate the IR. In AOT contexts, they must be derived from static values.
        """
        ...

    @abstractmethod
    def return_tensor(self, t: Tensor) -> "TensorArg":
        """Marks the next return value as a Tensor.

        By default, it will be rank and dtype specialized but have completely dynamic
        dimensions. Dimensions can be further constrained by modifying the returned
        descriptor.
        """
        ...

    def return_new_tensor(self, size: list, dtype: torch.dtype) -> "TensorArg":
        """Constructs a new symbolic tensor and marks the next result as returning it.

        This delegates to `return_tensor` but takes care of some easy to mess
        up boiler plate for dynamic shapes.
        """
        return self.return_tensor(torch.empty(size, dtype=dtype, device="meta"))


class EagerKernelSelection(KernelSelection):
    """Kernel selection specialized for eager arguments."""

    __slots__ = [
        "args",
    ]

    def __init__(self, op: CustomOp, args: list[Any]):
        super().__init__(op, len(args))
        self.args = args

    def arg_tensor(self, arg: int, *, inplace_tied: bool = False) -> "TensorArg":
        arg_descs = self.arg_descs
        arg_value = self.args[arg]
        assert arg_descs[arg] is None, f"Already constrained argument {arg}"
        assert isinstance(
            arg_value, Tensor
        ), f"Argument type mismatch from Torch for {arg}: Expected tensor, got {type(arg_value)}"
        arg_descs[arg] = desc = TensorArg(arg_value)
        if inplace_tied:
            self.inplace_tied_arg_descs.append(desc)
        return desc

    def arg_tensor_list(self, arg: int) -> "TensorListArg":
        arg_descs = self.arg_descs
        arg_value = self.args[arg]
        assert arg_descs[arg] is None, f"Already constrained argument {arg}"
        assert isinstance(
            arg_value, list
        ), f"Argument type mismatch from Torch for {arg}: Expected list, got {type(arg_value)}"
        arg_descs[arg] = desc = TensorListArg(arg_value)
        return desc

    def arg_int(self, arg: int) -> "IntArg":
        arg_descs = self.arg_descs
        arg_value = self.args[arg]
        assert arg_descs[arg] is None, f"Already constrained argument {arg}"
        assert isinstance(
            arg_value, int
        ), f"Argument type mismatch from Torch for {arg}: Expected int, got {type(arg_value)}"
        arg_descs[arg] = desc = IntArg(arg_value)
        return desc

    def attr_str(self, arg: int) -> "AttrArg":
        arg_descs = self.arg_descs
        arg_value = self.args[arg]
        assert arg_descs[arg] is None, f"Already constrained argument {arg}"
        assert isinstance(
            arg_value, str
        ), f"Argument type mismatch from Torch for {arg}: Expected str, got {type(arg_value)}"
        arg_descs[arg] = desc = AttrArg(arg_value)
        return desc

    def attr_int(self, arg: int) -> "AttrArg":
        arg_descs = self.arg_descs
        arg_value = self.args[arg]
        assert arg_descs[arg] is None, f"Already constrained argument {arg}"
        assert isinstance(
            arg_value, int
        ), f"Argument type mismatch from Torch for {arg}: Expected int, got {type(arg_value)}"
        arg_descs[arg] = desc = AttrArg(arg_value)
        return desc

    def attr_list_int(self, arg: int) -> "AttrArg":
        arg_descs = self.arg_descs
        arg_value = self.args[arg]
        assert arg_descs[arg] is None, f"Already constrained argument {arg}"
        assert isinstance(
            arg_value, list
        ), f"Argument type mismatch from Torch for {arg}: Expected list, got {type(arg_value)}"
        if len(arg_value) > 0:
            assert isinstance(
                arg_value[0], int
            ), f"Argument type mismatch from Torch for {arg}: Expected list of int, got element type of {type(arg_value[0])}"
        arg_descs[arg] = desc = AttrArg(arg_value)
        return desc

    def attr_float(self, arg: int) -> "AttrArg":
        arg_descs = self.arg_descs
        arg_value = self.args[arg]
        assert arg_descs[arg] is None, f"Already constrained argument {arg}"
        assert isinstance(
            arg_value, float
        ), f"Argument type mismatch from Torch for {arg}: Expected float, got {type(arg_value)}"
        arg_descs[arg] = desc = AttrArg(arg_value)
        return desc

    def attr_list_float(self, arg: int) -> "AttrArg":
        arg_descs = self.arg_descs
        arg_value = self.args[arg]
        assert arg_descs[arg] is None, f"Already constrained argument {arg}"
        assert isinstance(
            arg_value, list
        ), f"Argument type mismatch from Torch for {arg}: Expected list, got {type(arg_value)}"
        for arg_value_i in arg_value:
            assert isinstance(
                arg_value_i, float
            ), f"Argument type mismatch from Torch for {arg}: Expected list of float, got element type of {type(arg_value_i)}"
        arg_descs[arg] = desc = AttrArg(arg_value)
        return desc

    def return_tensor(self, t: Tensor) -> "TensorArg":
        desc = TensorArg(t)
        self.result_descs.append(desc)
        return desc


class AttrArg:
    ir_arity: int = 0
    maybe_tensor_value: Optional[Tensor] = None
    is_list: bool = False

    __slots__ = [
        "v",
        "spec_value",
    ]

    def __init__(self, v: object):
        self.v = v
        # We specialize on every distinct value.
        self.spec_value: Optional[Any] = v

    def __repr__(self):
        return f"AttrArg(<{self.spec_value}>)"

    def generate_meta(self) -> object:
        return self.v

    @property
    def spec_key(self) -> str:
        """Generates a key that will be the same for all specializations."""
        return f"attr<{self.spec_value}>"

    @property
    def mlir_type_asm(self) -> str:
        raise AssertionError("Cannot resolve `mlir_type_asm` for an AttrArg")


class IntArg:
    __slots__ = [
        "ir_arity",
        "spec_value",
        "v",
    ]

    # All descriptors have an attribute to indicate their value
    # as a tensor, and those that aren't are fixated to None.
    # This is to enable fast lookup in the hot path of determining
    # how to dispatch.
    maybe_tensor_value: Optional[Tensor] = None
    is_list: bool = False

    def __init__(self, v: int):
        self.v = v
        self.spec_value: Optional[Any] = None
        self.ir_arity: int = 1

    def __repr__(self):
        return f"IntArg({self.v}, spec_value={self.spec_value}, is_ir_arg={self.is_ir_arg})"

    def generate_meta(self) -> int:
        return self.v

    @property
    def spec_key(self) -> str:
        """Generates a key that will be the same for all specializations."""
        return f"int<{self.spec_value}>"

    @property
    def mlir_type_asm(self) -> str:
        # TODO: We can have individual kernels constrain this to a narrower
        # type.
        return "i64"


_NoneInt: Optional[int] = None


class TensorArg:
    __slots__ = [
        "t",
        "spec_dims",
        "maybe_tensor_value",
    ]

    ir_arity: int = 1
    is_list: bool = False

    def __init__(self, t: Tensor):
        self.t = t
        # Any static dims that we are specializing. Defaults to all dynamic.
        self.spec_dims = len(t.shape) * [_NoneInt]
        # All descriptors have an attribute to indicate their value
        # as a tensor, and those that aren't are fixated to None.
        # This is to enable fast lookup in the hot path of determining
        # how to dispatch.
        self.maybe_tensor_value: Tensor = t

    def specialize_all_dims(self):
        """Marks all dimensions as specialized."""
        self.spec_dims = list(self.t.shape)

    def specialize_dims(self, *indices: int):
        """Specializes individual dimensions.

        `i` can have negative indexing.
        """
        for i in indices:
            self.spec_dims[i] = self.t.size(i)

    def __repr__(self):
        return (
            f"TensorArg(shape={self.t.shape}, dtype={self.t.dtype}, "
            f"spec_dims={self.spec_dims})"
        )

    def generate_meta(self) -> Tensor:
        t = self.t
        if t.device == "meta":
            return t
        else:
            return t.clone().detach().to("meta")

    @property
    def spec_key(self) -> str:
        """Generates a key that will be the same for all specializations."""
        t = self.t
        return f"tensor[{len(t.shape)}:{str(t.dtype)}]<{self.spec_dims}>"

    @property
    def mlir_type_asm(self) -> str:
        t = self.t
        try:
            dtype_asm = TORCH_DTYPE_TO_IREE_TYPE_ASM[t.dtype]
        except KeyError as e:
            raise KeyError(
                f"Unknown mapping of torch dtype {t.dtype} to MLIR "
                f"(possibly missing in TORCH_DTYPE_TO_IREE_TYPE_ASM table)"
            ) from e
        dim_asm = "x".join(["?" if d is None else str(d) for d in self.spec_dims])
        spec = f"{dim_asm}x{dtype_asm}" if dim_asm else dtype_asm
        return f"tensor<{spec}>"


class TensorListArg:
    __slots__ = [
        "ts",
        "spec_dims",
        "ir_arity",
        "maybe_tensor_value",
    ]

    is_list: bool = True

    def __init__(self, ts: list[Tensor]):
        self.ts = ts
        self.ir_arity = len(ts)
        # Any static dims that we are specializing. Defaults to all dynamic.
        self.spec_dims: list[list[Optional[int]]] = [len(t.shape) * [None] for t in ts]  # type: ignore
        # All descriptors have an attribute to indicate their value
        # as a tensor, and those that aren't are fixated to None.
        # This is to enable fast lookup in the hot path of determining
        # how to dispatch.
        self.maybe_tensor_value: list[Tensor] = ts

    def __repr__(self):
        return (
            f"TensorListArg(shape={[t.shape for t in self.ts]}, "
            f"dtype={[t.dtype for t in self.ts]}, "
            f"spec_dims={self.spec_dims}, ir_arity={self.ir_arity})"
        )

    def generate_meta(self) -> list[Tensor]:
        metas = []
        for t in self.ts:
            if t.device == "meta":
                metas.append(t)
            else:
                metas.append(t.clone().detach().to("meta"))
        return metas

    @property
    def spec_key(self) -> str:
        """Generates a key that will be the same for all specializations."""
        return (
            f"tensor[{[len(t.shape) for t in self.ts]}"
            f":{[str(t.dtype) for t in self.ts]}]<{self.spec_dims}>"
        )

    @property
    def mlir_type_asm(self) -> list[str]:
        asms = []
        for t, spec_dims in zip(self.ts, self.spec_dims):
            try:
                dtype_asm = TORCH_DTYPE_TO_IREE_TYPE_ASM[t.dtype]
            except KeyError as e:
                raise KeyError(
                    f"Unknown mapping of torch dtype {t.dtype} to MLIR "
                    f"(possibly missing in TORCH_DTYPE_TO_IREE_TYPE_ASM table)"
                ) from e
            dim_asm = "x".join(["?" if d is None else str(d) for d in spec_dims])
            spec = f"{dim_asm}x{dtype_asm}" if dim_asm else dtype_asm
            asms.append(f"tensor<{spec}>")
        return asms


ArgDescriptor = Union[AttrArg, IntArg, TensorArg, TensorListArg]

###############################################################################
# KernelBuilder
# Helper object for constructing IR
###############################################################################


class KernelBuilder(ABC):
    """Support class for building a kernel."""

    def __init__(
        self,
        ksel: KernelSelection,
        arg_bindings: list[Union[Value, list[Value]]],
        *,
        ip: InsertionPoint,
        module_body: Block,
        symbol_table: SymbolTable,
    ):
        self.ksel = ksel
        self.arg_bindings = arg_bindings
        self.ip = ip
        self.module_body = module_body
        self.context = module_body.owner.context
        self.symbol_table = symbol_table
        self.yielded = False

    def arg_value(self, index: int) -> Union[list[Value], Value]:
        """Gets the concrete IR `Value` for the argument at `index`.

        This will assert if the corresponding argument was set as `ir_arity=0`
        during kernel selection.
        """
        try:
            v = self.arg_bindings[index]
        except IndexError as e:
            raise AssertionError(
                f"Out of range access to kernel arg. Expected 0..{len(self.arg_bindings)}. Got {index}"
            ) from e
        assert (
            v is not None
        ), f"No `Value` is available for arg {index}: it was marked as `is_ir_arg=False` during kernel selection."
        return v

    @abstractmethod
    def yield_results(self, *results: Value):
        """Yields results of the kernel computation."""
        ...

    def constant_index(self, i: int) -> Value:
        """Builds a constant index value."""
        return arith_d.constant(IndexType.get(), IntegerAttr.get(IndexType.get(), i))


class FreeFuncKernelBuilder(KernelBuilder):
    """Kernel builder that emits the body of the kernel into a free function.

    This is intended to be used when compiling a standalone module that will
    be directly invoked by the runtime. Further variants exist that generate
    into a func but also emit a call into another local context.
    """

    def __init__(
        self,
        ksel: KernelSelection,
        *,
        module_body: Block,
        symbol_table: SymbolTable,
        func_name: Optional[str] = None,
        is_public: bool = True,
    ):
        self.module_op = module_body.owner
        context = self.module_op.context
        if func_name is None:
            func_name = ksel.op.name
        with context, Location.unknown(), InsertionPoint(module_body):
            # Assemble arg types.
            arg_types = []
            for d in ksel.arg_descs:
                assert d is not None, "NYI: None arguments"
                arity = d.ir_arity
                if not d.is_list:
                    if arity == 1:
                        arg_types.append(IrType.parse(d.mlir_type_asm))
                    else:
                        continue
                else:
                    for i in range(arity):
                        arg_types.append(IrType.parse(d.mlir_type_asm[i]))

            # Assemble result types.
            result_types = []
            for d in (*ksel.result_descs, *ksel.inplace_tied_arg_descs):
                if not d.is_list:
                    if d.ir_arity == 1:
                        result_types.append(IrType.parse(d.mlir_type_asm))
                    else:
                        continue
                else:
                    raise AssertionError("NYI: arity > 1 results")

            # Create the func.
            ftype = FunctionType.get(arg_types, result_types)
            func_op = func_d.FuncOp(func_name, ftype)
            if not is_public:
                func_op.attributes["sym_visibility"] = StringAttr.get("private")
            entry_block: Block = func_op.add_entry_block()
            symbol_table.insert(func_op)

        # Map inputs to arg bindings, lining up with arguments that are elided.
        block_arguments = list(entry_block.arguments)
        block_arg_index = 0
        arg_bindings: list[Optional[Value]] = []
        for desc in ksel.arg_descs:
            assert desc is not None, "NYI: None arguments"
            arity = desc.ir_arity
            if not desc.is_list:
                if arity == 1:
                    arg_bindings.append(block_arguments[block_arg_index])
                    block_arg_index += 1
                else:
                    arg_bindings.append(None)
            else:
                arg_bindings.append(
                    block_arguments[block_arg_index : block_arg_index + arity]
                )
                block_arg_index += arity

        super().__init__(
            ksel,
            arg_bindings,
            ip=InsertionPoint(entry_block),
            module_body=module_body,
            symbol_table=symbol_table,
        )

    @staticmethod
    def create_module(
        ksel: KernelSelection,
        *,
        context: Optional[Context] = None,
        func_name: Optional[str] = None,
        is_public: bool = True,
    ) -> "FreeFuncKernelBuilder":
        """Short-cut to create a new module with a single function in one shot."""
        if context is None:
            context = Context()
        with context, Location.unknown():
            module_op = builtin_d.ModuleOp()
            return FreeFuncKernelBuilder(
                ksel,
                module_body=module_op.body,
                symbol_table=SymbolTable(module_op),
                func_name=func_name,
                is_public=is_public,
            )

    def yield_results(self, *results: Value):
        """Yields results of the kernel computation."""
        assert not self.yielded, "yield_results has already been called"
        ksel = self.ksel
        expected_count = len(ksel.result_descs) + len(ksel.inplace_tied_arg_descs)
        assert (
            len(results) == expected_count
        ), f"Mismatched yielded results and declared+inplace: Expected={expected_count}, Got={len(results)}"
        with self.ip, Location.unknown():
            func_d.ReturnOp(results)
        self.yielded = True


###############################################################################
# Private utilities
###############################################################################


def _get_library_op(library: torch.library.Library, name: str) -> Any:
    ns = getattr(torch.ops, library.ns)
    return getattr(ns, name)


def _get_meta_impl(op: CustomOp):
    def meta(*args):
        sel = EagerKernelSelection(op, args)
        op.select(sel)
        if logger.isEnabledFor(logging.DEBUG):
            logging.debug(
                "Meta dispatch on %s for specialization %s", op.name, sel.spec_key
            )
        return sel.generate_meta_returns()

    return meta


def _create_impl_trampoline(op: CustomOp):
    # Import lazily when an implementation trampoline is requested to avoid
    # circular dependency between base objects and eager runtime goo.
    from .eager import (
        eager_dispatch,
    )

    def handler(*args):
        eager_override = op.eager_execute(*args)
        if eager_override is not NotImplemented:
            return eager_override

        ksel = EagerKernelSelection(op, args)
        op.select(ksel)
        if logger.isEnabledFor(logging.DEBUG):
            logging.debug(
                "Dispatch on %s for specialization %s", op.name, ksel.spec_key
            )
        return eager_dispatch(ksel)

    return handler


def _define_signature_in_library(lib: torch.library.Library, signature: str) -> str:
    """Helper to define a schema in the library.

    This handles the interlocked process of uniqueing, reserving the name,
    and calling `lib.define` on the resulting schema.
    """
    ns = lib.ns
    with _CONFIG_LOCK:
        name, call_args = _split_signature(signature)

        # Unique the name.
        if "@UNIQUE@" in name:
            # Uniqueify.
            unique_key = (ns, name)
            counter = UNIQUE_OP_NAME_COUNTER.get(unique_key, 0)
            counter += 1
            name = name.replace("@UNIQUE@", str(counter))
            UNIQUE_OP_NAME_COUNTER[unique_key] = counter

        # Define it, recording in the defined op names.
        key = (lib.ns, name)
        schema = f"{name}{call_args}"
        if key in DEFINED_OP_NAMES:
            raise RuntimeError(
                f"Duplicate turbine custom op registration: library={lib.ns}, "
                f"name={name}"
            )
        lib.define(schema)
        DEFINED_OP_NAMES.add(key)
    return name


_SIGNATURE_NAME_PATTERN = re.compile(r"^([^(]+)(\(.+)$")


def _split_signature(sig: str) -> tuple[str, str]:
    """Splits a signature into name and call-args parts."""
    m = re.match(_SIGNATURE_NAME_PATTERN, sig)
    if not m:
        raise ValueError(f"Expected signature of form `name(...) -> type. Got: {sig}")
    return m.group(1), m.group(2)
