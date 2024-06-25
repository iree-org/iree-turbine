# Copyright 2023 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Callable

import torch
from torch import Tensor
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import ShapeEnv

from ...dynamo.type_conversion import (
    NativeTypeConverter,
)

from ...runtime.op_reg.base import (
    ALL_CUSTOM_OP_REGS,
    AttrArg,
    IntArg,
    CustomOp,
    KernelBuilder,
    KernelSelection,
    TensorArg,
    TensorListArg,
)

from ...support.conversions import (
    MLIR_TYPE_ASM_TO_TORCH_DTYPE,
)

from ...support.ir_imports import (
    Block,
    FloatAttr,
    IrType,
    InsertionPoint,
    IntegerAttr,
    OpResult,
    Operation,
    RankedTensorType,
    StringAttr,
    SymbolTable,
    Value,
)

from ..rewriter import (
    Pass,
)


class ExpandCustomOpsPass(Pass):
    def __init__(
        self, root_op: Operation, reg: dict[str, CustomOp] = ALL_CUSTOM_OP_REGS
    ):
        super().__init__(root_op)
        self.reg = reg
        # Track pending deletions in a dict to preserve order and unique.
        self.ops_to_delete: dict[Operation, None] = {}
        self.type_converter = NativeTypeConverter(root_op.context)
        self.symbol_table = SymbolTable(root_op)
        self.shape_env = ShapeEnv()
        self.fake_mode = FakeTensorMode(shape_env=self.shape_env)

    def delete_op(self, op):
        self.ops_to_delete[op.operation] = None

    def run(self):
        for mr in self.funcs:
            self.expand_func(mr.op)
        for op in self.ops_to_delete.keys():
            self.erase_unused_op(op)

    def expand_func(self, func_op: Operation):
        """Expands custom ops in a traced torch function.

        This finds operations of the form:
        %0 = torch.operator "torch.ns.op"

        And looks them up in the reg dict. If it originated from one of those
        registered ops, then it will be expanded in place.
        """
        name_prefix = "torch."

        for block in func_op.regions[0].blocks:
            for op in block.operations:
                if op.name == "torch.operator":
                    custom_op_name = StringAttr(op.attributes["name"]).value
                    if custom_op_name.startswith(name_prefix):
                        local_name = custom_op_name[len(name_prefix) :]
                        custom_op_reg = self.reg.get(local_name)
                        if custom_op_reg is not None:
                            self.expand_custom_op(custom_op_reg, op)

    def expand_custom_op(self, op_reg: CustomOp, op: Operation):
        original_operands: list[Value] = list(op.operands)
        ksel = AOTKernelSelection(
            op_reg,
            original_operands,
            list(op.results),
            self.type_converter,
            self.shape_env,
        )
        with self.fake_mode:
            op_reg.select(ksel)
        ksel._run_validators()

        module_body = self.root_op.regions[0].blocks[0]
        kb = InlineKernelBuilder(
            ksel,
            op,
            type_converter=self.type_converter,
            module_body=module_body,
            symbol_table=self.symbol_table,
        )
        with kb.ip, kb.location:
            op_reg.generate(ksel, kb)
        assert kb.yielded, "Custom op generation did not yield_results()"

        self.delete_op(op)


class AOTKernelSelection(KernelSelection):
    __slots__ = [
        "operands",
        "results",
        "type_converter",
        "shape_env",
        "_validators",
    ]

    def __init__(
        self,
        op: CustomOp,
        operands: list[Value],
        results: list[Value],
        type_converter: NativeTypeConverter,
        shape_env: ShapeEnv,
    ):
        super().__init__(op, len(operands))
        self.operands = operands
        self.results = results
        self.type_converter = type_converter
        self.shape_env = shape_env
        self._validators: list[Callable] = []

    def _run_validators(self):
        for v in self._validators:
            v()

    def arg_tensor(self, arg: int, *, inplace_tied: bool = False) -> TensorArg:
        # This is annoying: We have to go from the Torch MLIR type system to the
        # original torch.tensor Python type system. We do this by way of the native
        # type converter because it has the mapping pathway we need. This is one of the
        # only places in the code where we have to go this way to preserve the facade.
        # Everywhere else is going from Torch -> IREE native.
        arg_descs = self.arg_descs
        assert arg_descs[arg] is None, f"Already constrained argument {arg}"
        operand = self.operands[arg]
        signed_native_type = self.type_converter.torch_type_to_native(
            operand.type, signless=False
        )
        try:
            rtt = RankedTensorType(signed_native_type)
        except TypeError as e:
            raise TypeError(
                f"Argument type mismatch from Torch IR for arg {arg}: Expected ranked tensor, got {signed_native_type}"
            ) from e
        element_type_asm = str(rtt.element_type)
        try:
            dtype = MLIR_TYPE_ASM_TO_TORCH_DTYPE[element_type_asm]
        except KeyError as e:
            raise AssertionError(
                f"Could not find dtype mapping for {element_type_asm} in MLIR_TYPE_ASM_TO_TORCH_DTYPE"
            )

        # Because we are operating in fake_mode, replace MLIR dyn dims with
        # symints for the PyTorch type system.
        shape_env = self.shape_env
        sym_shape = [
            d if d >= 0 else shape_env.create_unbacked_symint() for d in rtt.shape
        ]
        t = torch.empty(sym_shape, dtype=dtype)
        arg_descs[arg] = desc = TensorArg(t)
        if inplace_tied:
            self.inplace_tied_arg_descs.append(desc)

        def validator():
            rank = rtt.rank
            for i in range(rank):
                spec_dim = desc.spec_dims[i]
                if rtt.is_dynamic_dim(i):
                    # Make sure that it wasn't specialized.
                    if spec_dim is not None:
                        raise ValueError(
                            f"Custom op {self.op}, arg {arg} requires a static dim "
                            f"at index {i} but it is dynamic: {rtt}"
                        )
                else:
                    # Make sure specialized dim matches.
                    actual_dim = rtt.get_dim_size(i)
                    if spec_dim is not None and actual_dim != spec_dim:
                        raise ValueError(
                            f"Custom op {self.op}, arg {arg} has a mismatched static "
                            f"dim at index {i}: actual = {actual_dim}, expected = {spec_dim}"
                        )

        self._validators.append(validator)
        return desc

    def arg_tensor_list(self, arg: int) -> TensorListArg:
        raise NotImplementedError("NYI: AOT arg_tensor_list")

    def arg_int(self, arg: int) -> IntArg:
        raise NotImplementedError("NYI: AOT arg_int")

    def attr_str(self, arg: int) -> AttrArg:
        arg_descs = self.arg_descs
        assert arg_descs[arg] is None, f"Already constrained argument {arg}"
        operand = self.operands[arg]
        ty = operand.type
        assert (
            str(ty) == "!torch.str"
        ), f"Argument type mismatch from Torch IR for {arg}: Expected !torch.str, got {ty}"
        str_value = _get_constant_str_from_value(operand)
        arg_descs[arg] = desc = AttrArg(str_value)
        return desc

    def attr_int(self, arg: int) -> AttrArg:
        arg_descs = self.arg_descs
        assert arg_descs[arg] is None, f"Already constrained argument {arg}"
        operand = self.operands[arg]
        ty = operand.type
        assert (
            str(ty) == "!torch.int"
        ), f"Argument type mismatch from Torch IR for {arg}: Expected !torch.int, got {ty}"
        int_value = _get_constant_int_from_value(operand)
        arg_descs[arg] = desc = AttrArg(int_value)
        return desc

    def attr_list_int(self, arg: int) -> AttrArg:
        arg_descs = self.arg_descs
        assert arg_descs[arg] is None, f"Already constrained argument {arg}"
        operand = self.operands[arg]
        ty = operand.type
        assert (
            str(ty) == "!torch.list<int>"
        ), f"Argument type mismatch from Torch IR for {arg}: Expected !torch.list<int>, got {ty}"
        list_value = _get_constant_list_int_from_value(operand)
        arg_descs[arg] = desc = AttrArg(list_value)
        return desc

    def attr_float(self, arg: int) -> AttrArg:
        arg_descs = self.arg_descs
        assert arg_descs[arg] is None, f"Already constrained argument {arg}"
        operand = self.operands[arg]
        ty = operand.type
        assert (
            str(ty) == "!torch.float"
        ), f"Argument type mismatch from Torch IR for {arg}: Expected !torch.float, got {ty}"
        int_value = _get_constant_float_from_value(operand)
        arg_descs[arg] = desc = AttrArg(int_value)
        return desc

    def attr_list_float(self, arg: int) -> AttrArg:
        arg_descs = self.arg_descs
        assert arg_descs[arg] is None, f"Already constrained argument {arg}"
        operand = self.operands[arg]
        ty = operand.type
        assert (
            str(ty) == "!torch.list<float>"
        ), f"Argument type mismatch from Torch IR for {arg}: Expected !torch.list<float>, got {ty}"
        list_value = _get_constant_list_float_from_value(operand)
        arg_descs[arg] = desc = AttrArg(list_value)
        return desc

    def return_tensor(self, t: Tensor) -> TensorArg:
        desc = TensorArg(t)
        self.result_descs.append(desc)
        return desc


def _get_constant_str_from_value(v: Value) -> str:
    """Given a constant str producer, return the str.

    Example: %str = torch.constant.str "TEST"
    """
    constant_op = OpResult(v).owner
    assert (
        constant_op.name == "torch.constant.str"
    ), f"Expected constant !torch.str to be produced by a torch.constant.str op but got: {constant_op}"
    return StringAttr(constant_op.attributes["value"]).value


def _get_constant_int_from_value(v: Value) -> int:
    """Given a constant int producer, return the int.

    Example: %int = torch.constant.int 1
    """
    constant_op = OpResult(v).owner
    assert (
        constant_op.name == "torch.constant.int"
    ), f"Expected constant !torch.int to be produced by a torch.constant.int op but got: {constant_op}"
    return IntegerAttr(constant_op.attributes["value"]).value


def _get_constant_list_int_from_value(v: Value) -> list[int]:
    """Given a constant int list producer, return the list."""
    constant_op = OpResult(v).owner
    assert (
        constant_op.name == "torch.prim.ListConstruct"
    ), f"Expected constant !torch.list<int> to be produced by a torch.prim.ListConstruct op but got: {constant_op}"
    lst = []
    for i in constant_op.operands:
        v = _get_constant_int_from_value(i)
        lst.append(v)
    return lst


def _get_constant_float_from_value(v: Value) -> float:
    """Given a constant float producer, return the float.

    Example: %int = torch.constant.int 1
    """
    constant_op = OpResult(v).owner
    assert (
        constant_op.name == "torch.constant.float"
    ), f"Expected constant !torch.float to be produced by a torch.constant.float op but got: {constant_op}"
    return FloatAttr(constant_op.attributes["value"]).value


def _get_constant_list_float_from_value(v: Value) -> list[float]:
    """Given a constant float list producer, return the list."""
    constant_op = OpResult(v).owner
    assert (
        constant_op.name == "torch.prim.ListConstruct"
    ), f"Expected constant !torch.list<float> to be produced by a torch.prim.ListConstruct op but got: {constant_op}"
    lst = []
    for i in constant_op.operands:
        v = _get_constant_int_from_value(i)
        lst.append(v)
    return lst


class InlineKernelBuilder(KernelBuilder):
    def __init__(
        self,
        ksel: KernelSelection,
        torch_op: Operation,
        *,
        type_converter: NativeTypeConverter,
        module_body: Block,
        symbol_table: SymbolTable,
    ):
        location = torch_op.location
        ip = InsertionPoint(torch_op)
        with ip, location:
            operands = list(torch_op.operands)
            arg_bindings = []
            for desc, operand in zip(ksel.arg_descs, operands):
                assert desc is not None, "NYI: None arguments"
                arity = desc.ir_arity
                if not desc.is_list:
                    if arity == 1:
                        arg_bindings.append(
                            type_converter.materialize_torch_to_native(
                                operand,
                                static_info_cast_to=IrType.parse(desc.mlir_type_asm),
                            )
                        )
                    else:
                        arg_bindings.append(None)
                else:
                    # arg_bindings.extend(native_operands)
                    raise NotImplementedError("NYI: AOT custom op list arguments")

        super().__init__(
            ksel,
            arg_bindings=arg_bindings,
            ip=ip,
            module_body=module_body,
            symbol_table=symbol_table,
        )
        self.location = location
        self.torch_op = torch_op
        self.type_converter = type_converter

    def yield_results(self, *results: Value):
        """Yields results of the kernel computation."""
        assert not self.yielded, "yield_results has already been called"
        ksel = self.ksel
        expected_count = len(ksel.result_descs) + len(ksel.inplace_tied_arg_descs)
        assert (
            len(results) == expected_count
        ), f"Mismatched yielded results and declared+inplace: Expected={expected_count}, Got={len(results)}"
        with self.ip, self.location:
            torch_op_results: list[Value] = list(self.torch_op.results)
            assert len(results) == len(
                torch_op_results
            ), f"Mismatched yield_results with custom op results"
            for new_result, old_result in zip(results, torch_op_results):
                torch_type = old_result.type
                new_result = self.type_converter.materialize_native_to_torch(
                    new_result,
                    torch_type,
                    static_info_cast=True,
                )
                old_result.replace_all_uses_with(new_result)
        self.yielded = True
