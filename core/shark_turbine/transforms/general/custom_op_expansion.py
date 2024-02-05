# Copyright 2023 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch
from torch import Tensor

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
    InsertionPoint,
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
            op_reg, original_operands, list(op.results), self.type_converter
        )
        op_reg.select(ksel)

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
    ]

    def __init__(
        self,
        op: CustomOp,
        operands: list[Value],
        results: list[Value],
        type_converter: NativeTypeConverter,
    ):
        super().__init__(op, len(operands))
        self.operands = operands
        self.results = results
        self.type_converter = type_converter

    def arg_tensor(self, arg: int) -> TensorArg:
        # This is annoying: We have to go from the Torch MLIR type system to the
        # original torch.tensor Python type system. We do this by way of the native
        # type converter because it has the mapping pathway we need. This is one of the
        # only places in the code where we have to go this way to preserve the facade.
        # Everywhere else is going from Torch -> IREE native.
        arg_descs = self.arg_descs
        assert arg_descs[arg] is None, f"Already constrained argument {arg}"
        operand = self.operands[arg]
        signed_native_type = self.type_converter.torch_type_to_native(operand.type)
        try:
            rtt = RankedTensorType(signed_native_type)
            # TODO: We need to do the FakeMode/ShapeEnv dance to create a symbolic
            # fake tensor here.
        except TypeError as e:
            raise TypeError(
                f"Argument type mismatch from Torch IR for arg {arg}: Expected ranked tensor, got {signed_native_type}"
            ) from e
        assert not any(
            rtt.is_dynamic_dim(i) for i in range(rtt.rank)
        ), "NYI: Dynamic shape tensors in custom op AOT mode"
        element_type_asm = str(rtt.element_type)
        try:
            dtype = MLIR_TYPE_ASM_TO_TORCH_DTYPE[element_type_asm]
        except KeyError as e:
            raise AssertionError(
                f"Could not find dtype mapping for {element_type_asm} in MLIR_TYPE_ASM_TO_TORCH_DTYPE"
            )
        t = torch.empty(rtt.shape, dtype=dtype, device="meta")
        arg_descs[arg] = desc = TensorArg(t)
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
                arity = desc.ir_arity
                if not desc.is_list:
                    if arity == 1:
                        arg_bindings.append(
                            type_converter.materialize_torch_to_native(operand)
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
        with self.ip, self.location:
            torch_op_results: list[Value] = list(self.torch_op.results)
            assert len(results) == len(
                torch_op_results
            ), f"Mismatched yield_results with custom op results"
            for new_result, old_result in zip(results, torch_op_results):
                torch_type = old_result.type
                new_result = self.type_converter.materialize_native_to_torch(
                    new_result, torch_type
                )
                old_result.replace_all_uses_with(new_result)
        self.yielded = True
