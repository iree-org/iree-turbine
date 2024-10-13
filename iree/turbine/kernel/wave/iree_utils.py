# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch
from typing import Any
from .utils import compile_and_invoke
from ...support.conversions import TORCH_DTYPE_TO_MLIR_TYPE_ASM


def get_mmt_asm(
    lhs_type: str, rhs_type: str, acc_type: str, batch: bool = False
) -> str:
    acc_dtype = acc_type.split("x")[-1]
    operator = "batch_matmul_transpose_b" if batch else "matmul_transpose_b"
    matmul_function = f"""
    func.func @mmt(%lhs: tensor<{lhs_type}>, %rhs: tensor<{rhs_type}>) -> tensor<{acc_type}> {{
      %c0 = arith.constant 0.0 : {acc_dtype}
      %init = tensor.empty() : tensor<{acc_type}>
      %inital_result = linalg.fill ins(%c0 : {acc_dtype}) outs(%init : tensor<{acc_type}>) -> tensor<{acc_type}>
      %result = linalg.{operator} ins(%lhs, %rhs: tensor<{lhs_type}>, tensor<{rhs_type}>)
                 outs(%inital_result: tensor<{acc_type}>) -> tensor<{acc_type}>
      return %result : tensor<{acc_type}>
    }}"""
    return matmul_function


def get_conv_asm(
    conv_type: str, lhs_type: str, rhs_type: str, res_type: str, stride: int
) -> str:
    res_dtype = res_type.split("x")[-1]
    return f"""
    func.func @conv_{conv_type}(%lhs: tensor<{lhs_type}>, %rhs: tensor<{rhs_type}>) -> tensor<{res_type}> {{
      %c0 = arith.constant 0.0 : {res_dtype}
      %init = tensor.empty() : tensor<{res_type}>
      %inital_result = linalg.fill ins(%c0 : {res_dtype}) outs(%init : tensor<{res_type}>) -> tensor<{res_type}>
      %result = linalg.conv_{conv_type}
                {{dilations = dense<1> : tensor<2xi64>, strides = dense<{stride}> : tensor<2xi64>}}
                ins(%lhs, %rhs : tensor<{lhs_type}>, tensor<{rhs_type}>)
                outs(%inital_result : tensor<{res_type}>) -> tensor<{res_type}>
      return %result : tensor<{res_type}>
    }}"""


def dtype_str(dtype: torch.dtype) -> str:
    dtype_str = TORCH_DTYPE_TO_MLIR_TYPE_ASM.get(dtype, None)
    if dtype_str is None:
        raise ValueError(f"Unsupported dtype: {dtype}")
    return dtype_str


def get_type_str(shape: tuple[int], dtype: torch.dtype) -> str:
    return "x".join([str(x) for x in shape] + [dtype_str(dtype)])


def generate_iree_ref(
    kernel_type: str,
    kernel_inputs: list[torch.Tensor],
    kernel_outputs: list[torch.Tensor],
    config: dict[str, str],
    **kwargs: dict[str, Any],
):
    """
    Generate a reference output for the given kernel type and arguments.
    """

    asm = None
    conv_str = "conv_"
    if kernel_type == "mmt":
        lhs_type = get_type_str(kernel_inputs[0].shape, kernel_inputs[0].dtype)
        rhs_type = get_type_str(kernel_inputs[1].shape, kernel_inputs[1].dtype)
        acc_type = get_type_str(kernel_outputs[0].shape, kernel_outputs[0].dtype)
        asm = get_mmt_asm(lhs_type, rhs_type, acc_type)
    elif kernel_type == "bmmt":
        lhs_type = get_type_str(kernel_inputs[0].shape, kernel_inputs[0].dtype)
        rhs_type = get_type_str(kernel_inputs[1].shape, kernel_inputs[1].dtype)
        acc_type = get_type_str(kernel_outputs[0].shape, kernel_outputs[0].dtype)
        asm = get_mmt_asm(lhs_type, rhs_type, acc_type, batch=True)
    elif kernel_type.startswith(conv_str):
        lhs_type = get_type_str(kernel_inputs[0].shape, kernel_inputs[0].dtype)
        rhs_type = get_type_str(kernel_inputs[1].shape, kernel_inputs[1].dtype)
        acc_type = get_type_str(kernel_outputs[0].shape, kernel_outputs[0].dtype)
        conv_type = kernel_type[len(conv_str) :]
        asm = get_conv_asm(
            conv_type, lhs_type, rhs_type, acc_type, int(kwargs["stride"])
        )
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")

    compile_and_invoke(
        asm,
        kernel_type,
        config,
        kernel_inputs,
        kernel_outputs,
        run=True,
        run_bench=kwargs.get("run_bench", False),
    )
