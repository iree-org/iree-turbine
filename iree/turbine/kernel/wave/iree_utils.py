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
    lhs_type: str,
    rhs_type: str,
    acc_type: str,
    batch: bool = False,
    cast_fp8: bool = False,
) -> str:
    acc_dtype = acc_type.split("x")[-1]
    operator = "batch_matmul_transpose_b" if batch else "matmul_transpose_b"
    func_name = "bmmt" if batch else "mmt"
    func_name = func_name + "_f8" if cast_fp8 else func_name
    if not cast_fp8:
        matmul_function = f"""
        func.func @{func_name}(%lhs: tensor<{lhs_type}>, %rhs: tensor<{rhs_type}>) -> tensor<{acc_type}> {{
          %c0 = arith.constant 0.0 : {acc_dtype}
          %init = tensor.empty() : tensor<{acc_type}>
          %inital_result = linalg.fill ins(%c0 : {acc_dtype}) outs(%init : tensor<{acc_type}>) -> tensor<{acc_type}>
          %result = linalg.{operator} ins(%lhs, %rhs: tensor<{lhs_type}>, tensor<{rhs_type}>)
                     outs(%inital_result: tensor<{acc_type}>) -> tensor<{acc_type}>
          return %result : tensor<{acc_type}>
        }}"""
    else:
        dtype = lhs_type.split("x")[-1]
        f8_dtype = "f8E4M3FNUZ"
        lhs_type_f8 = lhs_type.replace(dtype, f8_dtype)
        dtype = rhs_type.split("x")[-1]
        rhs_type_f8 = rhs_type.replace(dtype, f8_dtype)
        matmul_function = f"""
        func.func @{func_name}(%lhs: tensor<{lhs_type}>, %rhs: tensor<{rhs_type}>) -> tensor<{acc_type}> {{
          %c0 = arith.constant 0.0 : {acc_dtype}
          %init = tensor.empty() : tensor<{acc_type}>
          %inital_result = linalg.fill ins(%c0 : {acc_dtype}) outs(%init : tensor<{acc_type}>) -> tensor<{acc_type}>
          %lhs_f8 = arith.truncf %lhs : tensor<{lhs_type}> to tensor<{lhs_type_f8}>
          %rhs_f8 = arith.truncf %rhs : tensor<{rhs_type}> to tensor<{rhs_type_f8}>
          %result = linalg.{operator} ins(%lhs_f8, %rhs_f8: tensor<{lhs_type_f8}>, tensor<{rhs_type_f8}>)
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
    if kernel_type == "mmt" or kernel_type == "mmt_f8":
        lhs_type = get_type_str(kernel_inputs[0].shape, kernel_inputs[0].dtype)
        rhs_type = get_type_str(kernel_inputs[1].shape, kernel_inputs[1].dtype)
        acc_type = get_type_str(kernel_outputs[0].shape, kernel_outputs[0].dtype)
        asm = get_mmt_asm(
            lhs_type, rhs_type, acc_type, batch=False, cast_fp8=kernel_type == "mmt_f8"
        )
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
