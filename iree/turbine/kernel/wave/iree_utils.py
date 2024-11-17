# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch
from typing import Any
from .utils import compile_and_invoke
from ...support.conversions import TORCH_DTYPE_TO_MLIR_TYPE_ASM


def get_chain_mmt_asm(
    query_type: str, key_type: str, value_type: str, output_type: str
) -> str:
    B, M, K1, input_dtype = query_type.split("x")
    B, K2, K1, input_dtype = key_type.split("x")
    B, N, K2, input_dtype = value_type.split("x")
    B, N, M, output_dtype = output_type.split("x")
    intermediate_output_type = f"{B}x{K2}x{M}x{output_dtype}"
    intermediate_cast_type = f"{B}x{K2}x{M}x{input_dtype}"
    transposed_cast_type = f"{B}x{M}x{K2}x{input_dtype}"
    transposed_output_type = f"{B}x{M}x{N}x{output_dtype}"
    return f"""
    func.func @chain_mmt(%query: tensor<{query_type}>, %key: tensor<{key_type}>, %value: tensor<{value_type}>) -> tensor<{output_type}> {{
      %c0 = arith.constant 0.0 : f32
      %init = tensor.empty() : tensor<{intermediate_output_type}>
      %inital_result = linalg.fill ins(%c0 : f32) outs(%init : tensor<{intermediate_output_type}>) -> tensor<{intermediate_output_type}>
      %result = linalg.batch_matmul_transpose_b ins(%key, %query : tensor<{key_type}>, tensor<{query_type}>)
                outs(%inital_result : tensor<{intermediate_output_type}>) -> tensor<{intermediate_output_type}>
      %trunc = arith.truncf %result : tensor<{intermediate_output_type}> to tensor<{intermediate_cast_type}>
      %init2 = tensor.empty() : tensor<{transposed_cast_type}>
      %transpose = linalg.transpose ins(%trunc: tensor<{intermediate_cast_type}>) outs(%init2: tensor<{transposed_cast_type}>) permutation=[0, 2, 1]
      %init3 = tensor.empty() : tensor<{transposed_output_type}>
      %inital_result3 = linalg.fill ins(%c0 : f32) outs(%init3 : tensor<{transposed_output_type}>) -> tensor<{transposed_output_type}>
      %result2 = linalg.batch_matmul_transpose_b ins(%transpose, %value: tensor<{transposed_cast_type}>, tensor<{value_type}>)
                outs(%inital_result3 : tensor<{transposed_output_type}>) -> tensor<{transposed_output_type}>
      %init4 = tensor.empty() : tensor<{output_type}>
      %transpose2 = linalg.transpose ins(%result2: tensor<{transposed_output_type}>) outs(%init4: tensor<{output_type}>) permutation=[0, 2, 1]
      return %transpose2 : tensor<{output_type}>
    }}"""


def get_chain_mmt_f8_asm(
    query_type: str, key_type: str, value_type: str, output_type: str
) -> str:
    B, M, K1, input_dtype = query_type.split("x")
    B, K2, K1, input_dtype = key_type.split("x")
    B, N, K2, input_dtype = value_type.split("x")
    B, N, M, output_dtype = output_type.split("x")
    f8_dtype = "f8E4M3FNUZ"
    intermediate_output_type = f"{B}x{K2}x{M}x{output_dtype}"
    intermediate_cast_type = f"{B}x{K2}x{M}x{f8_dtype}"
    transposed_cast_type = f"{B}x{M}x{K2}x{f8_dtype}"
    transposed_output_type = f"{B}x{M}x{N}x{output_dtype}"
    query_f8_type = "x".join([B, M, K1, f8_dtype])
    key_f8_type = "x".join([B, K2, K1, f8_dtype])
    value_f8_type = "x".join([B, N, K2, f8_dtype])
    return f"""
    func.func @chain_mmt_f8(%query: tensor<{query_type}>, %key: tensor<{key_type}>, %value: tensor<{value_type}>) -> tensor<{output_type}> {{
      %c0 = arith.constant 0.0 : f32
      %init = tensor.empty() : tensor<{intermediate_output_type}>
      %query_f8 = arith.truncf %query : tensor<{query_type}> to tensor<{query_f8_type}>
      %key_f8 = arith.truncf %key : tensor<{key_type}> to tensor<{key_f8_type}>
      %inital_result = linalg.fill ins(%c0 : f32) outs(%init : tensor<{intermediate_output_type}>) -> tensor<{intermediate_output_type}>
      %result = linalg.batch_matmul_transpose_b ins(%key_f8, %query_f8 : tensor<{key_f8_type}>, tensor<{query_f8_type}>)
                outs(%inital_result : tensor<{intermediate_output_type}>) -> tensor<{intermediate_output_type}>
      %trunc = arith.truncf %result : tensor<{intermediate_output_type}> to tensor<{intermediate_cast_type}>
      %init2 = tensor.empty() : tensor<{transposed_cast_type}>
      %transpose = linalg.transpose ins(%trunc: tensor<{intermediate_cast_type}>) outs(%init2: tensor<{transposed_cast_type}>) permutation=[0, 2, 1]
      %init3 = tensor.empty() : tensor<{transposed_output_type}>
      %inital_result3 = linalg.fill ins(%c0 : f32) outs(%init3 : tensor<{transposed_output_type}>) -> tensor<{transposed_output_type}>
      %value_f8 = arith.truncf %value : tensor<{value_type}> to tensor<{value_f8_type}>
      %result2 = linalg.batch_matmul_transpose_b ins(%transpose, %value_f8: tensor<{transposed_cast_type}>, tensor<{value_f8_type}>)
                outs(%inital_result3 : tensor<{transposed_output_type}>) -> tensor<{transposed_output_type}>
      %init4 = tensor.empty() : tensor<{output_type}>
      %transpose2 = linalg.transpose ins(%result2: tensor<{transposed_output_type}>) outs(%init4: tensor<{output_type}>) permutation=[0, 2, 1]
      return %transpose2 : tensor<{output_type}>
    }}"""


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
    elif kernel_type == "chain_mmt":
        query_type = get_type_str(kernel_inputs[0].shape, kernel_inputs[0].dtype)
        key_type = get_type_str(kernel_inputs[1].shape, kernel_inputs[1].dtype)
        value_type = get_type_str(kernel_inputs[2].shape, kernel_inputs[2].dtype)
        output_type = get_type_str(kernel_outputs[0].shape, kernel_outputs[0].dtype)
        asm = get_chain_mmt_asm(query_type, key_type, value_type, output_type)
    elif kernel_type == "chain_mmt_f8":
        query_type = get_type_str(kernel_inputs[0].shape, kernel_inputs[0].dtype)
        key_type = get_type_str(kernel_inputs[1].shape, kernel_inputs[1].dtype)
        value_type = get_type_str(kernel_inputs[2].shape, kernel_inputs[2].dtype)
        output_type = get_type_str(kernel_outputs[0].shape, kernel_outputs[0].dtype)
        asm = get_chain_mmt_f8_asm(query_type, key_type, value_type, output_type)
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
