// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#map0 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d1)>

!accum_type = {{accum_dtype}}
!X_asm_type = {{X_asm_type}}
!W_asm_type = {{W_asm_type}}
!result_asm_type = {{result_asm_type}}
!dynamic_result_asm_type = tensor<?x?x?x?x{{accum_dtype}}>

module {

util.func private @conv_2d_nhwc_fhwc_{{spec_sig}}
  (%input_pad: !X_asm_type, %weights: !W_asm_type)
    -> !result_asm_type {
  %zero = arith.constant 0.0: !accum_type
  %c0 = arith.constant 0: index
  %c1 = arith.constant 1: index
  %c2 = arith.constant 2: index
  %c3 = arith.constant 3: index

  %rN = tensor.dim %input_pad, %c0 : !X_asm_type
  %rC = tensor.dim %weights, %c0 : !W_asm_type
  %rDim0 = arith.constant {{OUT_DIM0}} : index
  %rDim1 = arith.constant {{OUT_DIM1}} : index
  %result_empty_dynamic = tensor.empty(%rN, %rDim0, %rDim1, %rC) : !dynamic_result_asm_type
  %result_empty = tensor.cast %result_empty_dynamic : !dynamic_result_asm_type to !result_asm_type
  %result_fill = linalg.fill ins(%zero: !accum_type) outs(%result_empty: !result_asm_type) -> !result_asm_type
  %result = linalg.conv_2d_nhwc_fhwc
    {dilations = dense<[{{D0}}, {{D1}}]> : tensor<2xi64>,
     strides = dense<[{{S0}}, {{S1}}]> : tensor<2xi64>}
    ins(%input_pad, %weights: !X_asm_type, !W_asm_type)
    outs(%result_fill: !result_asm_type) -> !result_asm_type
  util.return %result : !result_asm_type
}

}
