# Copyright 2023 Nod Labs, Inc
# Portions Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
from iree.compiler.ir import (
    Context,
    Operation,
)

from iree.turbine.transforms.quantization import mm_group_quant

MM_F32_TO_INT4_DYNAMIC_M = r"""
module @state_update {
  util.global private @_params.model.layers.0.self_attn.q_proj.weight {noinline} : tensor<4096x4096xf32>
  func.func @initialize(%arg0: !torch.vtensor<[?,4096],f32>) -> (!torch.vtensor<[?,4096],f32>) {
    %_params.model.layers.0.self_attn.q_proj.weight = util.global.load @_params.model.layers.0.self_attn.q_proj.weight : tensor<4096x4096xf32>
    %55 = torch_c.from_builtin_tensor %_params.model.layers.0.self_attn.q_proj.weight : tensor<4096x4096xf32> -> !torch.vtensor<[4096,4096],f32>
    %int0_74 = torch.constant.int 0
    %int1_75 = torch.constant.int 1
    %56 = torch.aten.transpose.int %55, %int0_74, %int1_75 : !torch.vtensor<[4096,4096],f32>, !torch.int, !torch.int -> !torch.vtensor<[4096,4096],f32>
    %59 = torch.aten.mm %arg0, %56 : !torch.vtensor<[?,4096],f32>, !torch.vtensor<[4096,4096],f32> -> !torch.vtensor<[?,4096],f32>
    return %59 : !torch.vtensor<[?,4096],f32>
  }
}
"""

MM_F32_TO_INT4_STATIC_M = r"""
module @state_update {
  util.global private @_params.model.layers.0.self_attn.q_proj.weight {noinline} : tensor<4096x4096xf32>
  func.func @initialize(%arg0: !torch.vtensor<[32,4096],f32>) -> (!torch.vtensor<[32,4096],f32>) {
    %_params.model.layers.0.self_attn.q_proj.weight = util.global.load @_params.model.layers.0.self_attn.q_proj.weight : tensor<4096x4096xf32>
    %55 = torch_c.from_builtin_tensor %_params.model.layers.0.self_attn.q_proj.weight : tensor<4096x4096xf32> -> !torch.vtensor<[4096,4096],f32>
    %int0_74 = torch.constant.int 0
    %int1_75 = torch.constant.int 1
    %56 = torch.aten.transpose.int %55, %int0_74, %int1_75 : !torch.vtensor<[4096,4096],f32>, !torch.int, !torch.int -> !torch.vtensor<[4096,4096],f32>
    %59 = torch.aten.mm %arg0, %56 : !torch.vtensor<[32,4096],f32>, !torch.vtensor<[4096,4096],f32> -> !torch.vtensor<[32,4096],f32>
    return %59 : !torch.vtensor<[32,4096],f32>
  }
}
"""


@pytest.mark.parametrize(
    "contents",
    [
        pytest.param(MM_F32_TO_INT4_DYNAMIC_M, id="dynamic_m"),
        pytest.param(MM_F32_TO_INT4_STATIC_M, id="static_m"),
    ],
)
def test_group_quant(contents):
    with Context():
        module_op = Operation.parse(contents)
        mm_group_quant.MMGroupQuantRewriterPass(module_op).run()
        module_asm = str(module_op)
        assert "torch.aten.mm" not in module_asm
        assert "@_params.model.layers.0.self_attn.q_proj.weight " not in module_asm
        assert "linalg.generic" in module_asm
