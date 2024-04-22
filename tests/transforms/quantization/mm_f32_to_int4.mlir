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
