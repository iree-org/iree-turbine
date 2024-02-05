builtin.module {

func.func @forward(%arg0: !torch.vtensor<[97,8],f32>) -> !torch.vtensor<[97,8],f32> {
  %0 = torch.operator "torch.expand_custom_op_pass_test.identity_tensor"(%arg0) : (!torch.vtensor<[97,8],f32>) -> (!torch.vtensor<[97,8],f32>)
  return %0 : !torch.vtensor<[97,8],f32>
}

}
