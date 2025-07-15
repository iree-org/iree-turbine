builtin.module {

func.func @forward(%arg0: !torch.vtensor<[2,4],f32>, %arg1: !torch.vtensor<[4,3],f32>) -> !torch.vtensor<[2,3],f32> attributes {torch.assume_strict_symbolic_shapes} {
  %none = torch.constant.none
  %0 = torch.operator "torch._turbine_jinja_test.test_linear_middle_optional"(%arg0, %none, %arg1) : (!torch.vtensor<[2,4],f32>, !torch.none, !torch.vtensor<[4,3],f32>) -> !torch.vtensor<[2,3],f32>
  return %0 : !torch.vtensor<[2,3],f32>
}

}
