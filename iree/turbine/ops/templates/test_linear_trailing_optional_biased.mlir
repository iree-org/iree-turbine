!tensor_A_type = {{tensor_A_type}}
!tensor_B_type = {{tensor_B_type}}
!tensor_bias_type = {{tensor_bias_type}}
!tensor_Result_type = {{tensor_Result_type}}

module {

util.func private @turbine_test_linear_trailing_optional_{{rank}}d_{{element_type}}_biased(
    %A: !tensor_A_type, %B: !tensor_B_type, %bias: !tensor_bias_type
) -> !tensor_Result_type {
  %c0 = arith.constant 0.0 : {{element_type}}
  %init = tensor.empty() : !tensor_Result_type
  %filled = linalg.fill ins(%c0 : {{element_type}}) outs(%init : !tensor_Result_type) -> !tensor_Result_type
  %AB = linalg.matmul ins(%A, %B : !tensor_A_type, !tensor_B_type) outs(%filled : !tensor_Result_type) -> !tensor_Result_type

  %out = linalg.generic
    { indexing_maps = [
        affine_map<(d0, d1) -> (d1)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"] }
    ins(%bias: !tensor_bias_type)
    outs(%AB : !tensor_Result_type) {
  ^bb0(%b: f32, %c: f32):
    %sum = arith.addf %c, %b : f32
    linalg.yield %sum : f32
  } -> !tensor_Result_type

  util.return %out : !tensor_Result_type
}
}
