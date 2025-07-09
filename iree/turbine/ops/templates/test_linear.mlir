!tensor_A_type = {{tensor_A_type}}
!tensor_B_type = {{tensor_B_type}}
!tensor_Result_type = {{tensor_Result_type}}

module {

util.func private @turbine_test_linear_{{rank}}d_{{element_type}}(
    %a: !tensor_A_type, %b: !tensor_B_type
) -> !tensor_Result_type {
  %c0 = arith.constant 0.0 : {{element_type}}
  %init = tensor.empty() : !tensor_Result_type
  %out = linalg.fill ins(%c0 : {{element_type}}) outs(%init : !tensor_Result_type) -> !tensor_Result_type
  %0 = linalg.matmul ins(%a, %b : !tensor_A_type, !tensor_B_type) outs(%out : !tensor_Result_type) -> !tensor_Result_type
  util.return %0 : !tensor_Result_type
}
}
