!tensor_type = {{tensor_type}}

module {

util.func private @turbine_test_add_jinja_{{rank}}d_{{element_type}}(
    %a: !tensor_type, %b: !tensor_type
) -> !tensor_type {
  %out = tensor.empty() : !tensor_type
  %0 = linalg.add ins(%a, %b : !tensor_type, !tensor_type) outs(%out : !tensor_type) -> !tensor_type
  util.return %0 : !tensor_type
}
}
