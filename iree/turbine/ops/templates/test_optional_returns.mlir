!tensor_type_a = {{tensor_type_a}}
!tensor_type_b = {{tensor_type_b}}

module {
util.func private @turbine_test_optional_returns_{{mask}}_{{rank_a}}d_{{rank_b}}d(
    %a: !tensor_type_a, %b: !tensor_type_b
) -> {{func_return_type}} {
  util.return {{return_string}}
}
}
