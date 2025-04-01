!accum_type = {{accum_dtype}}
!X_dtype = {{X_dtype}}
!X_asm_type = {{X_asm_type}}
!W_dtype = {{W_dtype}}
!W_asm_type = {{W_asm_type}}
!result_asm_type = {{result_asm_type}}

module @module {

util.func private @generic_conv_{{spec_sig}}
  (%input_pad: !X_asm_type, %weights: !W_asm_type)
    -> !result_asm_type {
    %cst = arith.constant 0.000000e+00 : !accum_type
    // empty like result
    %2 = tensor.empty() : !result_asm_type
    // zero-fill like result
    %3 = linalg.fill ins(%cst : !accum_type) outs(%2: !result_asm_type) -> !result_asm_type
    %4 = linalg.generic {indexing_maps = [
       {{X_indexing_map}},
       {{W_indexing_map}},
       {{result_indexing_map}}
       ], iterator_types = {{iterator_types}} }
       ins(%input_pad, %weights : !X_asm_type, !W_asm_type)
       outs(%3 : !result_asm_type) {
    ^bb0(%in: !X_dtype, %in_0: !W_dtype, %out: !accum_type):
{% if accum_dtype == X_dtype and accum_dtype == W_dtype %}
      %mul = arith.mulf %in, %in_0 : !accum_type
{% endif %}
{% if accum_dtype != X_dtype and accum_dtype == W_dtype %}
      %x_ext = arith.extf %in : !X_dtype to !accum_type
      %mul = arith.mulf %x_ext, %in_0 : !accum_type
{% endif %}
{% if accum_dtype == X_dtype and accum_dtype != W_dtype %}
      %w_ext = arith.extf %in_0 : !W_dtype to !accum_type
      %mul = arith.mulf %in, %w_ext : !accum_type
{% endif %}
{% if accum_dtype != X_dtype and accum_dtype != W_dtype %}
      %x_ext = arith.extf %in : !X_dtype to !accum_type
      %w_ext = arith.extf %in_0 : !W_dtype to !accum_type
      %mul = arith.mulf %x_ext, %w_ext : !accum_type
{% endif %}
      %acc = arith.addf %out, %mul : !accum_type
      linalg.yield %acc : !accum_type
    } -> !result_asm_type
    util.return %4: !result_asm_type
  }
}
