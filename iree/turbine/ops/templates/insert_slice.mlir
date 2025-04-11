!src_type = {{src_type}}
!dst_type = {{dst_type}}

module @module {

util.func private @insert_slice_{{spec_sig}}
  (%src: !src_type, %dst: !dst_type)
    -> !dst_type {
    {{dynamic_dim_lines}}
    %result = tensor.insert_slice %src into %dst{{offset}} {{sizes}} {{stride}} : !src_type into !dst_type
    util.return %result : !dst_type
    }
}
