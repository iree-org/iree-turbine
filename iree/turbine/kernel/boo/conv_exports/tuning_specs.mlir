module attributes {iree_codegen.tuning_spec_with_default_entrypoint, transform.with_named_sequence} {
  transform.named_sequence @apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_b_16x24x16x194_nhwc_192x3x3x194_fhwc_nhwf_1x1s_1x1p_1x1d_1g$async_dispatch_0_conv_16x24x16x192x3x3x194_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<16x26x18x194xbf16>, %arg2: tensor<192x3x3x194xbf16>, %arg3: tensor<16x24x16x192xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d4, d2 + d5, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d3, d4, d5, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<16x26x18x194xbf16>, tensor<192x3x3x194xbf16>) outs(%arg3 : tensor<16x24x16x192xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<16x24x16x192xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, padding = [8, 1, 16, 64, 32], promote_operands = [0, 1], reduction = [0, 0, 0, 0, 2], subgroup = [1, 1, 1, 4, 0], subgroup_m_count = 8 : i64, subgroup_n_count = 1 : i64, workgroup = [8, 1, 16, 64, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [512, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module5289363779729958125_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_16x225x225x16_nhwc_64x3x3x16_fhwc_nhwf_1x1s_1x1p_1x1d_1g$async_dispatch_0_conv_16x225x225x64x3x3x16_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<16x227x227x16xbf16>, %arg2: tensor<64x3x3x16xbf16>, %arg3: tensor<16x225x225x64xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d4, d2 + d5, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d3, d4, d5, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<16x227x227x16xbf16>, tensor<64x3x3x16xbf16>) outs(%arg3 : tensor<16x225x225x64xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<16x225x225x64xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_BF16>, padding = [2, 1, 32, 64, 16], promote_operands = [0, 1, 2], reduction = [0, 0, 0, 0, 2], subgroup = [1, 1, 1, 1, 0], subgroup_m_count = 2 : i64, subgroup_n_count = 2 : i64, workgroup = [2, 1, 32, 64, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [256, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @"module-1916327443772794327_apply_op_config"(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_16x225x225x64_nhwc_64x3x3x64_fhwc_nhwf_3x3s_1x1p_1x1d_1g$async_dispatch_0_conv_16x75x75x64x3x3x64_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<16x227x227x64xbf16>, %arg2: tensor<64x3x3x64xbf16>, %arg3: tensor<16x75x75x64xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 * 3 + d4, d2 * 3 + d5, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d3, d4, d5, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<16x227x227x64xbf16>, tensor<64x3x3x64xbf16>) outs(%arg3 : tensor<16x75x75x64xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<16x75x75x64xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, padding = [8, 1, 16, 64, 16], promote_operands = [0, 1], reduction = [0, 0, 0, 0, 1], subgroup = [4, 1, 1, 1, 0], subgroup_m_count = 2 : i64, subgroup_n_count = 4 : i64, workgroup = [8, 1, 16, 64, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [512, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module4100224581331179345_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_16x225x225x64_nhwc_64x1x3x64_fhwc_nhwf_1x1s_0x1p_1x1d_1g$async_dispatch_0_conv_16x225x225x64x3x64_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<16x225x227x64xbf16>, %arg2: tensor<64x3x64xbf16>, %arg3: tensor<16x225x225x64xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2 + d4, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d4, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<16x225x227x64xbf16>, tensor<64x3x64xbf16>) outs(%arg3 : tensor<16x225x225x64xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<16x225x225x64xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_BF16>, padding = [1, 1, 128, 64, 64], promote_operands = [0, 1, 2], reduction = [0, 0, 0, 0, 8], subgroup = [1, 1, 1, 1, 0], subgroup_m_count = 4 : i64, subgroup_n_count = 2 : i64, workgroup = [1, 1, 128, 64, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [512, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module6922405066951115100_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_16x225x225x64_nhwc_64x3x3x64_fhwc_nhwf_1x1s_1x1p_1x1d_1g$async_dispatch_0_conv_16x225x225x64x3x3x64_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<16x227x227x64xbf16>, %arg2: tensor<64x3x3x64xbf16>, %arg3: tensor<16x225x225x64xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d4, d2 + d5, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d3, d4, d5, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<16x227x227x64xbf16>, tensor<64x3x3x64xbf16>) outs(%arg3 : tensor<16x225x225x64xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<16x225x225x64xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_BF16>, padding = [1, 1, 128, 64, 64], promote_operands = [0, 1, 2], reduction = [0, 0, 0, 0, 8], subgroup = [1, 1, 1, 1, 0], subgroup_m_count = 4 : i64, subgroup_n_count = 2 : i64, workgroup = [1, 1, 128, 64, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [512, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module2677224730547331610_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_16x450x450x2_nhwc_2x1x1x2_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_3240000x2x2_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<3240000x2xbf16>, %arg2: tensor<2x2xbf16>, %arg3: tensor<3240000x2xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<3240000x2xbf16>, tensor<2x2xbf16>) outs(%arg3 : tensor<3240000x2xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<3240000x2xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, padding = [96, 16, 16], promote_operands = [0, 1, 2], reduction = [0, 0, 1], subgroup = [6, 1, 0], subgroup_m_count = 1 : i64, subgroup_n_count = 1 : i64, workgroup = [96, 16, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [64, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @"module-8121704562341514944_apply_op_config"(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_16x48x32x768_nhwc_2048x3x3x768_fhwc_nhwf_1x1s_1x1p_1x1d_1g$async_dispatch_0_conv_16x48x32x2048x3x3x768_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<16x50x34x768xbf16>, %arg2: tensor<2048x3x3x768xbf16>, %arg3: tensor<16x48x32x2048xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d4, d2 + d5, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d3, d4, d5, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<16x50x34x768xbf16>, tensor<2048x3x3x768xbf16>) outs(%arg3 : tensor<16x48x32x2048xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<16x48x32x2048xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_BF16>, promote_operands = [0, 1], reduction = [0, 0, 0, 0, 4], subgroup = [2, 2, 1, 2, 0], subgroup_m_count = 2 : i64, subgroup_n_count = 4 : i64, workgroup = [2, 4, 32, 256, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [512, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @"module-8204323381023235852_apply_op_config"(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_2x30x46x2016_nhwc_2016x1x1x2016_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_2760x2016x2016_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<2760x2016xbf16>, %arg2: tensor<2016x2016xbf16>, %arg3: tensor<2760x2016xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<2760x2016xbf16>, tensor<2016x2016xbf16>) outs(%arg3 : tensor<2760x2016xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<2760x2016xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_BF16>, padding = [96, 96, 48], promote_operands = [0, 1, 2], reduction = [0, 0, 6], subgroup = [1, 1, 0], subgroup_m_count = 3 : i64, subgroup_n_count = 3 : i64, workgroup = [96, 96, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [576, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module8189907815231141851_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_2x59x91x2016_nhwc_2016x3x3x56_fhwc_nhwf_2x2s_1x1p_1x1d_36g$async_dispatch_0_conv_2x30x46x36x56x3x3x56_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<2x61x93x36x56xbf16>, %arg2: tensor<36x56x3x3x56xbf16>, %arg3: tensor<2x30x46x36x56xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1 * 2 + d5, d2 * 2 + d6, d3, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d3, d4, d5, d6, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<2x61x93x36x56xbf16>, tensor<36x56x3x3x56xbf16>) outs(%arg3 : tensor<2x30x46x36x56xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<2x30x46x36x56xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, padding = [2, 1, 48, 1, 32, 16], promote_operands = [0, 1, 2], reduction = [0, 0, 0, 0, 0, 1], subgroup = [1, 1, 3, 1, 1, 0], subgroup_m_count = 2 : i64, subgroup_n_count = 2 : i64, workgroup = [2, 1, 48, 1, 32, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [256, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @"module-1904524209651108499_apply_op_config"(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_2x118x182x896_nhwc_896x3x3x56_fhwc_nhwf_2x2s_1x1p_1x1d_16g$async_dispatch_0_conv_2x59x91x16x56x3x3x56_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<2x120x184x16x56xbf16>, %arg2: tensor<16x56x3x3x56xbf16>, %arg3: tensor<2x59x91x16x56xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1 * 2 + d5, d2 * 2 + d6, d3, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d3, d4, d5, d6, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<2x120x184x16x56xbf16>, tensor<16x56x3x3x56xbf16>) outs(%arg3 : tensor<2x59x91x16x56xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<2x59x91x16x56xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_BF16>, padding = [2, 1, 32, 1, 64, 56], promote_operands = [0, 1, 2], reduction = [0, 0, 0, 0, 0, 7], subgroup = [1, 1, 1, 1, 1, 0], subgroup_m_count = 2 : i64, subgroup_n_count = 2 : i64, workgroup = [2, 1, 32, 1, 64, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [256, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module3432973299303531041_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_2x235x363x448_nhwc_448x3x3x56_fhwc_nhwf_2x2s_1x1p_1x1d_8g$async_dispatch_0_conv_2x118x182x8x56x3x3x56_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<2x237x365x8x56xbf16>, %arg2: tensor<8x56x3x3x56xbf16>, %arg3: tensor<2x118x182x8x56xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1 * 2 + d5, d2 * 2 + d6, d3, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d3, d4, d5, d6, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<2x237x365x8x56xbf16>, tensor<8x56x3x3x56xbf16>) outs(%arg3 : tensor<2x118x182x8x56xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<2x118x182x8x56xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_BF16>, padding = [1, 2, 64, 1, 64, 72], promote_operands = [0, 1, 2], reduction = [0, 0, 0, 0, 0, 9], subgroup = [1, 1, 1, 1, 1, 0], subgroup_m_count = 4 : i64, subgroup_n_count = 2 : i64, workgroup = [1, 2, 64, 1, 64, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [512, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @"module-5987359365513314828_apply_op_config"(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_2x470x725x224_nhwc_224x3x3x56_fhwc_nhwf_2x2s_1x1p_1x1d_4g$async_dispatch_0_conv_2x235x363x4x56x3x3x56_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<2x472x727x4x56xbf16>, %arg2: tensor<4x56x3x3x56xbf16>, %arg3: tensor<2x235x363x4x56xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1 * 2 + d5, d2 * 2 + d6, d3, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d3, d4, d5, d6, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<2x472x727x4x56xbf16>, tensor<4x56x3x3x56xbf16>) outs(%arg3 : tensor<2x235x363x4x56xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<2x235x363x4x56xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_BF16>, padding = [1, 1, 64, 1, 32, 24], promote_operands = [0, 1, 2], reduction = [0, 0, 0, 0, 0, 3], subgroup = [1, 1, 1, 1, 1, 0], subgroup_m_count = 2 : i64, subgroup_n_count = 1 : i64, workgroup = [1, 1, 64, 1, 32, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [128, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @"module-858300102527720657_apply_op_config"(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_2x118x182x448_nhwc_448x3x3x56_fhwc_nhwf_1x1s_1x1p_1x1d_8g$async_dispatch_0_conv_2x118x182x8x56x3x3x56_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<2x120x184x8x56xbf16>, %arg2: tensor<8x56x3x3x56xbf16>, %arg3: tensor<2x118x182x8x56xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1 + d5, d2 + d6, d3, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d3, d4, d5, d6, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<2x120x184x8x56xbf16>, tensor<8x56x3x3x56xbf16>) outs(%arg3 : tensor<2x118x182x8x56xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<2x118x182x8x56xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_BF16>, padding = [2, 1, 64, 1, 64, 72], promote_operands = [0, 1, 2], reduction = [0, 0, 0, 0, 0, 9], subgroup = [2, 1, 1, 1, 1, 0], subgroup_m_count = 2 : i64, subgroup_n_count = 2 : i64, workgroup = [2, 1, 64, 1, 64, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [256, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @"module-3230150107642515359_apply_op_config"(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_2x59x91x896_nhwc_896x3x3x56_fhwc_nhwf_1x1s_1x1p_1x1d_16g$async_dispatch_0_conv_2x59x91x16x56x3x3x56_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<2x61x93x16x56xbf16>, %arg2: tensor<16x56x3x3x56xbf16>, %arg3: tensor<2x59x91x16x56xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1 + d5, d2 + d6, d3, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d3, d4, d5, d6, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<2x61x93x16x56xbf16>, tensor<16x56x3x3x56xbf16>) outs(%arg3 : tensor<2x59x91x16x56xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<2x59x91x16x56xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, padding = [2, 1, 16, 1, 64, 32], promote_operands = [0, 1, 2], reduction = [0, 0, 0, 0, 0, 2], subgroup = [1, 1, 1, 1, 2, 0], subgroup_m_count = 2 : i64, subgroup_n_count = 2 : i64, workgroup = [2, 1, 16, 1, 64, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [256, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @"module-6706311577560758110_apply_op_config"(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_2x118x182x448_nhwc_896x1x1x448_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_42952x896x448_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<42952x448xbf16>, %arg2: tensor<896x448xbf16>, %arg3: tensor<42952x896xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<42952x448xbf16>, tensor<896x448xbf16>) outs(%arg3 : tensor<42952x896xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<42952x896xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, padding = [48, 32, 64], promote_operands = [0, 1, 2], reduction = [0, 0, 4], subgroup = [3, 2, 0], subgroup_m_count = 1 : i64, subgroup_n_count = 1 : i64, workgroup = [48, 32, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [64, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @"module-5325771805602660052_apply_op_config"(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_2x1x1x56_nhwc_448x1x1x56_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_2x448x56_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<2x56xbf16>, %arg2: tensor<448x56xbf16>, %arg3: tensor<2x448xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<2x56xbf16>, tensor<448x56xbf16>) outs(%arg3 : tensor<2x448xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<2x448xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, padding = [16, 224, 16], promote_operands = [0, 1, 2], reduction = [0, 0, 1], subgroup = [1, 14, 0], subgroup_m_count = 1 : i64, subgroup_n_count = 1 : i64, workgroup = [16, 224, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [64, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @"module-2620838220123778441_apply_op_config"(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_2x940x1450x4_nhwc_3x1x1x4_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_2726000x3x4_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<2726000x4xbf16>, %arg2: tensor<3x4xbf16>, %arg3: tensor<2726000x3xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<2726000x4xbf16>, tensor<3x4xbf16>) outs(%arg3 : tensor<2726000x3xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<2726000x3xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_BF16>, padding = [64, 32, 8], promote_operands = [0, 1, 2], reduction = [0, 0, 1], subgroup = [2, 1, 0], subgroup_m_count = 1 : i64, subgroup_n_count = 1 : i64, workgroup = [64, 32, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [64, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module2561809005636161782_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_2x1x1x2016_nhwc_224x1x1x2016_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_2x224x2016_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<2x2016xbf16>, %arg2: tensor<224x2016xbf16>, %arg3: tensor<2x224xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<2x2016xbf16>, tensor<224x2016xbf16>) outs(%arg3 : tensor<2x224xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<2x224xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, padding = [16, 32, 16], promote_operands = [0, 1, 2], reduction = [0, 0, 1], subgroup = [1, 1, 0], subgroup_m_count = 1 : i64, subgroup_n_count = 2 : i64, workgroup = [16, 32, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [128, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module8003494502233991073_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_2x1x1x112_nhwc_896x1x1x112_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_2x896x112_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<2x112xbf16>, %arg2: tensor<896x112xbf16>, %arg3: tensor<2x896xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<2x112xbf16>, tensor<896x112xbf16>) outs(%arg3 : tensor<2x896xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<2x896xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_BF16>, padding = [32, 128, 8], promote_operands = [0, 1, 2], reduction = [0, 0, 1], subgroup = [1, 2, 0], subgroup_m_count = 1 : i64, subgroup_n_count = 2 : i64, workgroup = [32, 128, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [128, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module2633052976314361527_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_2x118x182x448_nhwc_448x1x1x448_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_42952x448x448_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<42952x448xbf16>, %arg2: tensor<448x448xbf16>, %arg3: tensor<42952x448xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<42952x448xbf16>, tensor<448x448xbf16>) outs(%arg3 : tensor<42952x448xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<42952x448xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_BF16>, padding = [32, 64, 32], promote_operands = [0, 1, 2], reduction = [0, 0, 4], subgroup = [1, 1, 0], subgroup_m_count = 1 : i64, subgroup_n_count = 2 : i64, workgroup = [32, 64, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [128, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module1472885415414460969_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_2x59x91x896_nhwc_896x1x1x896_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_10738x896x896_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<10738x896xbf16>, %arg2: tensor<896x896xbf16>, %arg3: tensor<10738x896xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<10738x896xbf16>, tensor<896x896xbf16>) outs(%arg3 : tensor<10738x896xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<10738x896xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, padding = [48, 128, 32], promote_operands = [0, 1, 2], reduction = [0, 0, 2], subgroup = [3, 1, 0], subgroup_m_count = 1 : i64, subgroup_n_count = 8 : i64, workgroup = [48, 128, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [512, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module5029970341716553467_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @"module-6879911002761170773_apply_op_config"(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_2x1x1x8_nhwc_224x1x1x8_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_2x224x8_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<2x8xbf16>, %arg2: tensor<224x8xbf16>, %arg3: tensor<2x224xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<2x8xbf16>, tensor<224x8xbf16>) outs(%arg3 : tensor<2x224xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<2x224xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, padding = [16, 16, 16], promote_operands = [0, 1, 2], reduction = [0, 0, 1], subgroup = [1, 1, 0], subgroup_m_count = 1 : i64, subgroup_n_count = 1 : i64, workgroup = [16, 16, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [64, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @"module-4933892134847023523_apply_op_config"(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_2x1x1x224_nhwc_56x1x1x224_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_2x56x224_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<2x224xbf16>, %arg2: tensor<56x224xbf16>, %arg3: tensor<2x56xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<2x224xbf16>, tensor<56x224xbf16>) outs(%arg3 : tensor<2x56xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<2x56xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_BF16>, padding = [32, 64, 32], promote_operands = [0, 1, 2], reduction = [0, 0, 4], subgroup = [1, 2, 0], subgroup_m_count = 1 : i64, subgroup_n_count = 1 : i64, workgroup = [32, 64, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [64, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module3487602024594090653_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_2x1x1x224_nhwc_8x1x1x224_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_2x8x224_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<2x224xbf16>, %arg2: tensor<8x224xbf16>, %arg3: tensor<2x8xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<2x224xbf16>, tensor<8x224xbf16>) outs(%arg3 : tensor<2x8xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<2x8xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, padding = [16, 16, 16], promote_operands = [0, 1, 2], reduction = [0, 0, 1], subgroup = [1, 1, 0], subgroup_m_count = 1 : i64, subgroup_n_count = 1 : i64, workgroup = [16, 16, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [64, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module6813850560441988405_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_2x1x1x56_nhwc_224x1x1x56_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_2x224x56_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<2x56xbf16>, %arg2: tensor<224x56xbf16>, %arg3: tensor<2x224xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<2x56xbf16>, tensor<224x56xbf16>) outs(%arg3 : tensor<2x224xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<2x224xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, padding = [16, 112, 16], promote_operands = [0, 1, 2], reduction = [0, 0, 1], subgroup = [1, 1, 0], subgroup_m_count = 1 : i64, subgroup_n_count = 7 : i64, workgroup = [16, 112, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [448, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @"module-7169856524999635951_apply_op_config"(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_2x1x1x224_nhwc_2016x1x1x224_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_2x2016x224_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<2x224xbf16>, %arg2: tensor<2016x224xbf16>, %arg3: tensor<2x2016xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<2x224xbf16>, tensor<2016x224xbf16>) outs(%arg3 : tensor<2x2016xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<2x2016xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_BF16>, padding = [32, 288, 32], promote_operands = [0, 1, 2], reduction = [0, 0, 4], subgroup = [1, 1, 0], subgroup_m_count = 1 : i64, subgroup_n_count = 9 : i64, workgroup = [32, 288, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [576, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @"module-6974726527491627449_apply_op_config"(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @"module-2620838220123778441_match_conv_2d_bfloat16_forward_2x1x1x56_nhwc_448x1x1x56_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_2x448x56_bf16xbf16xf32"(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<2x56xbf16>, %arg2: tensor<448x56xbf16>, %arg3: tensor<2x448xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<2x56xbf16>, tensor<448x56xbf16>) outs(%arg3 : tensor<2x448xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<2x448xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, padding = [16, 448, 32], promote_operands = [0, 1, 2], reduction = [0, 0, 2], subgroup = [1, 28, 0], subgroup_m_count = 1 : i64, subgroup_n_count = 1 : i64, workgroup = [16, 448, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [64, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module2902764805326496053_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_2x1x1x448_nhwc_56x1x1x448_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_2x56x448_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<2x448xbf16>, %arg2: tensor<56x448xbf16>, %arg3: tensor<2x56xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<2x448xbf16>, tensor<56x448xbf16>) outs(%arg3 : tensor<2x56xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<2x56xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_BF16>, padding = [32, 64, 224], promote_operands = [0, 1, 2], reduction = [0, 0, 28], subgroup = [1, 1, 0], subgroup_m_count = 1 : i64, subgroup_n_count = 2 : i64, workgroup = [32, 64, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [128, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @"module-5880248334585732510_apply_op_config"(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @module9073207832815906171_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_2x940x1450x3_nhwc_32x3x3x3_fhwc_nhwf_2x2s_1x1p_1x1d_1g$async_dispatch_0_conv_2x470x725x32x3x3x3_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<2x942x1452x3xbf16>, %arg2: tensor<32x3x3x3xbf16>, %arg3: tensor<2x470x725x32xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 * 2 + d4, d2 * 2 + d5, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d3, d4, d5, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<2x942x1452x3xbf16>, tensor<32x3x3x3xbf16>) outs(%arg3 : tensor<2x470x725x32xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<2x470x725x32xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_BF16>, padding = [2, 1, 32, 32, 8], promote_operands = [0, 1, 2], reduction = [0, 0, 0, 0, 1], subgroup = [1, 1, 1, 1, 0], subgroup_m_count = 2 : i64, subgroup_n_count = 1 : i64, workgroup = [2, 1, 32, 32, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [128, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @__kernel_config(%arg0: !transform.any_op {transform.consumed}) -> !transform.any_op attributes {iree_codegen.tuning_spec_entrypoint} {
    %updated_root = transform.foreach_match in %arg0
        @match_conv_2d_bfloat16_forward_b_16x24x16x194_nhwc_192x3x3x194_fhwc_nhwf_1x1s_1x1p_1x1d_1g$async_dispatch_0_conv_16x24x16x192x3x3x194_bf16xbf16xf32 -> @apply_op_config,
        @match_conv_2d_bfloat16_forward_16x225x225x16_nhwc_64x3x3x16_fhwc_nhwf_1x1s_1x1p_1x1d_1g$async_dispatch_0_conv_16x225x225x64x3x3x16_bf16xbf16xf32 -> @module5289363779729958125_apply_op_config,
        @match_conv_2d_bfloat16_forward_16x225x225x64_nhwc_64x3x3x64_fhwc_nhwf_3x3s_1x1p_1x1d_1g$async_dispatch_0_conv_16x75x75x64x3x3x64_bf16xbf16xf32 -> @"module-1916327443772794327_apply_op_config",
        @match_conv_2d_bfloat16_forward_16x225x225x64_nhwc_64x1x3x64_fhwc_nhwf_1x1s_0x1p_1x1d_1g$async_dispatch_0_conv_16x225x225x64x3x64_bf16xbf16xf32 -> @module4100224581331179345_apply_op_config,
        @match_conv_2d_bfloat16_forward_16x225x225x64_nhwc_64x3x3x64_fhwc_nhwf_1x1s_1x1p_1x1d_1g$async_dispatch_0_conv_16x225x225x64x3x3x64_bf16xbf16xf32 -> @module6922405066951115100_apply_op_config,
        @match_conv_2d_bfloat16_forward_16x450x450x2_nhwc_2x1x1x2_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_3240000x2x2_bf16xbf16xf32 -> @module2677224730547331610_apply_op_config,
        @match_conv_2d_bfloat16_forward_16x48x32x768_nhwc_2048x3x3x768_fhwc_nhwf_1x1s_1x1p_1x1d_1g$async_dispatch_0_conv_16x48x32x2048x3x3x768_bf16xbf16xf32 -> @"module-8121704562341514944_apply_op_config",
        @match_conv_2d_bfloat16_forward_2x30x46x2016_nhwc_2016x1x1x2016_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_2760x2016x2016_bf16xbf16xf32 -> @"module-8204323381023235852_apply_op_config",
        @match_conv_2d_bfloat16_forward_2x59x91x2016_nhwc_2016x3x3x56_fhwc_nhwf_2x2s_1x1p_1x1d_36g$async_dispatch_0_conv_2x30x46x36x56x3x3x56_bf16xbf16xf32 -> @module8189907815231141851_apply_op_config,
        @match_conv_2d_bfloat16_forward_2x118x182x896_nhwc_896x3x3x56_fhwc_nhwf_2x2s_1x1p_1x1d_16g$async_dispatch_0_conv_2x59x91x16x56x3x3x56_bf16xbf16xf32 -> @"module-1904524209651108499_apply_op_config",
        @match_conv_2d_bfloat16_forward_2x235x363x448_nhwc_448x3x3x56_fhwc_nhwf_2x2s_1x1p_1x1d_8g$async_dispatch_0_conv_2x118x182x8x56x3x3x56_bf16xbf16xf32 -> @module3432973299303531041_apply_op_config,
        @match_conv_2d_bfloat16_forward_2x470x725x224_nhwc_224x3x3x56_fhwc_nhwf_2x2s_1x1p_1x1d_4g$async_dispatch_0_conv_2x235x363x4x56x3x3x56_bf16xbf16xf32 -> @"module-5987359365513314828_apply_op_config",
        @match_conv_2d_bfloat16_forward_2x118x182x448_nhwc_448x3x3x56_fhwc_nhwf_1x1s_1x1p_1x1d_8g$async_dispatch_0_conv_2x118x182x8x56x3x3x56_bf16xbf16xf32 -> @"module-858300102527720657_apply_op_config",
        @match_conv_2d_bfloat16_forward_2x59x91x896_nhwc_896x3x3x56_fhwc_nhwf_1x1s_1x1p_1x1d_16g$async_dispatch_0_conv_2x59x91x16x56x3x3x56_bf16xbf16xf32 -> @"module-3230150107642515359_apply_op_config",
        @match_conv_2d_bfloat16_forward_2x118x182x448_nhwc_896x1x1x448_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_42952x896x448_bf16xbf16xf32 -> @"module-6706311577560758110_apply_op_config",
        @match_conv_2d_bfloat16_forward_2x1x1x56_nhwc_448x1x1x56_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_2x448x56_bf16xbf16xf32 -> @"module-5325771805602660052_apply_op_config",
        @match_conv_2d_bfloat16_forward_2x940x1450x4_nhwc_3x1x1x4_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_2726000x3x4_bf16xbf16xf32 -> @"module-2620838220123778441_apply_op_config",
        @match_conv_2d_bfloat16_forward_2x1x1x2016_nhwc_224x1x1x2016_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_2x224x2016_bf16xbf16xf32 -> @module2561809005636161782_apply_op_config,
        @match_conv_2d_bfloat16_forward_2x1x1x112_nhwc_896x1x1x112_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_2x896x112_bf16xbf16xf32 -> @module8003494502233991073_apply_op_config,
        @match_conv_2d_bfloat16_forward_2x118x182x448_nhwc_448x1x1x448_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_42952x448x448_bf16xbf16xf32 -> @module2633052976314361527_apply_op_config,
        @match_conv_2d_bfloat16_forward_2x59x91x896_nhwc_896x1x1x896_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_10738x896x896_bf16xbf16xf32 -> @module1472885415414460969_apply_op_config,
        @match_conv_2d_bfloat16_forward_2x1x1x8_nhwc_224x1x1x8_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_2x224x8_bf16xbf16xf32 -> @"module-6879911002761170773_apply_op_config",
        @match_conv_2d_bfloat16_forward_2x1x1x224_nhwc_56x1x1x224_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_2x56x224_bf16xbf16xf32 -> @"module-4933892134847023523_apply_op_config",
        @match_conv_2d_bfloat16_forward_2x1x1x224_nhwc_8x1x1x224_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_2x8x224_bf16xbf16xf32 -> @module3487602024594090653_apply_op_config,
        @match_conv_2d_bfloat16_forward_2x1x1x56_nhwc_224x1x1x56_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_2x224x56_bf16xbf16xf32 -> @module6813850560441988405_apply_op_config,
        @match_conv_2d_bfloat16_forward_2x1x1x224_nhwc_2016x1x1x224_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_2x2016x224_bf16xbf16xf32 -> @"module-7169856524999635951_apply_op_config",
        @"module-2620838220123778441_match_conv_2d_bfloat16_forward_2x1x1x56_nhwc_448x1x1x56_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_2x448x56_bf16xbf16xf32" -> @"module-6974726527491627449_apply_op_config",
        @match_conv_2d_bfloat16_forward_2x1x1x448_nhwc_56x1x1x448_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_2x56x448_bf16xbf16xf32 -> @module2902764805326496053_apply_op_config,
        @match_conv_2d_bfloat16_forward_2x940x1450x3_nhwc_32x3x3x3_fhwc_nhwf_2x2s_1x1p_1x1d_1g$async_dispatch_0_conv_2x470x725x32x3x3x3_bf16xbf16xf32 -> @module9073207832815906171_apply_op_config : (!transform.any_op) -> !transform.any_op
    transform.yield %updated_root : !transform.any_op
  }
}
