module attributes {iree_codegen.tuning_spec_with_default_entrypoint, transform.with_named_sequence} {
  transform.named_sequence @apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
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
        @match_conv_2d_bfloat16_forward_2x118x182x448_nhwc_896x1x1x448_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_42952x896x448_bf16xbf16xf32 -> @apply_op_config,
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
