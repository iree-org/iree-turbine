module attributes {iree_codegen.tuning_spec_with_default_entrypoint, transform.with_named_sequence} {
  transform.named_sequence @apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_3x1x1x112_nhwc_448x1x1x112_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_3x448x112_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<3x112xbf16>, %arg2: tensor<448x112xbf16>, %arg3: tensor<3x448xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<3x112xbf16>, tensor<448x112xbf16>) outs(%arg3 : tensor<3x448xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<3x448xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_BF16>, padding = [32, 32, 16], promote_operands = [0, 1, 2], reduction = [0, 0, 2], subgroup = [1, 1, 0], subgroup_m_count = 1 : i64, subgroup_n_count = 1 : i64, workgroup = [32, 32, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [64, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module4559739451968120734_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_1x118x182x448_nhwc_896x1x1x448_fhwc_nhwf_2x2s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_59x91x896x448_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<59x91x448xbf16>, %arg2: tensor<896x448xbf16>, %arg3: tensor<59x91x896xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<59x91x448xbf16>, tensor<896x448xbf16>) outs(%arg3 : tensor<59x91x896xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<59x91x896xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, padding = [1, 32, 64, 64], promote_operands = [0, 1, 2], reduction = [0, 0, 0, 4], subgroup = [1, 2, 4, 0], subgroup_m_count = 1 : i64, subgroup_n_count = 1 : i64, workgroup = [1, 32, 64, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [64, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module7882439876625634314_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_2x235x363x224_nhwc_448x1x1x224_fhwc_nhwf_2x2s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_2x118x182x448x224_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<2x118x182x224xbf16>, %arg2: tensor<448x224xbf16>, %arg3: tensor<2x118x182x448xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<2x118x182x224xbf16>, tensor<448x224xbf16>) outs(%arg3 : tensor<2x118x182x448xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<2x118x182x448xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, padding = [1, 1, 64, 64, 32], promote_operands = [0, 1, 2], reduction = [0, 0, 0, 0, 2], subgroup = [1, 1, 2, 2, 0], subgroup_m_count = 2 : i64, subgroup_n_count = 2 : i64, workgroup = [1, 1, 64, 64, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [256, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @"module-2995968965573836068_apply_op_config"(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_7x1x1x2016_nhwc_224x1x1x2016_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_7x224x2016_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<7x2016xbf16>, %arg2: tensor<224x2016xbf16>, %arg3: tensor<7x224xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<7x2016xbf16>, tensor<224x2016xbf16>) outs(%arg3 : tensor<7x224xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<7x224xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, padding = [16, 16, 224], promote_operands = [0, 1, 2], reduction = [0, 0, 14], subgroup = [1, 1, 0], subgroup_m_count = 1 : i64, subgroup_n_count = 1 : i64, workgroup = [16, 16, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [64, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @"module-1383557259038100597_apply_op_config"(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_10x1x1x2016_nhwc_224x1x1x2016_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_10x224x2016_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<10x2016xbf16>, %arg2: tensor<224x2016xbf16>, %arg3: tensor<10x224xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<10x2016xbf16>, tensor<224x2016xbf16>) outs(%arg3 : tensor<10x224xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<10x224xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, padding = [16, 32, 224], promote_operands = [0, 1, 2], reduction = [0, 0, 14], subgroup = [1, 2, 0], subgroup_m_count = 1 : i64, subgroup_n_count = 1 : i64, workgroup = [16, 32, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [64, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module2820520407618095204_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_12x1x1x2016_nhwc_224x1x1x2016_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_12x224x2016_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<12x2016xbf16>, %arg2: tensor<224x2016xbf16>, %arg3: tensor<12x224xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<12x2016xbf16>, tensor<224x2016xbf16>) outs(%arg3 : tensor<12x224xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<12x224xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, padding = [16, 32, 96], promote_operands = [0, 1, 2], reduction = [0, 0, 6], subgroup = [1, 2, 0], subgroup_m_count = 1 : i64, subgroup_n_count = 1 : i64, workgroup = [16, 32, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [64, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module3713156600714724778_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_6x1x1x2016_nhwc_224x1x1x2016_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_6x224x2016_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<6x2016xbf16>, %arg2: tensor<224x2016xbf16>, %arg3: tensor<6x224xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<6x2016xbf16>, tensor<224x2016xbf16>) outs(%arg3 : tensor<6x224xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<6x224xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, padding = [16, 16, 288], promote_operands = [0, 1, 2], reduction = [0, 0, 18], subgroup = [1, 1, 0], subgroup_m_count = 1 : i64, subgroup_n_count = 1 : i64, workgroup = [16, 16, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [64, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module8560219421819835492_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_1x59x91x896_nhwc_2016x1x1x896_fhwc_nhwf_2x2s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_30x46x2016x896_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<30x46x896xbf16>, %arg2: tensor<2016x896xbf16>, %arg3: tensor<30x46x2016xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<30x46x896xbf16>, tensor<2016x896xbf16>) outs(%arg3 : tensor<30x46x2016xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<30x46x2016xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, padding = [6, 16, 32, 64], promote_operands = [0, 1, 2], reduction = [0, 0, 0, 4], subgroup = [6, 1, 1, 0], subgroup_m_count = 1 : i64, subgroup_n_count = 2 : i64, workgroup = [6, 16, 32, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [128, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module5028598877358614442_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_1x235x363x224_nhwc_448x1x1x224_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_235x363x448x224_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<235x363x224xbf16>, %arg2: tensor<448x224xbf16>, %arg3: tensor<235x363x448xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<235x363x224xbf16>, tensor<448x224xbf16>) outs(%arg3 : tensor<235x363x448xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<235x363x448xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_BF16>, padding = [1, 64, 64, 32], promote_operands = [0, 1, 2], reduction = [0, 0, 0, 4], subgroup = [1, 1, 1, 0], subgroup_m_count = 2 : i64, subgroup_n_count = 2 : i64, workgroup = [1, 64, 64, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [256, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module2675858531162868376_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_5x1x1x2016_nhwc_224x1x1x2016_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_5x224x2016_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<5x2016xbf16>, %arg2: tensor<224x2016xbf16>, %arg3: tensor<5x224xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<5x2016xbf16>, tensor<224x2016xbf16>) outs(%arg3 : tensor<5x224xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<5x224xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, padding = [16, 16, 32], promote_operands = [0, 1, 2], reduction = [0, 0, 2], subgroup = [1, 1, 0], subgroup_m_count = 1 : i64, subgroup_n_count = 1 : i64, workgroup = [16, 16, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [64, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module8272023259564752550_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_1x30x46x2016_nhwc_2048x1x1x2016_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_30x46x2048x2016_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<30x46x2016xbf16>, %arg2: tensor<2048x2016xbf16>, %arg3: tensor<30x46x2048xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<30x46x2016xbf16>, tensor<2048x2016xbf16>) outs(%arg3 : tensor<30x46x2048xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<30x46x2048xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, padding = [6, 16, 64, 96], promote_operands = [0, 1, 2], reduction = [0, 0, 0, 6], subgroup = [1, 1, 4, 0], subgroup_m_count = 6 : i64, subgroup_n_count = 1 : i64, workgroup = [6, 16, 64, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [384, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module3773559851783729558_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_2x30x46x2016_nhwc_1536x1x1x2016_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_2760x1536x2016_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<2760x2016xbf16>, %arg2: tensor<1536x2016xbf16>, %arg3: tensor<2760x1536xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<2760x2016xbf16>, tensor<1536x2016xbf16>) outs(%arg3 : tensor<2760x1536xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<2760x1536xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_BF16>, padding = [96, 96, 48], promote_operands = [0, 1, 2], reduction = [0, 0, 6], subgroup = [1, 1, 0], subgroup_m_count = 3 : i64, subgroup_n_count = 3 : i64, workgroup = [96, 96, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [576, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module6802500929324229951_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_1x30x46x2016_nhwc_2016x1x1x2016_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_30x46x2016x2016_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<30x46x2016xbf16>, %arg2: tensor<2016x2016xbf16>, %arg3: tensor<30x46x2016xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<30x46x2016xbf16>, tensor<2016x2016xbf16>) outs(%arg3 : tensor<30x46x2016xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<30x46x2016xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, padding = [2, 48, 48, 96], promote_operands = [0, 1, 2], reduction = [0, 0, 0, 6], subgroup = [2, 1, 1, 0], subgroup_m_count = 3 : i64, subgroup_n_count = 3 : i64, workgroup = [2, 48, 48, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [576, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @"module-7001239958652609188_apply_op_config"(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_4x1x1x112_nhwc_896x1x1x112_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_4x896x112_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<4x112xbf16>, %arg2: tensor<896x112xbf16>, %arg3: tensor<4x896xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<4x112xbf16>, tensor<896x112xbf16>) outs(%arg3 : tensor<4x896xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<4x896xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_BF16>, padding = [32, 32, 16], promote_operands = [0, 1, 2], reduction = [0, 0, 2], subgroup = [1, 1, 0], subgroup_m_count = 1 : i64, subgroup_n_count = 1 : i64, workgroup = [32, 32, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [64, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @"module-993111789093238903_apply_op_config"(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_1x118x182x448_nhwc_896x1x1x448_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_118x182x896x448_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<118x182x448xbf16>, %arg2: tensor<896x448xbf16>, %arg3: tensor<118x182x896xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<118x182x448xbf16>, tensor<896x448xbf16>) outs(%arg3 : tensor<118x182x896xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<118x182x896xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, padding = [1, 64, 128, 64], promote_operands = [0, 1, 2], reduction = [0, 0, 0, 4], subgroup = [1, 1, 4, 0], subgroup_m_count = 4 : i64, subgroup_n_count = 2 : i64, workgroup = [1, 64, 128, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [512, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module5885234969857106962_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_4x118x182x448_nhwc_896x1x1x448_fhwc_nhwf_2x2s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_4x59x91x896x448_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<4x59x91x448xbf16>, %arg2: tensor<896x448xbf16>, %arg3: tensor<4x59x91x896xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<4x59x91x448xbf16>, tensor<896x448xbf16>) outs(%arg3 : tensor<4x59x91x896xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<4x59x91x896xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, padding = [4, 1, 32, 64, 64], promote_operands = [0, 1, 2], reduction = [0, 0, 0, 0, 4], subgroup = [2, 1, 1, 2, 0], subgroup_m_count = 4 : i64, subgroup_n_count = 2 : i64, workgroup = [4, 1, 32, 64, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [512, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module4586823122037315142_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
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
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, padding = [16, 32, 32], promote_operands = [0, 1, 2], reduction = [0, 0, 2], subgroup = [1, 2, 0], subgroup_m_count = 1 : i64, subgroup_n_count = 1 : i64, workgroup = [16, 32, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [64, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module356981495459478889_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_4x1x1x224_nhwc_2016x1x1x224_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_4x2016x224_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<4x224xbf16>, %arg2: tensor<2016x224xbf16>, %arg3: tensor<4x2016xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<4x224xbf16>, tensor<2016x224xbf16>) outs(%arg3 : tensor<4x2016xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<4x2016xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, padding = [16, 96, 224], promote_operands = [0, 1, 2], reduction = [0, 0, 14], subgroup = [1, 6, 0], subgroup_m_count = 1 : i64, subgroup_n_count = 1 : i64, workgroup = [16, 96, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [64, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @"module-8170860178698619477_apply_op_config"(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_4x30x46x2016_nhwc_1536x1x1x2016_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_5520x1536x2016_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<5520x2016xbf16>, %arg2: tensor<1536x2016xbf16>, %arg3: tensor<5520x1536xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<5520x2016xbf16>, tensor<1536x2016xbf16>) outs(%arg3 : tensor<5520x1536xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<5520x1536xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, promote_operands = [0, 1], reduction = [0, 0, 2], subgroup = [15, 2, 0], subgroup_m_count = 1 : i64, subgroup_n_count = 3 : i64, workgroup = [240, 96, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [192, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module7994357265823291861_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_3x1x1x224_nhwc_2016x1x1x224_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_3x2016x224_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<3x224xbf16>, %arg2: tensor<2016x224xbf16>, %arg3: tensor<3x2016xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<3x224xbf16>, tensor<2016x224xbf16>) outs(%arg3 : tensor<3x2016xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<3x2016xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_BF16>, padding = [32, 96, 112], promote_operands = [0, 1, 2], reduction = [0, 0, 14], subgroup = [1, 3, 0], subgroup_m_count = 1 : i64, subgroup_n_count = 1 : i64, workgroup = [32, 96, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [64, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module7431102081689966800_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_5x30x46x2016_nhwc_768x1x1x2016_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_6900x768x2016_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<6900x2016xbf16>, %arg2: tensor<768x2016xbf16>, %arg3: tensor<6900x768xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<6900x2016xbf16>, tensor<768x2016xbf16>) outs(%arg3 : tensor<6900x768xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<6900x768xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_BF16>, padding = [64, 96, 96], promote_operands = [0, 1, 2], reduction = [0, 0, 12], subgroup = [1, 1, 0], subgroup_m_count = 2 : i64, subgroup_n_count = 3 : i64, workgroup = [64, 96, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [384, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module4332867699959355936_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_1x59x91x896_nhwc_2016x1x1x896_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_59x91x2016x896_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<59x91x896xbf16>, %arg2: tensor<2016x896xbf16>, %arg3: tensor<59x91x2016xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<59x91x896xbf16>, tensor<2016x896xbf16>) outs(%arg3 : tensor<59x91x2016xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<59x91x2016xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, padding = [1, 96, 48, 32], promote_operands = [0, 1, 2], reduction = [0, 0, 0, 2], subgroup = [1, 2, 3, 0], subgroup_m_count = 3 : i64, subgroup_n_count = 1 : i64, workgroup = [1, 96, 48, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [192, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @"module-806051145099938734_apply_op_config"(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_4x1x1x112_nhwc_448x1x1x112_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_4x448x112_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<4x112xbf16>, %arg2: tensor<448x112xbf16>, %arg3: tensor<4x448xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<4x112xbf16>, tensor<448x112xbf16>) outs(%arg3 : tensor<4x448xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<4x448xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_BF16>, padding = [32, 64, 112], promote_operands = [0, 1, 2], reduction = [0, 0, 14], subgroup = [1, 2, 0], subgroup_m_count = 1 : i64, subgroup_n_count = 1 : i64, workgroup = [32, 64, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [64, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @"module-7676363064080114619_apply_op_config"(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_2x59x91x896_nhwc_2016x1x1x896_fhwc_nhwf_2x2s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_2x30x46x2016x896_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<2x30x46x896xbf16>, %arg2: tensor<2016x896xbf16>, %arg3: tensor<2x30x46x2016xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<2x30x46x896xbf16>, tensor<2016x896xbf16>) outs(%arg3 : tensor<2x30x46x2016xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<2x30x46x2016xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, padding = [1, 6, 16, 48, 128], promote_operands = [0, 1, 2], reduction = [0, 0, 0, 0, 8], subgroup = [1, 3, 1, 1, 0], subgroup_m_count = 2 : i64, subgroup_n_count = 3 : i64, workgroup = [1, 6, 16, 48, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [384, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module6588666152883867170_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_3x59x91x896_nhwc_2016x1x1x896_fhwc_nhwf_2x2s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_3x30x46x2016x896_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<3x30x46x896xbf16>, %arg2: tensor<2016x896xbf16>, %arg3: tensor<3x30x46x2016xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<3x30x46x896xbf16>, tensor<2016x896xbf16>) outs(%arg3 : tensor<3x30x46x2016xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<3x30x46x2016xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, padding = [3, 2, 16, 96, 64], promote_operands = [0, 1, 2], reduction = [0, 0, 0, 0, 4], subgroup = [1, 1, 1, 6, 0], subgroup_m_count = 6 : i64, subgroup_n_count = 1 : i64, workgroup = [3, 2, 16, 96, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [384, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @"module-8030569889332935828_apply_op_config"(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_3x59x91x2016_nhwc_2016x3x3x56_fhwc_nhwf_2x2s_1x1p_1x1d_36g$async_dispatch_0_conv_3x30x46x36x56x3x3x56_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<3x61x93x36x56xbf16>, %arg2: tensor<36x56x3x3x56xbf16>, %arg3: tensor<3x30x46x36x56xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1 * 2 + d5, d2 * 2 + d6, d3, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d3, d4, d5, d6, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<3x61x93x36x56xbf16>, tensor<36x56x3x3x56xbf16>) outs(%arg3 : tensor<3x30x46x36x56xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<3x30x46x36x56xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_BF16>, padding = [1, 3, 64, 1, 64, 24], promote_operands = [0, 1, 2], reduction = [0, 0, 0, 0, 0, 3], subgroup = [1, 1, 2, 1, 2, 0], subgroup_m_count = 3 : i64, subgroup_n_count = 1 : i64, workgroup = [1, 3, 64, 1, 64, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [192, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module5363443944264822312_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_2x59x91x896_nhwc_2016x1x1x896_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_10738x2016x896_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<10738x896xbf16>, %arg2: tensor<2016x896xbf16>, %arg3: tensor<10738x2016xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<10738x896xbf16>, tensor<2016x896xbf16>) outs(%arg3 : tensor<10738x2016xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<10738x2016xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, padding = [96, 96, 64], promote_operands = [0, 1, 2], reduction = [0, 0, 4], subgroup = [2, 3, 0], subgroup_m_count = 3 : i64, subgroup_n_count = 2 : i64, workgroup = [96, 96, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [384, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module2458017827010063645_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_4x59x91x896_nhwc_2016x1x1x896_fhwc_nhwf_2x2s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_4x30x46x2016x896_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<4x30x46x896xbf16>, %arg2: tensor<2016x896xbf16>, %arg3: tensor<4x30x46x2016xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<4x30x46x896xbf16>, tensor<2016x896xbf16>) outs(%arg3 : tensor<4x30x46x2016xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<4x30x46x2016xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, padding = [1, 1, 48, 48, 32], promote_operands = [0, 1, 2], reduction = [0, 0, 0, 0, 2], subgroup = [1, 1, 3, 1, 0], subgroup_m_count = 1 : i64, subgroup_n_count = 3 : i64, workgroup = [1, 1, 48, 48, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [192, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @"module-8605844983155302868_apply_op_config"(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_4x59x91x2016_nhwc_2016x3x3x56_fhwc_nhwf_2x2s_1x1p_1x1d_36g$async_dispatch_0_conv_4x30x46x36x56x3x3x56_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<4x61x93x36x56xbf16>, %arg2: tensor<36x56x3x3x56xbf16>, %arg3: tensor<4x30x46x36x56xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1 * 2 + d5, d2 * 2 + d6, d3, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d3, d4, d5, d6, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<4x61x93x36x56xbf16>, tensor<36x56x3x3x56xbf16>) outs(%arg3 : tensor<4x30x46x36x56xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<4x30x46x36x56xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_BF16>, padding = [1, 3, 64, 1, 64, 24], promote_operands = [0, 1, 2], reduction = [0, 0, 0, 0, 0, 3], subgroup = [1, 1, 2, 1, 2, 0], subgroup_m_count = 3 : i64, subgroup_n_count = 1 : i64, workgroup = [1, 3, 64, 1, 64, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [192, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module4543929421270441657_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_4x30x46x2016_nhwc_2048x1x1x2016_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_5520x2048x2016_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<5520x2016xbf16>, %arg2: tensor<2048x2016xbf16>, %arg3: tensor<5520x2048xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<5520x2016xbf16>, tensor<2048x2016xbf16>) outs(%arg3 : tensor<5520x2048xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<5520x2048xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, promote_operands = [0, 1], reduction = [0, 0, 2], subgroup = [5, 8, 0], subgroup_m_count = 1 : i64, subgroup_n_count = 1 : i64, workgroup = [80, 128, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [64, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module7946456294075948850_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_5x30x46x2016_nhwc_1536x1x1x2016_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_6900x1536x2016_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<6900x2016xbf16>, %arg2: tensor<1536x2016xbf16>, %arg3: tensor<6900x1536xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<6900x2016xbf16>, tensor<1536x2016xbf16>) outs(%arg3 : tensor<6900x1536xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<6900x1536xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_BF16>, padding = [96, 64, 96], promote_operands = [0, 1, 2], reduction = [0, 0, 12], subgroup = [1, 1, 0], subgroup_m_count = 3 : i64, subgroup_n_count = 2 : i64, workgroup = [96, 64, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [384, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module855441149224401939_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_4x30x46x2016_nhwc_2016x1x1x2016_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_5520x2016x2016_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<5520x2016xbf16>, %arg2: tensor<2016x2016xbf16>, %arg3: tensor<5520x2016xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<5520x2016xbf16>, tensor<2016x2016xbf16>) outs(%arg3 : tensor<5520x2016xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<5520x2016xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, promote_operands = [0, 1], reduction = [0, 0, 2], subgroup = [15, 2, 0], subgroup_m_count = 1 : i64, subgroup_n_count = 3 : i64, workgroup = [240, 96, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [192, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @"module-7843621978812482176_apply_op_config"(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_4x118x182x448_nhwc_896x1x1x448_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_85904x896x448_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<85904x448xbf16>, %arg2: tensor<896x448xbf16>, %arg3: tensor<85904x896xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<85904x448xbf16>, tensor<896x448xbf16>) outs(%arg3 : tensor<85904x896xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<85904x896xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, promote_operands = [0, 1], reduction = [0, 0, 2], subgroup = [7, 2, 0], subgroup_m_count = 1 : i64, subgroup_n_count = 7 : i64, workgroup = [112, 224, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [448, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @"module-2373990324862608393_apply_op_config"(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_5x59x91x896_nhwc_2016x1x1x896_fhwc_nhwf_2x2s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_5x30x46x2016x896_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<5x30x46x896xbf16>, %arg2: tensor<2016x896xbf16>, %arg3: tensor<5x30x46x2016xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<5x30x46x896xbf16>, tensor<2016x896xbf16>) outs(%arg3 : tensor<5x30x46x2016xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<5x30x46x2016xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, padding = [1, 6, 16, 96, 64], promote_operands = [0, 1, 2], reduction = [0, 0, 0, 0, 4], subgroup = [1, 1, 1, 6, 0], subgroup_m_count = 6 : i64, subgroup_n_count = 1 : i64, workgroup = [1, 6, 16, 96, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [384, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module8288620826883058744_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_1x235x363x448_nhwc_448x3x3x56_fhwc_nhwf_2x2s_1x1p_1x1d_8g$async_dispatch_0_conv_118x182x8x56x3x3x56_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<237x365x8x56xbf16>, %arg2: tensor<8x56x3x3x56xbf16>, %arg3: tensor<118x182x8x56xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0 * 2 + d4, d1 * 2 + d5, d2, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2, d3, d4, d5, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<237x365x8x56xbf16>, tensor<8x56x3x3x56xbf16>) outs(%arg3 : tensor<118x182x8x56xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<118x182x8x56xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_BF16>, padding = [1, 64, 1, 64, 56], promote_operands = [0, 1, 2], reduction = [0, 0, 0, 0, 7], subgroup = [1, 2, 1, 2, 0], subgroup_m_count = 1 : i64, subgroup_n_count = 1 : i64, workgroup = [1, 64, 1, 64, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [64, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module2686472097007468535_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_2x1x1x224_nhwc_896x1x1x224_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_2x896x224_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<2x224xbf16>, %arg2: tensor<896x224xbf16>, %arg3: tensor<2x896xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<2x224xbf16>, tensor<896x224xbf16>) outs(%arg3 : tensor<2x896xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<2x896xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, padding = [16, 32, 32], promote_operands = [0, 1, 2], reduction = [0, 0, 2], subgroup = [1, 2, 0], subgroup_m_count = 1 : i64, subgroup_n_count = 1 : i64, workgroup = [16, 32, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [64, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module7325674527831915629_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_3x1x1x224_nhwc_896x1x1x224_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_3x896x224_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<3x224xbf16>, %arg2: tensor<896x224xbf16>, %arg3: tensor<3x896xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<3x224xbf16>, tensor<896x224xbf16>) outs(%arg3 : tensor<3x896xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<3x896xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_BF16>, padding = [32, 64, 32], promote_operands = [0, 1, 2], reduction = [0, 0, 4], subgroup = [1, 2, 0], subgroup_m_count = 1 : i64, subgroup_n_count = 1 : i64, workgroup = [32, 64, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [64, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @"module-6003646264170329989_apply_op_config"(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_10x30x46x2016_nhwc_768x1x1x2016_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_13800x768x2016_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<13800x2016xbf16>, %arg2: tensor<768x2016xbf16>, %arg3: tensor<13800x768xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<13800x2016xbf16>, tensor<768x2016xbf16>) outs(%arg3 : tensor<13800x768xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<13800x768xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_BF16>, padding = [96, 64, 96], promote_operands = [0, 1, 2], reduction = [0, 0, 12], subgroup = [1, 1, 0], subgroup_m_count = 3 : i64, subgroup_n_count = 2 : i64, workgroup = [96, 64, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [384, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @"module-4360576302824214298_apply_op_config"(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_1x59x91x896_nhwc_896x1x1x896_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_59x91x896x896_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<59x91x896xbf16>, %arg2: tensor<896x896xbf16>, %arg3: tensor<59x91x896xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<59x91x896xbf16>, tensor<896x896xbf16>) outs(%arg3 : tensor<59x91x896xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<59x91x896xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, padding = [1, 96, 32, 64], promote_operands = [0, 1, 2], reduction = [0, 0, 0, 4], subgroup = [1, 3, 2, 0], subgroup_m_count = 2 : i64, subgroup_n_count = 1 : i64, workgroup = [1, 96, 32, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [128, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module8668743119509608937_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_12x30x46x2016_nhwc_1536x1x1x2016_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_16560x1536x2016_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<16560x2016xbf16>, %arg2: tensor<1536x2016xbf16>, %arg3: tensor<16560x1536xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<16560x2016xbf16>, tensor<1536x2016xbf16>) outs(%arg3 : tensor<16560x1536xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<16560x1536xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, promote_operands = [0, 1], reduction = [0, 0, 2], subgroup = [15, 2, 0], subgroup_m_count = 1 : i64, subgroup_n_count = 3 : i64, workgroup = [240, 96, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [192, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @"module-6940871244302246445_apply_op_config"(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_6x59x91x896_nhwc_2016x1x1x896_fhwc_nhwf_2x2s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_6x30x46x2016x896_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<6x30x46x896xbf16>, %arg2: tensor<2016x896xbf16>, %arg3: tensor<6x30x46x2016xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<6x30x46x896xbf16>, tensor<2016x896xbf16>) outs(%arg3 : tensor<6x30x46x2016xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<6x30x46x2016xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, padding = [1, 6, 16, 96, 64], promote_operands = [0, 1, 2], reduction = [0, 0, 0, 0, 4], subgroup = [1, 2, 1, 3, 0], subgroup_m_count = 3 : i64, subgroup_n_count = 2 : i64, workgroup = [1, 6, 16, 96, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [384, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @"module-1821426178339838639_apply_op_config"(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_1x235x363x224_nhwc_224x3x3x56_fhwc_nhwf_1x1s_1x1p_1x1d_4g$async_dispatch_0_conv_235x363x4x56x3x3x56_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<237x365x4x56xbf16>, %arg2: tensor<4x56x3x3x56xbf16>, %arg3: tensor<235x363x4x56xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0 + d4, d1 + d5, d2, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2, d3, d4, d5, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<237x365x4x56xbf16>, tensor<4x56x3x3x56xbf16>) outs(%arg3 : tensor<235x363x4x56xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<235x363x4x56xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_BF16>, padding = [1, 192, 1, 64, 24], promote_operands = [0, 1, 2], reduction = [0, 0, 0, 0, 3], subgroup = [1, 2, 1, 2, 0], subgroup_m_count = 3 : i64, subgroup_n_count = 1 : i64, workgroup = [1, 192, 1, 64, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [192, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module4439023163120043581_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_12x30x46x2016_nhwc_768x1x1x2016_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_16560x768x2016_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<16560x2016xbf16>, %arg2: tensor<768x2016xbf16>, %arg3: tensor<16560x768xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<16560x2016xbf16>, tensor<768x2016xbf16>) outs(%arg3 : tensor<16560x768xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<16560x768xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, promote_operands = [0, 1], reduction = [0, 0, 2], subgroup = [15, 2, 0], subgroup_m_count = 1 : i64, subgroup_n_count = 3 : i64, workgroup = [240, 96, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [192, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @"module-1401953935242321448_apply_op_config"(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_7x59x91x896_nhwc_2016x1x1x896_fhwc_nhwf_2x2s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_7x30x46x2016x896_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<7x30x46x896xbf16>, %arg2: tensor<2016x896xbf16>, %arg3: tensor<7x30x46x2016xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<7x30x46x896xbf16>, tensor<2016x896xbf16>) outs(%arg3 : tensor<7x30x46x2016xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<7x30x46x2016xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, padding = [1, 2, 48, 96, 64], promote_operands = [0, 1, 2], reduction = [0, 0, 0, 0, 4], subgroup = [1, 2, 1, 3, 0], subgroup_m_count = 3 : i64, subgroup_n_count = 2 : i64, workgroup = [1, 2, 48, 96, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [384, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module973092433758697018_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_1x470x725x224_nhwc_224x3x3x56_fhwc_nhwf_2x2s_1x1p_1x1d_4g$async_dispatch_0_conv_235x363x4x56x3x3x56_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<472x727x4x56xbf16>, %arg2: tensor<4x56x3x3x56xbf16>, %arg3: tensor<235x363x4x56xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0 * 2 + d4, d1 * 2 + d5, d2, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2, d3, d4, d5, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<472x727x4x56xbf16>, tensor<4x56x3x3x56xbf16>) outs(%arg3 : tensor<235x363x4x56xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<235x363x4x56xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_BF16>, padding = [1, 192, 1, 64, 24], promote_operands = [0, 1, 2], reduction = [0, 0, 0, 0, 3], subgroup = [1, 2, 1, 2, 0], subgroup_m_count = 3 : i64, subgroup_n_count = 1 : i64, workgroup = [1, 192, 1, 64, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [192, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module6365494400897409164_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_12x118x182x448_nhwc_896x1x1x448_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_257712x896x448_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<257712x448xbf16>, %arg2: tensor<896x448xbf16>, %arg3: tensor<257712x896xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<257712x448xbf16>, tensor<896x448xbf16>) outs(%arg3 : tensor<257712x896xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<257712x896xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, promote_operands = [0, 1], reduction = [0, 0, 2], subgroup = [7, 2, 0], subgroup_m_count = 1 : i64, subgroup_n_count = 7 : i64, workgroup = [112, 224, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [448, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module4462116553785334358_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
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
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_BF16>, padding = [1, 1, 64, 1, 64, 56], promote_operands = [0, 1, 2], reduction = [0, 0, 0, 0, 0, 7], subgroup = [1, 1, 2, 1, 2, 0], subgroup_m_count = 1 : i64, subgroup_n_count = 1 : i64, workgroup = [1, 1, 64, 1, 64, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [64, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module3453032289433897304_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_4x118x182x448_nhwc_448x1x1x448_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_85904x448x448_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<85904x448xbf16>, %arg2: tensor<448x448xbf16>, %arg3: tensor<85904x448xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<85904x448xbf16>, tensor<448x448xbf16>) outs(%arg3 : tensor<85904x448xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<85904x448xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, promote_operands = [0, 1], reduction = [0, 0, 2], subgroup = [7, 2, 0], subgroup_m_count = 1 : i64, subgroup_n_count = 7 : i64, workgroup = [112, 224, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [448, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module4367576303115157843_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_10x59x91x896_nhwc_2016x1x1x896_fhwc_nhwf_2x2s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_10x30x46x2016x896_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<10x30x46x896xbf16>, %arg2: tensor<2016x896xbf16>, %arg3: tensor<10x30x46x2016xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<10x30x46x896xbf16>, tensor<2016x896xbf16>) outs(%arg3 : tensor<10x30x46x2016xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<10x30x46x2016xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, padding = [1, 2, 48, 96, 64], promote_operands = [0, 1, 2], reduction = [0, 0, 0, 0, 4], subgroup = [1, 1, 1, 6, 0], subgroup_m_count = 6 : i64, subgroup_n_count = 1 : i64, workgroup = [1, 2, 48, 96, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [384, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @"module-4906234458156126534_apply_op_config"(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
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
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_BF16>, padding = [2, 1, 32, 1, 64, 56], promote_operands = [0, 1, 2], reduction = [0, 0, 0, 0, 0, 7], subgroup = [2, 1, 1, 1, 2, 0], subgroup_m_count = 1 : i64, subgroup_n_count = 1 : i64, workgroup = [2, 1, 32, 1, 64, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [64, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module8109018211679045215_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_1x118x182x448_nhwc_448x3x3x56_fhwc_nhwf_1x1s_1x1p_1x1d_8g$async_dispatch_0_conv_118x182x8x56x3x3x56_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<120x184x8x56xbf16>, %arg2: tensor<8x56x3x3x56xbf16>, %arg3: tensor<118x182x8x56xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0 + d4, d1 + d5, d2, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2, d3, d4, d5, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<120x184x8x56xbf16>, tensor<8x56x3x3x56xbf16>) outs(%arg3 : tensor<118x182x8x56xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<118x182x8x56xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_BF16>, padding = [2, 96, 1, 64, 24], promote_operands = [0, 1, 2], reduction = [0, 0, 0, 0, 3], subgroup = [2, 1, 1, 2, 0], subgroup_m_count = 3 : i64, subgroup_n_count = 1 : i64, workgroup = [2, 96, 1, 64, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [192, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module4014810933135631207_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_3x235x363x448_nhwc_448x3x3x56_fhwc_nhwf_2x2s_1x1p_1x1d_8g$async_dispatch_0_conv_3x118x182x8x56x3x3x56_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<3x237x365x8x56xbf16>, %arg2: tensor<8x56x3x3x56xbf16>, %arg3: tensor<3x118x182x8x56xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1 * 2 + d5, d2 * 2 + d6, d3, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d3, d4, d5, d6, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<3x237x365x8x56xbf16>, tensor<8x56x3x3x56xbf16>) outs(%arg3 : tensor<3x118x182x8x56xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<3x118x182x8x56xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_BF16>, padding = [3, 1, 64, 1, 64, 24], promote_operands = [0, 1, 2], reduction = [0, 0, 0, 0, 0, 3], subgroup = [1, 1, 2, 1, 2, 0], subgroup_m_count = 3 : i64, subgroup_n_count = 1 : i64, workgroup = [3, 1, 64, 1, 64, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [192, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @"module-8383162261389945333_apply_op_config"(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_12x30x46x2016_nhwc_2016x1x1x2016_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_16560x2016x2016_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<16560x2016xbf16>, %arg2: tensor<2016x2016xbf16>, %arg3: tensor<16560x2016xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<16560x2016xbf16>, tensor<2016x2016xbf16>) outs(%arg3 : tensor<16560x2016xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<16560x2016xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, promote_operands = [0, 1], reduction = [0, 0, 2], subgroup = [9, 3, 0], subgroup_m_count = 1 : i64, subgroup_n_count = 3 : i64, workgroup = [144, 144, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [192, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module1133905960676286034_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
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
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, padding = [16, 32, 32], promote_operands = [0, 1, 2], reduction = [0, 0, 2], subgroup = [1, 2, 0], subgroup_m_count = 1 : i64, subgroup_n_count = 1 : i64, workgroup = [16, 32, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [64, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @"module-282786097804479605_apply_op_config"(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_4x1x1x2016_nhwc_224x1x1x2016_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_4x224x2016_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<4x2016xbf16>, %arg2: tensor<224x2016xbf16>, %arg3: tensor<4x224xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<4x2016xbf16>, tensor<224x2016xbf16>) outs(%arg3 : tensor<4x224xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<4x224xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_BF16>, padding = [32, 32, 32], promote_operands = [0, 1, 2], reduction = [0, 0, 4], subgroup = [1, 1, 0], subgroup_m_count = 1 : i64, subgroup_n_count = 1 : i64, workgroup = [32, 32, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [64, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module300891171000512914_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_12x59x91x896_nhwc_2016x1x1x896_fhwc_nhwf_2x2s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_12x30x46x2016x896_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<12x30x46x896xbf16>, %arg2: tensor<2016x896xbf16>, %arg3: tensor<12x30x46x2016xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<12x30x46x896xbf16>, tensor<2016x896xbf16>) outs(%arg3 : tensor<12x30x46x2016xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<12x30x46x2016xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, padding = [1, 2, 48, 96, 64], promote_operands = [0, 1, 2], reduction = [0, 0, 0, 0, 4], subgroup = [1, 2, 1, 3, 0], subgroup_m_count = 3 : i64, subgroup_n_count = 2 : i64, workgroup = [1, 2, 48, 96, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [384, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @"module-4816075757461376162_apply_op_config"(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_4x1x1x224_nhwc_896x1x1x224_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_4x896x224_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<4x224xbf16>, %arg2: tensor<896x224xbf16>, %arg3: tensor<4x896xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<4x224xbf16>, tensor<896x224xbf16>) outs(%arg3 : tensor<4x896xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<4x896xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, padding = [16, 16, 32], promote_operands = [0, 1, 2], reduction = [0, 0, 2], subgroup = [1, 1, 0], subgroup_m_count = 1 : i64, subgroup_n_count = 1 : i64, workgroup = [16, 16, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [64, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module9111302107637546479_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_4x235x363x448_nhwc_448x3x3x56_fhwc_nhwf_2x2s_1x1p_1x1d_8g$async_dispatch_0_conv_4x118x182x8x56x3x3x56_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<4x237x365x8x56xbf16>, %arg2: tensor<8x56x3x3x56xbf16>, %arg3: tensor<4x118x182x8x56xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1 * 2 + d5, d2 * 2 + d6, d3, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d3, d4, d5, d6, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<4x237x365x8x56xbf16>, tensor<8x56x3x3x56xbf16>) outs(%arg3 : tensor<4x118x182x8x56xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<4x118x182x8x56xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_BF16>, padding = [1, 1, 64, 1, 64, 56], promote_operands = [0, 1, 2], reduction = [0, 0, 0, 0, 0, 7], subgroup = [1, 1, 2, 1, 2, 0], subgroup_m_count = 1 : i64, subgroup_n_count = 1 : i64, workgroup = [1, 1, 64, 1, 64, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [64, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module7616000247919958829_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_3x1x1x2016_nhwc_224x1x1x2016_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_3x224x2016_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<3x2016xbf16>, %arg2: tensor<224x2016xbf16>, %arg3: tensor<3x224xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<3x2016xbf16>, tensor<224x2016xbf16>) outs(%arg3 : tensor<3x224xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<3x224xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, padding = [16, 16, 288], promote_operands = [0, 1, 2], reduction = [0, 0, 18], subgroup = [1, 1, 0], subgroup_m_count = 1 : i64, subgroup_n_count = 1 : i64, workgroup = [16, 16, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [64, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module5155454980444068234_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_4x118x182x896_nhwc_896x3x3x56_fhwc_nhwf_2x2s_1x1p_1x1d_16g$async_dispatch_0_conv_4x59x91x16x56x3x3x56_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<4x120x184x16x56xbf16>, %arg2: tensor<16x56x3x3x56xbf16>, %arg3: tensor<4x59x91x16x56xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1 * 2 + d5, d2 * 2 + d6, d3, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d3, d4, d5, d6, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<4x120x184x16x56xbf16>, tensor<16x56x3x3x56xbf16>) outs(%arg3 : tensor<4x59x91x16x56xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<4x59x91x16x56xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_BF16>, padding = [2, 1, 32, 1, 64, 56], promote_operands = [0, 1, 2], reduction = [0, 0, 0, 0, 0, 7], subgroup = [2, 1, 1, 1, 2, 0], subgroup_m_count = 1 : i64, subgroup_n_count = 1 : i64, workgroup = [2, 1, 32, 1, 64, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [64, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @"module-8844875856657637720_apply_op_config"(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_6x118x182x896_nhwc_896x3x3x56_fhwc_nhwf_2x2s_1x1p_1x1d_16g$async_dispatch_0_conv_6x59x91x16x56x3x3x56_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<6x120x184x16x56xbf16>, %arg2: tensor<16x56x3x3x56xbf16>, %arg3: tensor<6x59x91x16x56xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1 * 2 + d5, d2 * 2 + d6, d3, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d3, d4, d5, d6, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<6x120x184x16x56xbf16>, tensor<16x56x3x3x56xbf16>) outs(%arg3 : tensor<6x59x91x16x56xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<6x59x91x16x56xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_BF16>, padding = [2, 1, 32, 1, 64, 56], promote_operands = [0, 1, 2], reduction = [0, 0, 0, 0, 0, 7], subgroup = [2, 1, 1, 1, 2, 0], subgroup_m_count = 1 : i64, subgroup_n_count = 1 : i64, workgroup = [2, 1, 32, 1, 64, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [64, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module4471347617703089637_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_5x235x363x448_nhwc_448x3x3x56_fhwc_nhwf_2x2s_1x1p_1x1d_8g$async_dispatch_0_conv_5x118x182x8x56x3x3x56_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<5x237x365x8x56xbf16>, %arg2: tensor<8x56x3x3x56xbf16>, %arg3: tensor<5x118x182x8x56xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1 * 2 + d5, d2 * 2 + d6, d3, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d3, d4, d5, d6, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<5x237x365x8x56xbf16>, tensor<8x56x3x3x56xbf16>) outs(%arg3 : tensor<5x118x182x8x56xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<5x118x182x8x56xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_BF16>, padding = [1, 1, 192, 1, 64, 24], promote_operands = [0, 1, 2], reduction = [0, 0, 0, 0, 0, 3], subgroup = [1, 1, 2, 1, 2, 0], subgroup_m_count = 3 : i64, subgroup_n_count = 1 : i64, workgroup = [1, 1, 192, 1, 64, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [192, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @"module-507249471843353843_apply_op_config"(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_10x118x182x896_nhwc_896x3x3x56_fhwc_nhwf_2x2s_1x1p_1x1d_16g$async_dispatch_0_conv_10x59x91x16x56x3x3x56_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<10x120x184x16x56xbf16>, %arg2: tensor<16x56x3x3x56xbf16>, %arg3: tensor<10x59x91x16x56xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1 * 2 + d5, d2 * 2 + d6, d3, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d3, d4, d5, d6, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<10x120x184x16x56xbf16>, tensor<16x56x3x3x56xbf16>) outs(%arg3 : tensor<10x59x91x16x56xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<10x59x91x16x56xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_BF16>, padding = [2, 1, 32, 1, 64, 56], promote_operands = [0, 1, 2], reduction = [0, 0, 0, 0, 0, 7], subgroup = [2, 1, 1, 1, 2, 0], subgroup_m_count = 1 : i64, subgroup_n_count = 1 : i64, workgroup = [2, 1, 32, 1, 64, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [64, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @"module-7376692197419214668_apply_op_config"(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
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
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_BF16>, padding = [1, 1, 192, 1, 64, 24], promote_operands = [0, 1, 2], reduction = [0, 0, 0, 0, 0, 3], subgroup = [1, 1, 2, 1, 2, 0], subgroup_m_count = 3 : i64, subgroup_n_count = 1 : i64, workgroup = [1, 1, 192, 1, 64, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [192, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module2954593435895805179_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_12x118x182x896_nhwc_896x3x3x56_fhwc_nhwf_2x2s_1x1p_1x1d_16g$async_dispatch_0_conv_12x59x91x16x56x3x3x56_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<12x120x184x16x56xbf16>, %arg2: tensor<16x56x3x3x56xbf16>, %arg3: tensor<12x59x91x16x56xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1 * 2 + d5, d2 * 2 + d6, d3, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d3, d4, d5, d6, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<12x120x184x16x56xbf16>, tensor<16x56x3x3x56xbf16>) outs(%arg3 : tensor<12x59x91x16x56xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<12x59x91x16x56xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_BF16>, padding = [2, 1, 32, 1, 64, 56], promote_operands = [0, 1, 2], reduction = [0, 0, 0, 0, 0, 7], subgroup = [2, 1, 1, 1, 2, 0], subgroup_m_count = 1 : i64, subgroup_n_count = 1 : i64, workgroup = [2, 1, 32, 1, 64, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [64, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module1498352111523621947_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_6x235x363x448_nhwc_448x3x3x56_fhwc_nhwf_2x2s_1x1p_1x1d_8g$async_dispatch_0_conv_6x118x182x8x56x3x3x56_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<6x237x365x8x56xbf16>, %arg2: tensor<8x56x3x3x56xbf16>, %arg3: tensor<6x118x182x8x56xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1 * 2 + d5, d2 * 2 + d6, d3, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d3, d4, d5, d6, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<6x237x365x8x56xbf16>, tensor<8x56x3x3x56xbf16>) outs(%arg3 : tensor<6x118x182x8x56xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<6x118x182x8x56xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_BF16>, padding = [6, 1, 32, 1, 64, 24], promote_operands = [0, 1, 2], reduction = [0, 0, 0, 0, 0, 3], subgroup = [2, 1, 1, 1, 2, 0], subgroup_m_count = 3 : i64, subgroup_n_count = 1 : i64, workgroup = [6, 1, 32, 1, 64, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [192, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @"module-4591256468792391059_apply_op_config"(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_7x235x363x448_nhwc_448x3x3x56_fhwc_nhwf_2x2s_1x1p_1x1d_8g$async_dispatch_0_conv_7x118x182x8x56x3x3x56_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<7x237x365x8x56xbf16>, %arg2: tensor<8x56x3x3x56xbf16>, %arg3: tensor<7x118x182x8x56xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1 * 2 + d5, d2 * 2 + d6, d3, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d3, d4, d5, d6, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<7x237x365x8x56xbf16>, tensor<8x56x3x3x56xbf16>) outs(%arg3 : tensor<7x118x182x8x56xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<7x118x182x8x56xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_BF16>, padding = [1, 1, 192, 1, 64, 24], promote_operands = [0, 1, 2], reduction = [0, 0, 0, 0, 0, 3], subgroup = [1, 1, 2, 1, 2, 0], subgroup_m_count = 3 : i64, subgroup_n_count = 1 : i64, workgroup = [1, 1, 192, 1, 64, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [192, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @"module-3494160608888899646_apply_op_config"(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_10x235x363x448_nhwc_448x3x3x56_fhwc_nhwf_2x2s_1x1p_1x1d_8g$async_dispatch_0_conv_10x118x182x8x56x3x3x56_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<10x237x365x8x56xbf16>, %arg2: tensor<8x56x3x3x56xbf16>, %arg3: tensor<10x118x182x8x56xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1 * 2 + d5, d2 * 2 + d6, d3, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d3, d4, d5, d6, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<10x237x365x8x56xbf16>, tensor<8x56x3x3x56xbf16>) outs(%arg3 : tensor<10x118x182x8x56xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<10x118x182x8x56xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_BF16>, padding = [1, 1, 192, 1, 64, 24], promote_operands = [0, 1, 2], reduction = [0, 0, 0, 0, 0, 3], subgroup = [1, 1, 2, 1, 2, 0], subgroup_m_count = 3 : i64, subgroup_n_count = 1 : i64, workgroup = [1, 1, 192, 1, 64, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [192, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @"module-5609055561883516000_apply_op_config"(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
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
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_BF16>, padding = [128, 64, 64], promote_operands = [0, 1, 2], reduction = [0, 0, 8], subgroup = [1, 1, 0], subgroup_m_count = 4 : i64, subgroup_n_count = 2 : i64, workgroup = [128, 64, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [512, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module7151347827273063051_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_12x118x182x448_nhwc_448x1x1x448_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_257712x448x448_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<257712x448xbf16>, %arg2: tensor<448x448xbf16>, %arg3: tensor<257712x448xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<257712x448xbf16>, tensor<448x448xbf16>) outs(%arg3 : tensor<257712x448xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<257712x448xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, promote_operands = [0, 1], reduction = [0, 0, 2], subgroup = [7, 2, 0], subgroup_m_count = 1 : i64, subgroup_n_count = 7 : i64, workgroup = [112, 224, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [448, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @"module-6365402289902751883_apply_op_config"(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_3x118x182x448_nhwc_448x3x3x56_fhwc_nhwf_1x1s_1x1p_1x1d_8g$async_dispatch_0_conv_3x118x182x8x56x3x3x56_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<3x120x184x8x56xbf16>, %arg2: tensor<8x56x3x3x56xbf16>, %arg3: tensor<3x118x182x8x56xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1 + d5, d2 + d6, d3, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d3, d4, d5, d6, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<3x120x184x8x56xbf16>, tensor<8x56x3x3x56xbf16>) outs(%arg3 : tensor<3x118x182x8x56xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<3x118x182x8x56xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_BF16>, padding = [1, 1, 192, 1, 64, 24], promote_operands = [0, 1, 2], reduction = [0, 0, 0, 0, 0, 3], subgroup = [1, 1, 2, 1, 2, 0], subgroup_m_count = 3 : i64, subgroup_n_count = 1 : i64, workgroup = [1, 1, 192, 1, 64, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [192, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @"module-2802817007326691698_apply_op_config"(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_4x118x182x448_nhwc_448x3x3x56_fhwc_nhwf_1x1s_1x1p_1x1d_8g$async_dispatch_0_conv_4x118x182x8x56x3x3x56_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<4x120x184x8x56xbf16>, %arg2: tensor<8x56x3x3x56xbf16>, %arg3: tensor<4x118x182x8x56xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1 + d5, d2 + d6, d3, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d3, d4, d5, d6, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<4x120x184x8x56xbf16>, tensor<8x56x3x3x56xbf16>) outs(%arg3 : tensor<4x118x182x8x56xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<4x118x182x8x56xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_BF16>, padding = [1, 1, 192, 1, 64, 24], promote_operands = [0, 1, 2], reduction = [0, 0, 0, 0, 0, 3], subgroup = [1, 1, 2, 1, 2, 0], subgroup_m_count = 3 : i64, subgroup_n_count = 1 : i64, workgroup = [1, 1, 192, 1, 64, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [192, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @"module-8847663828986506085_apply_op_config"(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_3x59x91x896_nhwc_896x1x1x896_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_16107x896x896_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<16107x896xbf16>, %arg2: tensor<896x896xbf16>, %arg3: tensor<16107x896xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<16107x896xbf16>, tensor<896x896xbf16>) outs(%arg3 : tensor<16107x896xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<16107x896xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_BF16>, padding = [128, 64, 64], promote_operands = [0, 1, 2], reduction = [0, 0, 8], subgroup = [1, 1, 0], subgroup_m_count = 4 : i64, subgroup_n_count = 2 : i64, workgroup = [128, 64, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [512, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @"module-1484268700394101154_apply_op_config"(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
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
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_BF16>, padding = [1, 1, 192, 1, 64, 24], promote_operands = [0, 1, 2], reduction = [0, 0, 0, 0, 0, 3], subgroup = [1, 1, 2, 1, 2, 0], subgroup_m_count = 3 : i64, subgroup_n_count = 1 : i64, workgroup = [1, 1, 192, 1, 64, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [192, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @"module-2678622183367858380_apply_op_config"(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_2x235x363x224_nhwc_224x3x3x56_fhwc_nhwf_1x1s_1x1p_1x1d_4g$async_dispatch_0_conv_2x235x363x4x56x3x3x56_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<2x237x365x4x56xbf16>, %arg2: tensor<4x56x3x3x56xbf16>, %arg3: tensor<2x235x363x4x56xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1 + d5, d2 + d6, d3, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d3, d4, d5, d6, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<2x237x365x4x56xbf16>, tensor<4x56x3x3x56xbf16>) outs(%arg3 : tensor<2x235x363x4x56xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<2x235x363x4x56xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_BF16>, padding = [1, 1, 192, 1, 64, 24], promote_operands = [0, 1, 2], reduction = [0, 0, 0, 0, 0, 3], subgroup = [1, 1, 2, 1, 2, 0], subgroup_m_count = 3 : i64, subgroup_n_count = 1 : i64, workgroup = [1, 1, 192, 1, 64, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [192, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module1357578566083234864_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_3x470x725x224_nhwc_224x3x3x56_fhwc_nhwf_2x2s_1x1p_1x1d_4g$async_dispatch_0_conv_3x235x363x4x56x3x3x56_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<3x472x727x4x56xbf16>, %arg2: tensor<4x56x3x3x56xbf16>, %arg3: tensor<3x235x363x4x56xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1 * 2 + d5, d2 * 2 + d6, d3, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d3, d4, d5, d6, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<3x472x727x4x56xbf16>, tensor<4x56x3x3x56xbf16>) outs(%arg3 : tensor<3x235x363x4x56xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<3x235x363x4x56xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_BF16>, padding = [1, 1, 192, 1, 64, 24], promote_operands = [0, 1, 2], reduction = [0, 0, 0, 0, 0, 3], subgroup = [1, 1, 2, 1, 2, 0], subgroup_m_count = 3 : i64, subgroup_n_count = 1 : i64, workgroup = [1, 1, 192, 1, 64, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [192, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @"module-7272548824195366221_apply_op_config"(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_4x59x91x896_nhwc_896x3x3x56_fhwc_nhwf_1x1s_1x1p_1x1d_16g$async_dispatch_0_conv_4x59x91x16x56x3x3x56_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<4x61x93x16x56xbf16>, %arg2: tensor<16x56x3x3x56xbf16>, %arg3: tensor<4x59x91x16x56xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1 + d5, d2 + d6, d3, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d3, d4, d5, d6, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<4x61x93x16x56xbf16>, tensor<16x56x3x3x56xbf16>) outs(%arg3 : tensor<4x59x91x16x56xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<4x59x91x16x56xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_BF16>, padding = [2, 1, 96, 1, 64, 24], promote_operands = [0, 1, 2], reduction = [0, 0, 0, 0, 0, 3], subgroup = [2, 1, 1, 1, 2, 0], subgroup_m_count = 3 : i64, subgroup_n_count = 1 : i64, workgroup = [2, 1, 96, 1, 64, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [192, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module3767595398975031806_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_12x235x363x448_nhwc_448x3x3x56_fhwc_nhwf_2x2s_1x1p_1x1d_8g$async_dispatch_0_conv_12x118x182x8x56x3x3x56_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<12x237x365x8x56xbf16>, %arg2: tensor<8x56x3x3x56xbf16>, %arg3: tensor<12x118x182x8x56xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1 * 2 + d5, d2 * 2 + d6, d3, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d3, d4, d5, d6, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<12x237x365x8x56xbf16>, tensor<8x56x3x3x56xbf16>) outs(%arg3 : tensor<12x118x182x8x56xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<12x118x182x8x56xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_BF16>, padding = [1, 1, 192, 1, 64, 24], promote_operands = [0, 1, 2], reduction = [0, 0, 0, 0, 0, 3], subgroup = [1, 1, 2, 1, 2, 0], subgroup_m_count = 3 : i64, subgroup_n_count = 1 : i64, workgroup = [1, 1, 192, 1, 64, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [192, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module4768356428233192197_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_4x59x91x896_nhwc_896x1x1x896_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_21476x896x896_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<21476x896xbf16>, %arg2: tensor<896x896xbf16>, %arg3: tensor<21476x896xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<21476x896xbf16>, tensor<896x896xbf16>) outs(%arg3 : tensor<21476x896xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<21476x896xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_BF16>, padding = [64, 128, 64], promote_operands = [0, 1, 2], reduction = [0, 0, 8], subgroup = [1, 1, 0], subgroup_m_count = 2 : i64, subgroup_n_count = 4 : i64, workgroup = [64, 128, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [512, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @"module-8175233066150774447_apply_op_config"(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_5x118x182x448_nhwc_448x3x3x56_fhwc_nhwf_1x1s_1x1p_1x1d_8g$async_dispatch_0_conv_5x118x182x8x56x3x3x56_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<5x120x184x8x56xbf16>, %arg2: tensor<8x56x3x3x56xbf16>, %arg3: tensor<5x118x182x8x56xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1 + d5, d2 + d6, d3, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d3, d4, d5, d6, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<5x120x184x8x56xbf16>, tensor<8x56x3x3x56xbf16>) outs(%arg3 : tensor<5x118x182x8x56xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<5x118x182x8x56xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_BF16>, padding = [1, 1, 192, 1, 64, 24], promote_operands = [0, 1, 2], reduction = [0, 0, 0, 0, 0, 3], subgroup = [1, 1, 2, 1, 2, 0], subgroup_m_count = 3 : i64, subgroup_n_count = 1 : i64, workgroup = [1, 1, 192, 1, 64, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [192, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module5997569232756307197_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_3x235x363x224_nhwc_224x3x3x56_fhwc_nhwf_1x1s_1x1p_1x1d_4g$async_dispatch_0_conv_3x235x363x4x56x3x3x56_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<3x237x365x4x56xbf16>, %arg2: tensor<4x56x3x3x56xbf16>, %arg3: tensor<3x235x363x4x56xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1 + d5, d2 + d6, d3, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d3, d4, d5, d6, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<3x237x365x4x56xbf16>, tensor<4x56x3x3x56xbf16>) outs(%arg3 : tensor<3x235x363x4x56xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<3x235x363x4x56xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_BF16>, padding = [1, 1, 192, 1, 64, 24], promote_operands = [0, 1, 2], reduction = [0, 0, 0, 0, 0, 3], subgroup = [1, 1, 2, 1, 2, 0], subgroup_m_count = 3 : i64, subgroup_n_count = 1 : i64, workgroup = [1, 1, 192, 1, 64, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [192, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @"module-9157235887781922563_apply_op_config"(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_4x470x725x224_nhwc_224x3x3x56_fhwc_nhwf_2x2s_1x1p_1x1d_4g$async_dispatch_0_conv_4x235x363x4x56x3x3x56_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<4x472x727x4x56xbf16>, %arg2: tensor<4x56x3x3x56xbf16>, %arg3: tensor<4x235x363x4x56xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1 * 2 + d5, d2 * 2 + d6, d3, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d3, d4, d5, d6, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<4x472x727x4x56xbf16>, tensor<4x56x3x3x56xbf16>) outs(%arg3 : tensor<4x235x363x4x56xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<4x235x363x4x56xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_BF16>, padding = [1, 1, 192, 1, 64, 24], promote_operands = [0, 1, 2], reduction = [0, 0, 0, 0, 0, 3], subgroup = [1, 1, 2, 1, 2, 0], subgroup_m_count = 3 : i64, subgroup_n_count = 1 : i64, workgroup = [1, 1, 192, 1, 64, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [192, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module6208162720997979308_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_6x118x182x448_nhwc_448x3x3x56_fhwc_nhwf_1x1s_1x1p_1x1d_8g$async_dispatch_0_conv_6x118x182x8x56x3x3x56_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<6x120x184x8x56xbf16>, %arg2: tensor<8x56x3x3x56xbf16>, %arg3: tensor<6x118x182x8x56xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1 + d5, d2 + d6, d3, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d3, d4, d5, d6, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<6x120x184x8x56xbf16>, tensor<8x56x3x3x56xbf16>) outs(%arg3 : tensor<6x118x182x8x56xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<6x118x182x8x56xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_BF16>, padding = [1, 1, 192, 1, 64, 24], promote_operands = [0, 1, 2], reduction = [0, 0, 0, 0, 0, 3], subgroup = [1, 1, 2, 1, 2, 0], subgroup_m_count = 3 : i64, subgroup_n_count = 1 : i64, workgroup = [1, 1, 192, 1, 64, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [192, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @"module-8008214301341210612_apply_op_config"(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_5x59x91x896_nhwc_896x1x1x896_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_26845x896x896_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<26845x896xbf16>, %arg2: tensor<896x896xbf16>, %arg3: tensor<26845x896xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<26845x896xbf16>, tensor<896x896xbf16>) outs(%arg3 : tensor<26845x896xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<26845x896xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, padding = [32, 64, 64], promote_operands = [0, 1, 2], reduction = [0, 0, 4], subgroup = [1, 2, 0], subgroup_m_count = 2 : i64, subgroup_n_count = 2 : i64, workgroup = [32, 64, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [256, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @"module-4277437435457599128_apply_op_config"(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_5x470x725x224_nhwc_224x3x3x56_fhwc_nhwf_2x2s_1x1p_1x1d_4g$async_dispatch_0_conv_5x235x363x4x56x3x3x56_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<5x472x727x4x56xbf16>, %arg2: tensor<4x56x3x3x56xbf16>, %arg3: tensor<5x235x363x4x56xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1 * 2 + d5, d2 * 2 + d6, d3, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d3, d4, d5, d6, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<5x472x727x4x56xbf16>, tensor<4x56x3x3x56xbf16>) outs(%arg3 : tensor<5x235x363x4x56xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<5x235x363x4x56xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_BF16>, padding = [1, 1, 192, 1, 64, 24], promote_operands = [0, 1, 2], reduction = [0, 0, 0, 0, 0, 3], subgroup = [1, 1, 2, 1, 2, 0], subgroup_m_count = 3 : i64, subgroup_n_count = 1 : i64, workgroup = [1, 1, 192, 1, 64, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [192, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @"module-5567820896292441710_apply_op_config"(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_4x235x363x224_nhwc_224x3x3x56_fhwc_nhwf_1x1s_1x1p_1x1d_4g$async_dispatch_0_conv_4x235x363x4x56x3x3x56_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<4x237x365x4x56xbf16>, %arg2: tensor<4x56x3x3x56xbf16>, %arg3: tensor<4x235x363x4x56xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1 + d5, d2 + d6, d3, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d3, d4, d5, d6, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<4x237x365x4x56xbf16>, tensor<4x56x3x3x56xbf16>) outs(%arg3 : tensor<4x235x363x4x56xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<4x235x363x4x56xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_BF16>, padding = [1, 1, 192, 1, 64, 24], promote_operands = [0, 1, 2], reduction = [0, 0, 0, 0, 0, 3], subgroup = [1, 1, 2, 1, 2, 0], subgroup_m_count = 3 : i64, subgroup_n_count = 1 : i64, workgroup = [1, 1, 192, 1, 64, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [192, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @"module-4426083179023641467_apply_op_config"(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_7x118x182x448_nhwc_448x3x3x56_fhwc_nhwf_1x1s_1x1p_1x1d_8g$async_dispatch_0_conv_7x118x182x8x56x3x3x56_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<7x120x184x8x56xbf16>, %arg2: tensor<8x56x3x3x56xbf16>, %arg3: tensor<7x118x182x8x56xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1 + d5, d2 + d6, d3, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d3, d4, d5, d6, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<7x120x184x8x56xbf16>, tensor<8x56x3x3x56xbf16>) outs(%arg3 : tensor<7x118x182x8x56xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<7x118x182x8x56xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_BF16>, padding = [1, 1, 192, 1, 64, 24], promote_operands = [0, 1, 2], reduction = [0, 0, 0, 0, 0, 3], subgroup = [1, 1, 2, 1, 2, 0], subgroup_m_count = 3 : i64, subgroup_n_count = 1 : i64, workgroup = [1, 1, 192, 1, 64, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [192, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module7624307817249334277_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_6x59x91x896_nhwc_896x3x3x56_fhwc_nhwf_1x1s_1x1p_1x1d_16g$async_dispatch_0_conv_6x59x91x16x56x3x3x56_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<6x61x93x16x56xbf16>, %arg2: tensor<16x56x3x3x56xbf16>, %arg3: tensor<6x59x91x16x56xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1 + d5, d2 + d6, d3, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d3, d4, d5, d6, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<6x61x93x16x56xbf16>, tensor<16x56x3x3x56xbf16>) outs(%arg3 : tensor<6x59x91x16x56xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<6x59x91x16x56xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_BF16>, padding = [2, 1, 96, 1, 64, 24], promote_operands = [0, 1, 2], reduction = [0, 0, 0, 0, 0, 3], subgroup = [2, 1, 1, 1, 2, 0], subgroup_m_count = 3 : i64, subgroup_n_count = 1 : i64, workgroup = [2, 1, 96, 1, 64, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [192, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module8629613322876569104_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_6x470x725x224_nhwc_224x3x3x56_fhwc_nhwf_2x2s_1x1p_1x1d_4g$async_dispatch_0_conv_6x235x363x4x56x3x3x56_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<6x472x727x4x56xbf16>, %arg2: tensor<4x56x3x3x56xbf16>, %arg3: tensor<6x235x363x4x56xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1 * 2 + d5, d2 * 2 + d6, d3, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d3, d4, d5, d6, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<6x472x727x4x56xbf16>, tensor<4x56x3x3x56xbf16>) outs(%arg3 : tensor<6x235x363x4x56xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<6x235x363x4x56xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_BF16>, padding = [1, 1, 192, 1, 64, 24], promote_operands = [0, 1, 2], reduction = [0, 0, 0, 0, 0, 3], subgroup = [1, 1, 2, 1, 2, 0], subgroup_m_count = 3 : i64, subgroup_n_count = 1 : i64, workgroup = [1, 1, 192, 1, 64, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [192, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module5183625046126744491_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_5x235x363x224_nhwc_224x3x3x56_fhwc_nhwf_1x1s_1x1p_1x1d_4g$async_dispatch_0_conv_5x235x363x4x56x3x3x56_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<5x237x365x4x56xbf16>, %arg2: tensor<4x56x3x3x56xbf16>, %arg3: tensor<5x235x363x4x56xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1 + d5, d2 + d6, d3, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d3, d4, d5, d6, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<5x237x365x4x56xbf16>, tensor<4x56x3x3x56xbf16>) outs(%arg3 : tensor<5x235x363x4x56xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<5x235x363x4x56xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_BF16>, padding = [1, 1, 192, 1, 64, 24], promote_operands = [0, 1, 2], reduction = [0, 0, 0, 0, 0, 3], subgroup = [1, 1, 2, 1, 2, 0], subgroup_m_count = 3 : i64, subgroup_n_count = 1 : i64, workgroup = [1, 1, 192, 1, 64, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [192, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module4859907193630060386_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_7x470x725x224_nhwc_224x3x3x56_fhwc_nhwf_2x2s_1x1p_1x1d_4g$async_dispatch_0_conv_7x235x363x4x56x3x3x56_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<7x472x727x4x56xbf16>, %arg2: tensor<4x56x3x3x56xbf16>, %arg3: tensor<7x235x363x4x56xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1 * 2 + d5, d2 * 2 + d6, d3, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d3, d4, d5, d6, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<7x472x727x4x56xbf16>, tensor<4x56x3x3x56xbf16>) outs(%arg3 : tensor<7x235x363x4x56xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<7x235x363x4x56xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_BF16>, padding = [1, 1, 192, 1, 64, 24], promote_operands = [0, 1, 2], reduction = [0, 0, 0, 0, 0, 3], subgroup = [1, 1, 2, 1, 2, 0], subgroup_m_count = 3 : i64, subgroup_n_count = 1 : i64, workgroup = [1, 1, 192, 1, 64, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [192, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module7886996018887940867_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_10x118x182x448_nhwc_448x3x3x56_fhwc_nhwf_1x1s_1x1p_1x1d_8g$async_dispatch_0_conv_10x118x182x8x56x3x3x56_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<10x120x184x8x56xbf16>, %arg2: tensor<8x56x3x3x56xbf16>, %arg3: tensor<10x118x182x8x56xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1 + d5, d2 + d6, d3, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d3, d4, d5, d6, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<10x120x184x8x56xbf16>, tensor<8x56x3x3x56xbf16>) outs(%arg3 : tensor<10x118x182x8x56xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<10x118x182x8x56xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_BF16>, padding = [1, 1, 192, 1, 64, 24], promote_operands = [0, 1, 2], reduction = [0, 0, 0, 0, 0, 3], subgroup = [1, 1, 2, 1, 2, 0], subgroup_m_count = 3 : i64, subgroup_n_count = 1 : i64, workgroup = [1, 1, 192, 1, 64, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [192, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module1542749415825119683_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_6x235x363x224_nhwc_224x3x3x56_fhwc_nhwf_1x1s_1x1p_1x1d_4g$async_dispatch_0_conv_6x235x363x4x56x3x3x56_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<6x237x365x4x56xbf16>, %arg2: tensor<4x56x3x3x56xbf16>, %arg3: tensor<6x235x363x4x56xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1 + d5, d2 + d6, d3, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d3, d4, d5, d6, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<6x237x365x4x56xbf16>, tensor<4x56x3x3x56xbf16>) outs(%arg3 : tensor<6x235x363x4x56xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<6x235x363x4x56xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_BF16>, padding = [1, 1, 192, 1, 64, 24], promote_operands = [0, 1, 2], reduction = [0, 0, 0, 0, 0, 3], subgroup = [1, 1, 2, 1, 2, 0], subgroup_m_count = 3 : i64, subgroup_n_count = 1 : i64, workgroup = [1, 1, 192, 1, 64, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [192, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module2279074681818677748_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_10x470x725x224_nhwc_224x3x3x56_fhwc_nhwf_2x2s_1x1p_1x1d_4g$async_dispatch_0_conv_10x235x363x4x56x3x3x56_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<10x472x727x4x56xbf16>, %arg2: tensor<4x56x3x3x56xbf16>, %arg3: tensor<10x235x363x4x56xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1 * 2 + d5, d2 * 2 + d6, d3, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d3, d4, d5, d6, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<10x472x727x4x56xbf16>, tensor<4x56x3x3x56xbf16>) outs(%arg3 : tensor<10x235x363x4x56xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<10x235x363x4x56xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_BF16>, padding = [1, 1, 192, 1, 64, 24], promote_operands = [0, 1, 2], reduction = [0, 0, 0, 0, 0, 3], subgroup = [1, 1, 2, 1, 2, 0], subgroup_m_count = 3 : i64, subgroup_n_count = 1 : i64, workgroup = [1, 1, 192, 1, 64, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [192, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module3648899047666359245_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_12x470x725x224_nhwc_224x3x3x56_fhwc_nhwf_2x2s_1x1p_1x1d_4g$async_dispatch_0_conv_12x235x363x4x56x3x3x56_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<12x472x727x4x56xbf16>, %arg2: tensor<4x56x3x3x56xbf16>, %arg3: tensor<12x235x363x4x56xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1 * 2 + d5, d2 * 2 + d6, d3, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d3, d4, d5, d6, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<12x472x727x4x56xbf16>, tensor<4x56x3x3x56xbf16>) outs(%arg3 : tensor<12x235x363x4x56xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<12x235x363x4x56xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_BF16>, padding = [1, 1, 192, 1, 64, 24], promote_operands = [0, 1, 2], reduction = [0, 0, 0, 0, 0, 3], subgroup = [1, 1, 2, 1, 2, 0], subgroup_m_count = 3 : i64, subgroup_n_count = 1 : i64, workgroup = [1, 1, 192, 1, 64, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [192, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @"module-5380161393819990948_apply_op_config"(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_7x235x363x224_nhwc_224x3x3x56_fhwc_nhwf_1x1s_1x1p_1x1d_4g$async_dispatch_0_conv_7x235x363x4x56x3x3x56_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<7x237x365x4x56xbf16>, %arg2: tensor<4x56x3x3x56xbf16>, %arg3: tensor<7x235x363x4x56xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1 + d5, d2 + d6, d3, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d3, d4, d5, d6, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<7x237x365x4x56xbf16>, tensor<4x56x3x3x56xbf16>) outs(%arg3 : tensor<7x235x363x4x56xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<7x235x363x4x56xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_BF16>, padding = [1, 1, 192, 1, 64, 24], promote_operands = [0, 1, 2], reduction = [0, 0, 0, 0, 0, 3], subgroup = [1, 1, 2, 1, 2, 0], subgroup_m_count = 3 : i64, subgroup_n_count = 1 : i64, workgroup = [1, 1, 192, 1, 64, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [192, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @"module-3129989938171215199_apply_op_config"(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_12x118x182x448_nhwc_448x3x3x56_fhwc_nhwf_1x1s_1x1p_1x1d_8g$async_dispatch_0_conv_12x118x182x8x56x3x3x56_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<12x120x184x8x56xbf16>, %arg2: tensor<8x56x3x3x56xbf16>, %arg3: tensor<12x118x182x8x56xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1 + d5, d2 + d6, d3, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d3, d4, d5, d6, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<12x120x184x8x56xbf16>, tensor<8x56x3x3x56xbf16>) outs(%arg3 : tensor<12x118x182x8x56xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<12x118x182x8x56xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_BF16>, padding = [1, 1, 192, 1, 64, 24], promote_operands = [0, 1, 2], reduction = [0, 0, 0, 0, 0, 3], subgroup = [1, 1, 2, 1, 2, 0], subgroup_m_count = 3 : i64, subgroup_n_count = 1 : i64, workgroup = [1, 1, 192, 1, 64, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [192, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module5805262495997135865_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_10x235x363x224_nhwc_224x3x3x56_fhwc_nhwf_1x1s_1x1p_1x1d_4g$async_dispatch_0_conv_10x235x363x4x56x3x3x56_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<10x237x365x4x56xbf16>, %arg2: tensor<4x56x3x3x56xbf16>, %arg3: tensor<10x235x363x4x56xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1 + d5, d2 + d6, d3, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d3, d4, d5, d6, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<10x237x365x4x56xbf16>, tensor<4x56x3x3x56xbf16>) outs(%arg3 : tensor<10x235x363x4x56xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<10x235x363x4x56xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_BF16>, padding = [1, 1, 192, 1, 64, 24], promote_operands = [0, 1, 2], reduction = [0, 0, 0, 0, 0, 3], subgroup = [1, 1, 2, 1, 2, 0], subgroup_m_count = 3 : i64, subgroup_n_count = 1 : i64, workgroup = [1, 1, 192, 1, 64, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [192, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @"module-1992552252842254751_apply_op_config"(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_12x235x363x224_nhwc_224x3x3x56_fhwc_nhwf_1x1s_1x1p_1x1d_4g$async_dispatch_0_conv_12x235x363x4x56x3x3x56_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<12x237x365x4x56xbf16>, %arg2: tensor<4x56x3x3x56xbf16>, %arg3: tensor<12x235x363x4x56xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1 + d5, d2 + d6, d3, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d3, d4, d5, d6, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<12x237x365x4x56xbf16>, tensor<4x56x3x3x56xbf16>) outs(%arg3 : tensor<12x235x363x4x56xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<12x235x363x4x56xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_BF16>, padding = [1, 1, 192, 1, 64, 24], promote_operands = [0, 1, 2], reduction = [0, 0, 0, 0, 0, 3], subgroup = [1, 1, 2, 1, 2, 0], subgroup_m_count = 3 : i64, subgroup_n_count = 1 : i64, workgroup = [1, 1, 192, 1, 64, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [192, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module950986832807159606_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_7x59x91x896_nhwc_896x1x1x896_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_37583x896x896_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<37583x896xbf16>, %arg2: tensor<896x896xbf16>, %arg3: tensor<37583x896xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<37583x896xbf16>, tensor<896x896xbf16>) outs(%arg3 : tensor<37583x896xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<37583x896xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, padding = [48, 64, 64], promote_operands = [0, 1, 2], reduction = [0, 0, 4], subgroup = [3, 2, 0], subgroup_m_count = 1 : i64, subgroup_n_count = 2 : i64, workgroup = [48, 64, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [128, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @"module-5482690283594198306_apply_op_config"(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_10x59x91x896_nhwc_896x3x3x56_fhwc_nhwf_1x1s_1x1p_1x1d_16g$async_dispatch_0_conv_10x59x91x16x56x3x3x56_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<10x61x93x16x56xbf16>, %arg2: tensor<16x56x3x3x56xbf16>, %arg3: tensor<10x59x91x16x56xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1 + d5, d2 + d6, d3, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d3, d4, d5, d6, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<10x61x93x16x56xbf16>, tensor<16x56x3x3x56xbf16>) outs(%arg3 : tensor<10x59x91x16x56xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<10x59x91x16x56xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_BF16>, padding = [2, 1, 96, 1, 64, 24], promote_operands = [0, 1, 2], reduction = [0, 0, 0, 0, 0, 3], subgroup = [2, 1, 1, 1, 2, 0], subgroup_m_count = 3 : i64, subgroup_n_count = 1 : i64, workgroup = [2, 1, 96, 1, 64, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [192, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module4494627814173524847_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_12x59x91x896_nhwc_896x3x3x56_fhwc_nhwf_1x1s_1x1p_1x1d_16g$async_dispatch_0_conv_12x59x91x16x56x3x3x56_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<12x61x93x16x56xbf16>, %arg2: tensor<16x56x3x3x56xbf16>, %arg3: tensor<12x59x91x16x56xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1 + d5, d2 + d6, d3, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d3, d4, d5, d6, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<12x61x93x16x56xbf16>, tensor<16x56x3x3x56xbf16>) outs(%arg3 : tensor<12x59x91x16x56xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<12x59x91x16x56xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_BF16>, padding = [2, 1, 96, 1, 64, 24], promote_operands = [0, 1, 2], reduction = [0, 0, 0, 0, 0, 3], subgroup = [2, 1, 1, 1, 2, 0], subgroup_m_count = 3 : i64, subgroup_n_count = 1 : i64, workgroup = [2, 1, 96, 1, 64, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [192, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @"module-7113439279041897089_apply_op_config"(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_10x59x91x896_nhwc_896x1x1x896_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_53690x896x896_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<53690x896xbf16>, %arg2: tensor<896x896xbf16>, %arg3: tensor<53690x896xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<53690x896xbf16>, tensor<896x896xbf16>) outs(%arg3 : tensor<53690x896xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<53690x896xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, padding = [64, 128, 64], promote_operands = [0, 1, 2], reduction = [0, 0, 4], subgroup = [2, 2, 0], subgroup_m_count = 2 : i64, subgroup_n_count = 4 : i64, workgroup = [64, 128, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [512, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module6158197132717631226_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_12x59x91x896_nhwc_896x1x1x896_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_64428x896x896_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<64428x896xbf16>, %arg2: tensor<896x896xbf16>, %arg3: tensor<64428x896xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<64428x896xbf16>, tensor<896x896xbf16>) outs(%arg3 : tensor<64428x896xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<64428x896xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_BF16>, padding = [64, 128, 64], promote_operands = [0, 1, 2], reduction = [0, 0, 8], subgroup = [1, 1, 0], subgroup_m_count = 2 : i64, subgroup_n_count = 4 : i64, workgroup = [64, 128, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [512, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @"module-5610834851670606643_apply_op_config"(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_16x24x16x288_nhwc_288x3x1x96_fhwc_nhwf_1x1s_1x0p_1x1d_3g$async_dispatch_0_conv_16x24x16x3x96x3x96_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<16x26x16x3x96xbf16>, %arg2: tensor<3x96x3x96xbf16>, %arg3: tensor<16x24x16x3x96xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d5, d2, d3, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d3, d4, d5, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<16x26x16x3x96xbf16>, tensor<3x96x3x96xbf16>) outs(%arg3 : tensor<16x24x16x3x96xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<16x24x16x3x96xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, promote_operands = [0, 1], reduction = [0, 0, 0, 0, 0, 6], subgroup = [2, 1, 1, 1, 2, 0], subgroup_m_count = 2 : i64, subgroup_n_count = 3 : i64, workgroup = [2, 2, 16, 1, 96, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [384, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @"module-8838592581327294770_apply_op_config"(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_128x24x48x384_nhwc_512x1x1x384_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_147456x512x384_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<147456x384xbf16>, %arg2: tensor<512x384xbf16>, %arg3: tensor<147456x512xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<147456x384xbf16>, tensor<512x384xbf16>) outs(%arg3 : tensor<147456x512xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<147456x512xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_BF16>, promote_operands = [0, 1], reduction = [0, 0, 4], subgroup = [2, 2, 0], subgroup_m_count = 2 : i64, subgroup_n_count = 4 : i64, workgroup = [128, 256, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [512, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @"module-4092354048484454229_apply_op_config"(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_16x96x64x96_nhwc_96x1x1x96_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_98304x96x96_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<98304x96xbf16>, %arg2: tensor<96x96xbf16>, %arg3: tensor<98304x96xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<98304x96xbf16>, tensor<96x96xbf16>) outs(%arg3 : tensor<98304x96xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<98304x96xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, promote_operands = [0, 1], reduction = [0, 0, 6], subgroup = [3, 1, 0], subgroup_m_count = 1 : i64, subgroup_n_count = 3 : i64, workgroup = [48, 48, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [192, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module2689198279833294263_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_16x48x32x48_nhwc_48x5x5x48_fhwc_nhwf_1x1s_8x8p_4x4d_1g$async_dispatch_0_conv_16x48x32x48x5x5x48_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<16x64x48x48xbf16>, %arg2: tensor<48x5x5x48xbf16>, %arg3: tensor<16x48x32x48xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d4 * 4, d2 + d5 * 4, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d3, d4, d5, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<16x64x48x48xbf16>, tensor<48x5x5x48xbf16>) outs(%arg3 : tensor<16x48x32x48xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<16x48x32x48xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_BF16>, padding = [1, 2, 32, 64, 48], promote_operands = [0, 1, 2], reduction = [0, 0, 0, 0, 6], subgroup = [1, 2, 1, 1, 0], subgroup_m_count = 1 : i64, subgroup_n_count = 2 : i64, workgroup = [1, 2, 32, 64, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [128, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module3994070714556536465_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_16x48x32x576_nhwc_576x1x1x576_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_24576x576x576_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<24576x576xbf16>, %arg2: tensor<576x576xbf16>, %arg3: tensor<24576x576xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<24576x576xbf16>, tensor<576x576xbf16>) outs(%arg3 : tensor<24576x576xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<24576x576xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, promote_operands = [0, 1], reduction = [0, 0, 4], subgroup = [4, 6, 0], subgroup_m_count = 4 : i64, subgroup_n_count = 2 : i64, workgroup = [256, 192, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [512, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module5276668663876473507_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_16x48x32x480_nhwc_480x1x1x96_fhwc_nhwf_1x1s_0x0p_1x1d_5g$async_dispatch_0_matmul_like_16x48x32x5x96x96_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<16x48x32x5x96xbf16>, %arg2: tensor<5x96x96xbf16>, %arg3: tensor<16x48x32x5x96xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d4, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<16x48x32x5x96xbf16>, tensor<5x96x96xbf16>) outs(%arg3 : tensor<16x48x32x5x96xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<16x48x32x5x96xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, promote_operands = [0, 1], reduction = [0, 0, 0, 0, 0, 2], subgroup = [4, 1, 1, 1, 3, 0], subgroup_m_count = 1 : i64, subgroup_n_count = 2 : i64, workgroup = [4, 1, 16, 1, 96, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [128, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @"module-4767593926805822391_apply_op_config"(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_128x24x48x480_nhwc_384x1x1x480_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_147456x384x480_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<147456x480xbf16>, %arg2: tensor<384x480xbf16>, %arg3: tensor<147456x384xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<147456x480xbf16>, tensor<384x480xbf16>) outs(%arg3 : tensor<147456x384xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<147456x384xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_BF16>, promote_operands = [0, 1], reduction = [0, 0, 4], subgroup = [4, 3, 0], subgroup_m_count = 2 : i64, subgroup_n_count = 4 : i64, workgroup = [256, 384, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [512, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @"module-1608284571499770249_apply_op_config"(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_3d_bfloat16_forward_b_16x2x48x32x288_ndhwc_288x3x1x1x96_fdhwc_ndhwf_1x1x1s_1x0x0p_1x1x1d_3g$async_dispatch_0_conv_16x2x48x32x3x96x3x96_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<16x4x48x32x3x96xbf16>, %arg2: tensor<3x96x3x96xbf16>, %arg3: tensor<16x2x48x32x3x96xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1 + d6, d2, d3, d4, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d4, d5, d6, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<16x4x48x32x3x96xbf16>, tensor<3x96x3x96xbf16>) outs(%arg3 : tensor<16x2x48x32x3x96xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<16x2x48x32x3x96xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, promote_operands = [0, 1], reduction = [0, 0, 0, 0, 0, 0, 2], subgroup = [1, 1, 3, 2, 1, 2, 0], subgroup_m_count = 1 : i64, subgroup_n_count = 3 : i64, workgroup = [1, 1, 3, 32, 1, 96, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [192, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @"module-464855148765289941_apply_op_config"(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_16x48x32x672_nhwc_576x1x1x672_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_24576x576x672_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<24576x672xbf16>, %arg2: tensor<576x672xbf16>, %arg3: tensor<24576x576xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<24576x672xbf16>, tensor<576x672xbf16>) outs(%arg3 : tensor<24576x576xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<24576x576xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, promote_operands = [0, 1], reduction = [0, 0, 2], subgroup = [2, 12, 0], subgroup_m_count = 4 : i64, subgroup_n_count = 1 : i64, workgroup = [128, 192, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [256, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module4378881155291204668_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_16x96x64x96_nhwc_96x3x1x96_fhwc_nhwf_1x1s_1x0p_1x1d_1g$async_dispatch_0_conv_16x96x64x96x3x96_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<16x98x64x96xbf16>, %arg2: tensor<96x3x96xbf16>, %arg3: tensor<16x96x64x96xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1 + d4, d2, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d4, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<16x98x64x96xbf16>, tensor<96x3x96xbf16>) outs(%arg3 : tensor<16x96x64x96xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<16x96x64x96xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, promote_operands = [0, 1], reduction = [0, 0, 0, 0, 2], subgroup = [2, 3, 1, 2, 0], subgroup_m_count = 2 : i64, subgroup_n_count = 3 : i64, workgroup = [2, 6, 16, 96, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [384, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @"module-763959733402495440_apply_op_config"(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_16x96x64x48_nhwc_48x3x3x48_fhwc_nhwf_1x1s_1x1p_1x1d_1g$async_dispatch_0_conv_16x96x64x48x3x3x48_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<16x98x66x48xbf16>, %arg2: tensor<48x3x3x48xbf16>, %arg3: tensor<16x96x64x48xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d4, d2 + d5, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d3, d4, d5, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<16x98x66x48xbf16>, tensor<48x3x3x48xbf16>) outs(%arg3 : tensor<16x96x64x48xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<16x96x64x48xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_BF16>, padding = [1, 1, 64, 64, 16], promote_operands = [0, 1, 2], reduction = [0, 0, 0, 0, 2], subgroup = [1, 1, 1, 2, 0], subgroup_m_count = 2 : i64, subgroup_n_count = 1 : i64, workgroup = [1, 1, 64, 64, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [128, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module2066699225626929295_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_16x96x64x192_nhwc_96x1x1x192_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_98304x96x192_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<98304x192xbf16>, %arg2: tensor<96x192xbf16>, %arg3: tensor<98304x96xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<98304x192xbf16>, tensor<96x192xbf16>) outs(%arg3 : tensor<98304x96xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<98304x96xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, promote_operands = [0, 1], reduction = [0, 0, 2], subgroup = [2, 6, 0], subgroup_m_count = 6 : i64, subgroup_n_count = 1 : i64, workgroup = [192, 96, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [384, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module7350505846474122665_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_16x96x64x48_nhwc_48x3x3x48_fhwc_nhwf_1x1s_4x4p_4x4d_1g$async_dispatch_0_conv_16x96x64x48x3x3x48_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<16x104x72x48xbf16>, %arg2: tensor<48x3x3x48xbf16>, %arg3: tensor<16x96x64x48xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d4 * 4, d2 + d5 * 4, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d3, d4, d5, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<16x104x72x48xbf16>, tensor<48x3x3x48xbf16>) outs(%arg3 : tensor<16x96x64x48xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<16x96x64x48xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_BF16>, padding = [1, 1, 64, 64, 48], promote_operands = [0, 1, 2], reduction = [0, 0, 0, 0, 6], subgroup = [1, 1, 2, 1, 0], subgroup_m_count = 1 : i64, subgroup_n_count = 2 : i64, workgroup = [1, 1, 64, 64, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [128, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @"module-7530347551705388041_apply_op_config"(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_3d_bfloat16_forward_b_16x4x48x32x288_ndhwc_288x3x1x1x96_fdhwc_ndhwf_1x1x1s_1x0x0p_1x1x1d_3g$async_dispatch_0_conv_16x4x48x32x3x96x3x96_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<16x6x48x32x3x96xbf16>, %arg2: tensor<3x96x3x96xbf16>, %arg3: tensor<16x4x48x32x3x96xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1 + d6, d2, d3, d4, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d4, d5, d6, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<16x6x48x32x3x96xbf16>, tensor<3x96x3x96xbf16>) outs(%arg3 : tensor<16x4x48x32x3x96xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<16x4x48x32x3x96xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, promote_operands = [0, 1], reduction = [0, 0, 0, 0, 0, 0, 2], subgroup = [1, 1, 2, 2, 1, 6, 0], subgroup_m_count = 3 : i64, subgroup_n_count = 1 : i64, workgroup = [1, 1, 6, 32, 1, 96, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [192, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @"module-1970581355946039598_apply_op_config"(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_3d_bfloat16_forward_b_16x2x48x32x288_ndhwc_288x1x3x3x96_fdhwc_ndhwf_1x1x1s_0x1x1p_1x1x1d_3g$async_dispatch_0_conv_16x2x48x32x3x96x3x3x96_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<16x2x50x34x3x96xbf16>, %arg2: tensor<3x96x3x3x96xbf16>, %arg3: tensor<16x2x48x32x3x96xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d2 + d6, d3 + d7, d4, d8)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d4, d5, d6, d7, d8)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<16x2x50x34x3x96xbf16>, tensor<3x96x3x3x96xbf16>) outs(%arg3 : tensor<16x2x48x32x3x96xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<16x2x48x32x3x96xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, promote_operands = [0, 1], reduction = [0, 0, 0, 0, 0, 0, 2], subgroup = [4, 1, 2, 1, 1, 3, 0], subgroup_m_count = 1 : i64, subgroup_n_count = 2 : i64, workgroup = [4, 1, 2, 16, 1, 96, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [128, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module3276315160131382713_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_16x96x64x48_nhwc_48x3x3x48_fhwc_nhwf_1x1s_2x2p_2x2d_1g$async_dispatch_0_conv_16x96x64x48x3x3x48_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<16x100x68x48xbf16>, %arg2: tensor<48x3x3x48xbf16>, %arg3: tensor<16x96x64x48xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d4 * 2, d2 + d5 * 2, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d3, d4, d5, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<16x100x68x48xbf16>, tensor<48x3x3x48xbf16>) outs(%arg3 : tensor<16x96x64x48xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<16x96x64x48xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_BF16>, padding = [1, 2, 32, 64, 16], promote_operands = [0, 1, 2], reduction = [0, 0, 0, 0, 2], subgroup = [1, 1, 1, 2, 0], subgroup_m_count = 2 : i64, subgroup_n_count = 1 : i64, workgroup = [1, 2, 32, 64, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [128, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @"module-3100570618954513192_apply_op_config"(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_3d_bfloat16_forward_16x2x48x32x288_ndhwc_288x2x1x1x288_fdhwc_ndhwf_2x1x1s_0x0x0p_1x1x1d_1g$async_dispatch_0_matmul_like_16x48x32x288x2x288_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<16x2x48x32x288xbf16>, %arg2: tensor<288x2x288xbf16>, %arg3: tensor<16x48x32x288xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d4, d1, d2, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d4, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<16x2x48x32x288xbf16>, tensor<288x2x288xbf16>) outs(%arg3 : tensor<16x48x32x288xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<16x48x32x288xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, promote_operands = [0, 1], reduction = [0, 0, 0, 0, 2, 2], subgroup = [1, 1, 2, 6, 0, 0], subgroup_m_count = 4 : i64, subgroup_n_count = 1 : i64, workgroup = [4, 1, 32, 96, 0, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [256, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @"module-5016717600198604863_apply_op_config"(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_16x96x64x96_nhwc_96x3x3x96_fhwc_nhwf_1x1s_1x1p_1x1d_1g$async_dispatch_0_conv_16x96x64x96x3x3x96_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<16x98x66x96xbf16>, %arg2: tensor<96x3x3x96xbf16>, %arg3: tensor<16x96x64x96xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d4, d2 + d5, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d3, d4, d5, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<16x98x66x96xbf16>, tensor<96x3x3x96xbf16>) outs(%arg3 : tensor<16x96x64x96xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<16x96x64x96xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_BF16>, promote_operands = [0, 1], reduction = [0, 0, 0, 0, 4], subgroup = [1, 3, 1, 1, 0], subgroup_m_count = 2 : i64, subgroup_n_count = 3 : i64, workgroup = [1, 6, 32, 96, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [384, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module4561946264403035130_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_128x24x48x384_nhwc_384x3x1x128_fhwc_nhwf_1x1s_1x0p_1x1d_3g$async_dispatch_0_conv_128x24x48x3x128x3x128_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<128x26x48x3x128xbf16>, %arg2: tensor<3x128x3x128xbf16>, %arg3: tensor<128x24x48x3x128xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d5, d2, d3, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d3, d4, d5, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<128x26x48x3x128xbf16>, tensor<3x128x3x128xbf16>) outs(%arg3 : tensor<128x24x48x3x128xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<128x24x48x3x128xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, promote_operands = [0, 1], reduction = [0, 0, 0, 0, 0, 4], subgroup = [1, 3, 1, 1, 4, 0], subgroup_m_count = 2 : i64, subgroup_n_count = 2 : i64, workgroup = [2, 3, 16, 1, 128, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [256, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @"module-6425628932703912647_apply_op_config"(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_3d_bfloat16_forward_b_16x8x48x32x288_ndhwc_288x3x1x1x96_fdhwc_ndhwf_1x1x1s_1x0x0p_1x1x1d_3g$async_dispatch_0_conv_16x8x48x32x3x96x3x96_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<16x10x48x32x3x96xbf16>, %arg2: tensor<3x96x3x96xbf16>, %arg3: tensor<16x8x48x32x3x96xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1 + d6, d2, d3, d4, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d4, d5, d6, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<16x10x48x32x3x96xbf16>, tensor<3x96x3x96xbf16>) outs(%arg3 : tensor<16x8x48x32x3x96xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<16x8x48x32x3x96xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_BF16>, promote_operands = [0, 1], reduction = [0, 0, 0, 0, 0, 0, 4], subgroup = [1, 1, 1, 1, 1, 3, 0], subgroup_m_count = 6 : i64, subgroup_n_count = 1 : i64, workgroup = [1, 1, 6, 32, 1, 96, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [384, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module6909153255479619441_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_3d_bfloat16_forward_16x2x48x32x288_ndhwc_288x3x1x1x96_fdhwc_ndhwf_1x1x1s_1x0x0p_1x1x1d_3g$async_dispatch_0_conv_16x2x48x32x3x96x3x96_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<16x4x48x32x3x96xbf16>, %arg2: tensor<3x96x3x96xbf16>, %arg3: tensor<16x2x48x32x3x96xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1 + d6, d2, d3, d4, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d4, d5, d6, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<16x4x48x32x3x96xbf16>, tensor<3x96x3x96xbf16>) outs(%arg3 : tensor<16x2x48x32x3x96xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<16x2x48x32x3x96xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, promote_operands = [0, 1], reduction = [0, 0, 0, 0, 0, 0, 2], subgroup = [4, 1, 1, 2, 1, 3, 0], subgroup_m_count = 1 : i64, subgroup_n_count = 2 : i64, workgroup = [4, 1, 1, 32, 1, 96, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [128, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module397248974270796247_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_3d_bfloat16_forward_b_16x4x48x32x288_ndhwc_288x1x3x3x96_fdhwc_ndhwf_1x1x1s_0x1x1p_1x1x1d_3g$async_dispatch_0_conv_16x4x48x32x3x96x3x3x96_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<16x4x50x34x3x96xbf16>, %arg2: tensor<3x96x3x3x96xbf16>, %arg3: tensor<16x4x48x32x3x96xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d2 + d6, d3 + d7, d4, d8)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d4, d5, d6, d7, d8)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<16x4x50x34x3x96xbf16>, tensor<3x96x3x3x96xbf16>) outs(%arg3 : tensor<16x4x48x32x3x96xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<16x4x48x32x3x96xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, promote_operands = [0, 1], reduction = [0, 0, 0, 0, 0, 0, 2], subgroup = [1, 2, 2, 2, 1, 3, 0], subgroup_m_count = 1 : i64, subgroup_n_count = 2 : i64, workgroup = [1, 2, 2, 32, 1, 96, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [128, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module7216368369122952845_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_128x24x48x384_nhwc_384x1x1x384_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_147456x384x384_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<147456x384xbf16>, %arg2: tensor<384x384xbf16>, %arg3: tensor<147456x384xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<147456x384xbf16>, tensor<384x384xbf16>) outs(%arg3 : tensor<147456x384xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<147456x384xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_BF16>, promote_operands = [0, 1], reduction = [0, 0, 4], subgroup = [2, 6, 0], subgroup_m_count = 4 : i64, subgroup_n_count = 2 : i64, workgroup = [256, 384, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [512, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @"module-4520889304652789583_apply_op_config"(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_16x96x64x48_nhwc_48x5x5x48_fhwc_nhwf_1x1s_8x8p_4x4d_1g$async_dispatch_0_conv_16x96x64x48x5x5x48_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<16x112x80x48xbf16>, %arg2: tensor<48x5x5x48xbf16>, %arg3: tensor<16x96x64x48xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d4 * 4, d2 + d5 * 4, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d3, d4, d5, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<16x112x80x48xbf16>, tensor<48x5x5x48xbf16>) outs(%arg3 : tensor<16x96x64x48xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<16x96x64x48xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_BF16>, padding = [1, 2, 32, 64, 48], promote_operands = [0, 1, 2], reduction = [0, 0, 0, 0, 6], subgroup = [1, 2, 1, 1, 0], subgroup_m_count = 1 : i64, subgroup_n_count = 2 : i64, workgroup = [1, 2, 32, 64, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [128, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module5408025256171335551_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_3d_bfloat16_forward_16x4x48x32x288_ndhwc_288x2x1x1x288_fdhwc_ndhwf_2x1x1s_0x0x0p_1x1x1d_1g$async_dispatch_0_conv_16x2x48x32x288x2x288_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<16x4x48x32x288xbf16>, %arg2: tensor<288x2x288xbf16>, %arg3: tensor<16x2x48x32x288xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 * 2 + d5, d2, d3, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d5, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<16x4x48x32x288xbf16>, tensor<288x2x288xbf16>) outs(%arg3 : tensor<16x2x48x32x288xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<16x2x48x32x288xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_BF16>, promote_operands = [0, 1], reduction = [0, 0, 0, 0, 0, 4], subgroup = [1, 1, 1, 1, 9, 0], subgroup_m_count = 6 : i64, subgroup_n_count = 1 : i64, workgroup = [1, 1, 6, 32, 288, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [384, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @"module-2215561608523455322_apply_op_config"(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_3d_bfloat16_forward_16x4x48x32x288_ndhwc_288x3x1x1x96_fdhwc_ndhwf_1x1x1s_1x0x0p_1x1x1d_3g$async_dispatch_0_conv_16x4x48x32x3x96x3x96_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<16x6x48x32x3x96xbf16>, %arg2: tensor<3x96x3x96xbf16>, %arg3: tensor<16x4x48x32x3x96xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1 + d6, d2, d3, d4, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d4, d5, d6, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<16x6x48x32x3x96xbf16>, tensor<3x96x3x96xbf16>) outs(%arg3 : tensor<16x4x48x32x3x96xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<16x4x48x32x3x96xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, promote_operands = [0, 1], reduction = [0, 0, 0, 0, 0, 0, 2], subgroup = [2, 1, 4, 1, 1, 3, 0], subgroup_m_count = 1 : i64, subgroup_n_count = 2 : i64, workgroup = [2, 1, 4, 16, 1, 96, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [128, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @"module-5385892577352669266_apply_op_config"(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_3d_bfloat16_forward_b_16x8x48x32x288_ndhwc_288x1x3x3x96_fdhwc_ndhwf_1x1x1s_0x1x1p_1x1x1d_3g$async_dispatch_0_conv_16x8x48x32x3x96x3x3x96_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<16x8x50x34x3x96xbf16>, %arg2: tensor<3x96x3x3x96xbf16>, %arg3: tensor<16x8x48x32x3x96xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d2 + d6, d3 + d7, d4, d8)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d4, d5, d6, d7, d8)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<16x8x50x34x3x96xbf16>, tensor<3x96x3x3x96xbf16>) outs(%arg3 : tensor<16x8x48x32x3x96xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<16x8x48x32x3x96xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, promote_operands = [0, 1], reduction = [0, 0, 0, 0, 0, 0, 2], subgroup = [1, 1, 3, 1, 1, 6, 0], subgroup_m_count = 2 : i64, subgroup_n_count = 1 : i64, workgroup = [1, 1, 6, 16, 1, 96, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [128, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module3515167887415924444_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_16x48x32x576_nhwc_576x3x3x576_fhwc_nhwf_1x1s_1x1p_1x1d_1g$async_dispatch_0_conv_16x48x32x576x3x3x576_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<16x50x34x576xbf16>, %arg2: tensor<576x3x3x576xbf16>, %arg3: tensor<16x48x32x576xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d4, d2 + d5, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d3, d4, d5, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<16x50x34x576xbf16>, tensor<576x3x3x576xbf16>) outs(%arg3 : tensor<16x48x32x576xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<16x48x32x576xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, promote_operands = [0, 1], reduction = [0, 0, 0, 0, 2], subgroup = [8, 1, 1, 3, 0], subgroup_m_count = 1 : i64, subgroup_n_count = 4 : i64, workgroup = [8, 1, 16, 192, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [256, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module1669037280268302384_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_128x48x32x384_nhwc_96x1x1x384_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_196608x96x384_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<196608x384xbf16>, %arg2: tensor<96x384xbf16>, %arg3: tensor<196608x96xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<196608x384xbf16>, tensor<96x384xbf16>) outs(%arg3 : tensor<196608x96xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<196608x96xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, promote_operands = [0, 1], reduction = [0, 0, 4], subgroup = [3, 3, 0], subgroup_m_count = 2 : i64, subgroup_n_count = 2 : i64, workgroup = [96, 96, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [256, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module1291937590125293741_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_3d_bfloat16_forward_16x2x48x32x288_ndhwc_288x1x3x3x96_fdhwc_ndhwf_1x1x1s_0x1x1p_1x1x1d_3g$async_dispatch_0_conv_16x2x48x32x3x96x3x3x96_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<16x2x50x34x3x96xbf16>, %arg2: tensor<3x96x3x3x96xbf16>, %arg3: tensor<16x2x48x32x3x96xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d2 + d6, d3 + d7, d4, d8)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d4, d5, d6, d7, d8)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<16x2x50x34x3x96xbf16>, tensor<3x96x3x3x96xbf16>) outs(%arg3 : tensor<16x2x48x32x3x96xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<16x2x48x32x3x96xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_BF16>, promote_operands = [0, 1], reduction = [0, 0, 0, 0, 0, 0, 4], subgroup = [2, 1, 1, 1, 1, 3, 0], subgroup_m_count = 2 : i64, subgroup_n_count = 1 : i64, workgroup = [4, 1, 1, 32, 1, 96, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [128, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @"module-6865705440217477667_apply_op_config"(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_16x48x32x288_nhwc_288x3x3x288_fhwc_nhwf_1x1s_1x1p_1x1d_1g$async_dispatch_0_conv_16x48x32x288x3x3x288_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<16x50x34x288xbf16>, %arg2: tensor<288x3x3x288xbf16>, %arg3: tensor<16x48x32x288xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d4, d2 + d5, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d3, d4, d5, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<16x50x34x288xbf16>, tensor<288x3x3x288xbf16>) outs(%arg3 : tensor<16x48x32x288xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<16x48x32x288xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, promote_operands = [0, 1], reduction = [0, 0, 0, 0, 2], subgroup = [2, 3, 1, 3, 0], subgroup_m_count = 1 : i64, subgroup_n_count = 3 : i64, workgroup = [2, 3, 16, 144, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [192, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module8204484670706657071_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
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
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, promote_operands = [0, 1], reduction = [0, 0, 0, 0, 2], subgroup = [4, 1, 1, 8, 0], subgroup_m_count = 2 : i64, subgroup_n_count = 2 : i64, workgroup = [8, 1, 16, 256, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [256, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module3663013193463791761_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_3d_bfloat16_forward_16x8x48x32x288_ndhwc_288x2x1x1x288_fdhwc_ndhwf_2x1x1s_0x0x0p_1x1x1d_1g$async_dispatch_0_conv_16x4x48x32x288x2x288_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<16x8x48x32x288xbf16>, %arg2: tensor<288x2x288xbf16>, %arg3: tensor<16x4x48x32x288xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 * 2 + d5, d2, d3, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d5, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<16x8x48x32x288xbf16>, tensor<288x2x288xbf16>) outs(%arg3 : tensor<16x4x48x32x288xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<16x4x48x32x288xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_BF16>, promote_operands = [0, 1], reduction = [0, 0, 0, 0, 0, 4], subgroup = [1, 1, 3, 1, 3, 0], subgroup_m_count = 2 : i64, subgroup_n_count = 3 : i64, workgroup = [1, 1, 6, 32, 288, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [384, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @"module-2183437910029537272_apply_op_config"(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_3d_bfloat16_forward_16x8x48x32x288_ndhwc_288x3x1x1x96_fdhwc_ndhwf_1x1x1s_1x0x0p_1x1x1d_3g$async_dispatch_0_conv_16x8x48x32x3x96x3x96_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<16x10x48x32x3x96xbf16>, %arg2: tensor<3x96x3x96xbf16>, %arg3: tensor<16x8x48x32x3x96xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1 + d6, d2, d3, d4, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d4, d5, d6, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<16x10x48x32x3x96xbf16>, tensor<3x96x3x96xbf16>) outs(%arg3 : tensor<16x8x48x32x3x96xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<16x8x48x32x3x96xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, promote_operands = [0, 1], reduction = [0, 0, 0, 0, 0, 0, 2], subgroup = [4, 1, 2, 1, 1, 3, 0], subgroup_m_count = 1 : i64, subgroup_n_count = 2 : i64, workgroup = [4, 1, 2, 16, 1, 96, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [128, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @"module-4567873843891099811_apply_op_config"(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_3d_bfloat16_forward_16x4x48x32x288_ndhwc_288x1x3x3x96_fdhwc_ndhwf_1x1x1s_0x1x1p_1x1x1d_3g$async_dispatch_0_conv_16x4x48x32x3x96x3x3x96_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<16x4x50x34x3x96xbf16>, %arg2: tensor<3x96x3x3x96xbf16>, %arg3: tensor<16x4x48x32x3x96xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d2 + d6, d3 + d7, d4, d8)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d4, d5, d6, d7, d8)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<16x4x50x34x3x96xbf16>, tensor<3x96x3x3x96xbf16>) outs(%arg3 : tensor<16x4x48x32x3x96xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<16x4x48x32x3x96xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, promote_operands = [0, 1], reduction = [0, 0, 0, 0, 0, 0, 2], subgroup = [2, 1, 2, 1, 1, 6, 0], subgroup_m_count = 2 : i64, subgroup_n_count = 1 : i64, workgroup = [4, 1, 2, 16, 1, 96, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [128, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @"module-2973162512074653305_apply_op_config"(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_16x48x32x2048_nhwc_2048x3x1x2048_fhwc_nhwf_3x1s_1x0p_1x1d_1g$async_dispatch_0_conv_16x16x32x2048x3x2048_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<16x50x32x2048xbf16>, %arg2: tensor<2048x3x2048xbf16>, %arg3: tensor<16x16x32x2048xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1 * 3 + d4, d2, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d4, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<16x50x32x2048xbf16>, tensor<2048x3x2048xbf16>) outs(%arg3 : tensor<16x16x32x2048xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<16x16x32x2048xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, promote_operands = [0, 1], reduction = [0, 0, 0, 0, 2], subgroup = [4, 1, 1, 8, 0], subgroup_m_count = 4 : i64, subgroup_n_count = 2 : i64, workgroup = [16, 1, 16, 256, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [512, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module6559857971860772791_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_3d_bfloat16_forward_16x8x48x32x288_ndhwc_288x1x3x3x96_fdhwc_ndhwf_1x1x1s_0x1x1p_1x1x1d_3g$async_dispatch_0_conv_16x8x48x32x3x96x3x3x96_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<16x8x50x34x3x96xbf16>, %arg2: tensor<3x96x3x3x96xbf16>, %arg3: tensor<16x8x48x32x3x96xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d2 + d6, d3 + d7, d4, d8)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d4, d5, d6, d7, d8)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<16x8x50x34x3x96xbf16>, tensor<3x96x3x3x96xbf16>) outs(%arg3 : tensor<16x8x48x32x3x96xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<16x8x48x32x3x96xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_BF16>, promote_operands = [0, 1], reduction = [0, 0, 0, 0, 0, 0, 4], subgroup = [1, 1, 1, 1, 1, 3, 0], subgroup_m_count = 6 : i64, subgroup_n_count = 1 : i64, workgroup = [1, 2, 3, 32, 1, 96, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [384, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module2310130817867977318_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_16x192x128x40_nhwc_40x3x3x40_fhwc_nhwf_2x2s_1x1p_1x1d_1g$async_dispatch_0_conv_16x96x64x40x3x3x40_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<16x194x130x40xbf16>, %arg2: tensor<40x3x3x40xbf16>, %arg3: tensor<16x96x64x40xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 * 2 + d4, d2 * 2 + d5, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d3, d4, d5, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<16x194x130x40xbf16>, tensor<40x3x3x40xbf16>) outs(%arg3 : tensor<16x96x64x40xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<16x96x64x40xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_BF16>, padding = [1, 3, 64, 64, 24], promote_operands = [0, 1, 2], reduction = [0, 0, 0, 0, 3], subgroup = [1, 1, 2, 2, 0], subgroup_m_count = 3 : i64, subgroup_n_count = 1 : i64, workgroup = [1, 3, 64, 64, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [192, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module2054123257679205068_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @module3663013193463791761_match_conv_2d_bfloat16_forward_16x48x32x768_nhwc_2048x3x3x768_fhwc_nhwf_1x1s_1x1p_1x1d_1g$async_dispatch_0_conv_16x48x32x2048x3x3x768_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
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
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, promote_operands = [0, 1], reduction = [0, 0, 0, 0, 2], subgroup = [2, 1, 2, 8, 0], subgroup_m_count = 2 : i64, subgroup_n_count = 2 : i64, workgroup = [4, 1, 32, 256, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [256, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module7426350141531559482_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @"module-6425628932703912647_match_conv_2d_bfloat16_forward_128x24x48x384_nhwc_384x3x1x128_fhwc_nhwf_1x1s_1x0p_1x1d_3g$async_dispatch_0_conv_128x24x48x3x128x3x128_bf16xbf16xf32"(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<128x26x48x3x128xbf16>, %arg2: tensor<3x128x3x128xbf16>, %arg3: tensor<128x24x48x3x128xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d5, d2, d3, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d3, d4, d5, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<128x26x48x3x128xbf16>, tensor<3x128x3x128xbf16>) outs(%arg3 : tensor<128x24x48x3x128xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<128x24x48x3x128xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, promote_operands = [0, 1], reduction = [0, 0, 0, 0, 0, 2], subgroup = [8, 1, 1, 1, 4, 0], subgroup_m_count = 2 : i64, subgroup_n_count = 2 : i64, workgroup = [16, 1, 16, 1, 128, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [256, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module7834922049502397065_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @module1669037280268302384_match_conv_2d_bfloat16_forward_16x48x32x576_nhwc_576x3x3x576_fhwc_nhwf_1x1s_1x1p_1x1d_1g$async_dispatch_0_conv_16x48x32x576x3x3x576_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<16x50x34x576xbf16>, %arg2: tensor<576x3x3x576xbf16>, %arg3: tensor<16x48x32x576xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d4, d2 + d5, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d3, d4, d5, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<16x50x34x576xbf16>, tensor<576x3x3x576xbf16>) outs(%arg3 : tensor<16x48x32x576xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<16x48x32x576xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, promote_operands = [0, 1], reduction = [0, 0, 0, 0, 2], subgroup = [4, 1, 2, 3, 0], subgroup_m_count = 1 : i64, subgroup_n_count = 4 : i64, workgroup = [4, 1, 32, 192, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [256, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @"module-5346593499180439123_apply_op_config"(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_16x48x32x96_nhwc_96x1x3x96_fhwc_nhwf_1x1s_0x2p_2x2d_1g$async_dispatch_0_conv_16x48x32x96x3x96_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<16x48x36x96xbf16>, %arg2: tensor<96x3x96xbf16>, %arg3: tensor<16x48x32x96xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2 + d4 * 2, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d4, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<16x48x36x96xbf16>, tensor<96x3x96xbf16>) outs(%arg3 : tensor<16x48x32x96xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<16x48x32x96xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, promote_operands = [0, 1], reduction = [0, 0, 0, 0, 3], subgroup = [1, 4, 1, 2, 0], subgroup_m_count = 1 : i64, subgroup_n_count = 1 : i64, workgroup = [1, 4, 16, 32, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [64, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module5802777366728024462_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_16x48x32x96_nhwc_96x1x3x96_fhwc_nhwf_1x1s_0x1p_1x1d_1g$async_dispatch_0_conv_16x48x32x96x3x96_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<16x48x34x96xbf16>, %arg2: tensor<96x3x96xbf16>, %arg3: tensor<16x48x32x96xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2 + d4, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d4, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<16x48x34x96xbf16>, tensor<96x3x96xbf16>) outs(%arg3 : tensor<16x48x32x96xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<16x48x32x96xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, promote_operands = [0, 1], reduction = [0, 0, 0, 0, 6], subgroup = [1, 6, 2, 1, 0], subgroup_m_count = 1 : i64, subgroup_n_count = 3 : i64, workgroup = [1, 6, 32, 48, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [192, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module6979160480657067220_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_16x48x32x96_nhwc_48x1x1x96_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_24576x48x96_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<24576x96xbf16>, %arg2: tensor<48x96xbf16>, %arg3: tensor<24576x48xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<24576x96xbf16>, tensor<48x96xbf16>) outs(%arg3 : tensor<24576x48xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<24576x48xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, promote_operands = [0, 1], reduction = [0, 0, 2], subgroup = [3, 1, 0], subgroup_m_count = 1 : i64, subgroup_n_count = 3 : i64, workgroup = [48, 48, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [192, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @"module-1545887152392129862_apply_op_config"(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_16x48x32x96_nhwc_288x1x1x96_fhwc_nhwf_2x2s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_16x24x16x288x96_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<16x24x16x96xbf16>, %arg2: tensor<288x96xbf16>, %arg3: tensor<16x24x16x288xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<16x24x16x96xbf16>, tensor<288x96xbf16>) outs(%arg3 : tensor<16x24x16x288xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<16x24x16x288xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, promote_operands = [0, 1], reduction = [0, 0, 0, 0, 6], subgroup = [2, 1, 1, 1, 0], subgroup_m_count = 3 : i64, subgroup_n_count = 2 : i64, workgroup = [2, 3, 16, 32, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [384, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module1140498713111542990_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_16x48x32x96_nhwc_288x1x1x96_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_24576x288x96_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<24576x96xbf16>, %arg2: tensor<288x96xbf16>, %arg3: tensor<24576x288xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<24576x96xbf16>, tensor<288x96xbf16>) outs(%arg3 : tensor<24576x288xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<24576x288xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, promote_operands = [0, 1], reduction = [0, 0, 2], subgroup = [2, 6, 0], subgroup_m_count = 1 : i64, subgroup_n_count = 1 : i64, workgroup = [32, 96, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [64, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @"module-6012110716509894178_apply_op_config"(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @"module-8950407434555634540_apply_op_config"(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_16x24x16x96_nhwc_96x3x1x96_fhwc_nhwf_1x1s_4x0p_4x4d_1g$async_dispatch_0_conv_16x24x16x96x3x96_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<16x32x16x96xbf16>, %arg2: tensor<96x3x96xbf16>, %arg3: tensor<16x24x16x96xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1 + d4 * 4, d2, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d4, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<16x32x16x96xbf16>, tensor<96x3x96xbf16>) outs(%arg3 : tensor<16x24x16x96xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<16x24x16x96xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, promote_operands = [0, 1], reduction = [0, 0, 0, 0, 2], subgroup = [1, 4, 1, 2, 0], subgroup_m_count = 1 : i64, subgroup_n_count = 1 : i64, workgroup = [1, 4, 16, 32, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [64, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module4308463292893212318_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_16x24x16x96_nhwc_96x3x1x96_fhwc_nhwf_1x1s_2x0p_2x2d_1g$async_dispatch_0_conv_16x24x16x96x3x96_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<16x28x16x96xbf16>, %arg2: tensor<96x3x96xbf16>, %arg3: tensor<16x24x16x96xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1 + d4 * 2, d2, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d4, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<16x28x16x96xbf16>, tensor<96x3x96xbf16>) outs(%arg3 : tensor<16x24x16x96xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<16x24x16x96xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, promote_operands = [0, 1], reduction = [0, 0, 0, 0, 2], subgroup = [1, 2, 1, 2, 0], subgroup_m_count = 1 : i64, subgroup_n_count = 1 : i64, workgroup = [1, 2, 16, 32, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [64, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module6755737478891846325_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_16x24x16x96_nhwc_96x1x3x96_fhwc_nhwf_1x1s_0x2p_2x2d_1g$async_dispatch_0_conv_16x24x16x96x3x96_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<16x24x20x96xbf16>, %arg2: tensor<96x3x96xbf16>, %arg3: tensor<16x24x16x96xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2 + d4 * 2, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d4, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<16x24x20x96xbf16>, tensor<96x3x96xbf16>) outs(%arg3 : tensor<16x24x16x96xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<16x24x16x96xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, promote_operands = [0, 1], reduction = [0, 0, 0, 0, 2], subgroup = [1, 2, 1, 1, 0], subgroup_m_count = 1 : i64, subgroup_n_count = 2 : i64, workgroup = [1, 2, 16, 32, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [128, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module7294564384001904877_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_16x24x16x96_nhwc_96x1x3x96_fhwc_nhwf_1x1s_0x1p_1x1d_1g$async_dispatch_0_conv_16x24x16x96x3x96_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<16x24x18x96xbf16>, %arg2: tensor<96x3x96xbf16>, %arg3: tensor<16x24x16x96xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2 + d4, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d4, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<16x24x18x96xbf16>, tensor<96x3x96xbf16>) outs(%arg3 : tensor<16x24x16x96xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<16x24x16x96xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, promote_operands = [0, 1], reduction = [0, 0, 0, 0, 6], subgroup = [1, 1, 1, 6, 0], subgroup_m_count = 2 : i64, subgroup_n_count = 1 : i64, workgroup = [1, 2, 16, 96, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [128, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module6549106516543380535_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @module7426350141531559482_match_conv_2d_bfloat16_forward_16x48x32x768_nhwc_2048x3x3x768_fhwc_nhwf_1x1s_1x1p_1x1d_1g$async_dispatch_0_conv_16x48x32x2048x3x3x768_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
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
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, promote_operands = [0, 1], reduction = [0, 0, 0, 0, 2], subgroup = [2, 1, 2, 8, 0], subgroup_m_count = 4 : i64, subgroup_n_count = 2 : i64, workgroup = [8, 1, 32, 256, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [512, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module4826834601099627801_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @module4378881155291204668_match_conv_2d_bfloat16_forward_16x48x32x672_nhwc_576x1x1x672_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_24576x576x672_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<24576x672xbf16>, %arg2: tensor<576x672xbf16>, %arg3: tensor<24576x576xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<24576x672xbf16>, tensor<576x672xbf16>) outs(%arg3 : tensor<24576x576xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<24576x576xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_BF16>, promote_operands = [0, 1], reduction = [0, 0, 4], subgroup = [3, 3, 0], subgroup_m_count = 2 : i64, subgroup_n_count = 3 : i64, workgroup = [192, 288, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [384, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @"module-8700669116357993800_apply_op_config"(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_16x38x38x64_nhwc_64x3x3x64_fhwc_nhwf_1x1s_1x1p_1x1d_1g$async_dispatch_0_conv_16x38x38x64x3x3x64_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<16x40x40x64xbf16>, %arg2: tensor<64x3x3x64xbf16>, %arg3: tensor<16x38x38x64xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d4, d2 + d5, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d3, d4, d5, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<16x40x40x64xbf16>, tensor<64x3x3x64xbf16>) outs(%arg3 : tensor<16x38x38x64xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<16x38x38x64xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, padding = [1, 2, 48, 32, 64], promote_operands = [0, 1, 2], reduction = [0, 0, 0, 0, 4], subgroup = [1, 2, 3, 1, 0], subgroup_m_count = 1 : i64, subgroup_n_count = 2 : i64, workgroup = [1, 2, 48, 32, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [128, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module8379457896070944627_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_16x38x38x64_nhwc_64x3x1x64_fhwc_nhwf_1x1s_1x0p_1x1d_1g$async_dispatch_0_conv_16x38x38x64x3x64_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<16x40x38x64xbf16>, %arg2: tensor<64x3x64xbf16>, %arg3: tensor<16x38x38x64xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1 + d4, d2, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d4, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<16x40x38x64xbf16>, tensor<64x3x64xbf16>) outs(%arg3 : tensor<16x38x38x64xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<16x38x38x64xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, padding = [2, 1, 48, 64, 64], promote_operands = [0, 1, 2], reduction = [0, 0, 0, 0, 4], subgroup = [1, 1, 3, 2, 0], subgroup_m_count = 2 : i64, subgroup_n_count = 2 : i64, workgroup = [2, 1, 48, 64, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [256, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @"module-5039491333864607957_apply_op_config"(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_16x38x38x64_nhwc_64x1x3x64_fhwc_nhwf_1x2s_0x1p_1x1d_1g$async_dispatch_0_conv_16x38x19x64x3x64_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<16x38x40x64xbf16>, %arg2: tensor<64x3x64xbf16>, %arg3: tensor<16x38x19x64xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2 * 2 + d4, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d4, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<16x38x40x64xbf16>, tensor<64x3x64xbf16>) outs(%arg3 : tensor<16x38x19x64xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<16x38x19x64xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, padding = [2, 1, 32, 32, 64], promote_operands = [0, 1, 2], reduction = [0, 0, 0, 0, 4], subgroup = [2, 1, 2, 1, 0], subgroup_m_count = 1 : i64, subgroup_n_count = 2 : i64, workgroup = [2, 1, 32, 32, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [128, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module1970200436349607352_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_16x1x32x64_nhwc_64x1x1x64_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_16x32x64x64_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<16x32x64xbf16>, %arg2: tensor<64x64xbf16>, %arg3: tensor<16x32x64xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<16x32x64xbf16>, tensor<64x64xbf16>) outs(%arg3 : tensor<16x32x64xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<16x32x64xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, promote_operands = [0, 1], reduction = [0, 0, 0, 4], subgroup = [1, 2, 4, 0], subgroup_m_count = 1 : i64, subgroup_n_count = 1 : i64, workgroup = [1, 32, 64, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [64, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module7070744609584329381_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @"module-5346593499180439123_match_conv_2d_bfloat16_forward_16x48x32x576_nhwc_576x3x3x576_fhwc_nhwf_1x1s_1x1p_1x1d_1g$async_dispatch_0_conv_16x48x32x576x3x3x576_bf16xbf16xf32"(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<16x50x34x576xbf16>, %arg2: tensor<576x3x3x576xbf16>, %arg3: tensor<16x48x32x576xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d4, d2 + d5, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d3, d4, d5, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<16x50x34x576xbf16>, tensor<576x3x3x576xbf16>) outs(%arg3 : tensor<16x48x32x576xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<16x48x32x576xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, promote_operands = [0, 1], reduction = [0, 0, 0, 0, 4], subgroup = [1, 2, 1, 12, 0], subgroup_m_count = 8 : i64, subgroup_n_count = 1 : i64, workgroup = [8, 2, 16, 192, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [512, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @"module-3610251752505085311_apply_op_config"(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @module8615966579730679787_match_conv_2d_bfloat16_forward_16x48x32x576_nhwc_576x1x1x576_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_24576x576x576_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<24576x576xbf16>, %arg2: tensor<576x576xbf16>, %arg3: tensor<24576x576xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<24576x576xbf16>, tensor<576x576xbf16>) outs(%arg3 : tensor<24576x576xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<24576x576xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_BF16>, promote_operands = [0, 1], reduction = [0, 0, 4], subgroup = [2, 3, 0], subgroup_m_count = 2 : i64, subgroup_n_count = 2 : i64, workgroup = [128, 192, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [256, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @"module-5863701520949056473_apply_op_config"(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_16x48x32x576_nhwc_512x1x1x576_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_24576x512x576_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<24576x576xbf16>, %arg2: tensor<512x576xbf16>, %arg3: tensor<24576x512xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<24576x576xbf16>, tensor<512x576xbf16>) outs(%arg3 : tensor<24576x512xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<24576x512xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, promote_operands = [0, 1], reduction = [0, 0, 2], subgroup = [3, 8, 0], subgroup_m_count = 8 : i64, subgroup_n_count = 1 : i64, workgroup = [384, 128, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [512, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @"module-3870568879285452189_apply_op_config"(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_16x24x16x576_nhwc_288x1x1x576_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_6144x288x576_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<6144x576xbf16>, %arg2: tensor<288x576xbf16>, %arg3: tensor<6144x288xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<6144x576xbf16>, tensor<288x576xbf16>) outs(%arg3 : tensor<6144x288xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<6144x288xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, promote_operands = [0, 1], reduction = [0, 0, 12], subgroup = [2, 2, 0], subgroup_m_count = 2 : i64, subgroup_n_count = 3 : i64, workgroup = [64, 96, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [384, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @"module-5617105841929747007_apply_op_config"(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_16x1x30x576_nhwc_8192x1x1x576_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_16x30x8192x576_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<16x30x576xbf16>, %arg2: tensor<8192x576xbf16>, %arg3: tensor<16x30x8192xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<16x30x576xbf16>, tensor<8192x576xbf16>) outs(%arg3 : tensor<16x30x8192xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<16x30x8192xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, padding = [2, 32, 64, 48], promote_operands = [0, 1, 2], reduction = [0, 0, 0, 3], subgroup = [1, 2, 4, 0], subgroup_m_count = 2 : i64, subgroup_n_count = 1 : i64, workgroup = [2, 32, 64, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [128, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module1623047442867551033_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_16x1x30x576_nhwc_3x1x1x576_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_16x30x3x576_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<16x30x576xbf16>, %arg2: tensor<3x576xbf16>, %arg3: tensor<16x30x3xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<16x30x576xbf16>, tensor<3x576xbf16>) outs(%arg3 : tensor<16x30x3xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<16x30x3xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, padding = [1, 32, 16, 192], promote_operands = [0, 1, 2], reduction = [0, 0, 0, 12], subgroup = [1, 1, 1, 0], subgroup_m_count = 2 : i64, subgroup_n_count = 1 : i64, workgroup = [1, 32, 16, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [128, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module3504644795186292955_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_16x1x30x576_nhwc_1024x1x1x576_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_16x30x1024x576_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<16x30x576xbf16>, %arg2: tensor<1024x576xbf16>, %arg3: tensor<16x30x1024xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<16x30x576xbf16>, tensor<1024x576xbf16>) outs(%arg3 : tensor<16x30x1024xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<16x30x1024xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, padding = [1, 32, 64, 192], promote_operands = [0, 1, 2], reduction = [0, 0, 0, 12], subgroup = [1, 1, 4, 0], subgroup_m_count = 2 : i64, subgroup_n_count = 1 : i64, workgroup = [1, 32, 64, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [128, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module8970479007554311969_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_16x1x2x576_nhwc_192x1x1x576_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_16x2x192x576_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<16x2x576xbf16>, %arg2: tensor<192x576xbf16>, %arg3: tensor<16x2x192xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<16x2x576xbf16>, tensor<192x576xbf16>) outs(%arg3 : tensor<16x2x192xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<16x2x192xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, padding = [1, 16, 64, 192], promote_operands = [0, 1, 2], reduction = [0, 0, 0, 12], subgroup = [1, 1, 4, 0], subgroup_m_count = 1 : i64, subgroup_n_count = 1 : i64, workgroup = [1, 16, 64, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [64, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @"module-5443390639632347215_apply_op_config"(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_16x1x2x576_nhwc_1024x1x1x576_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_16x2x1024x576_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<16x2x576xbf16>, %arg2: tensor<1024x576xbf16>, %arg3: tensor<16x2x1024xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<16x2x576xbf16>, tensor<1024x576xbf16>) outs(%arg3 : tensor<16x2x1024xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<16x2x1024xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, padding = [4, 16, 32, 48], promote_operands = [0, 1, 2], reduction = [0, 0, 0, 3], subgroup = [4, 1, 2, 0], subgroup_m_count = 1 : i64, subgroup_n_count = 1 : i64, workgroup = [4, 16, 32, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [64, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module2614628224725838856_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_16x1x1x576_nhwc_576x1x1x576_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_16x576x576_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<16x576xbf16>, %arg2: tensor<576x576xbf16>, %arg3: tensor<16x576xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<16x576xbf16>, tensor<576x576xbf16>) outs(%arg3 : tensor<16x576xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<16x576xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, promote_operands = [0, 1], reduction = [0, 0, 12], subgroup = [1, 2, 0], subgroup_m_count = 1 : i64, subgroup_n_count = 2 : i64, workgroup = [16, 64, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [128, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module891295249512997994_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @"module-4767593926805822391_match_conv_2d_bfloat16_forward_16x48x32x480_nhwc_480x1x1x96_fhwc_nhwf_1x1s_0x0p_1x1d_5g$async_dispatch_0_matmul_like_16x48x32x5x96x96_bf16xbf16xf32"(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<16x48x32x5x96xbf16>, %arg2: tensor<5x96x96xbf16>, %arg3: tensor<16x48x32x5x96xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d4, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<16x48x32x5x96xbf16>, tensor<5x96x96xbf16>) outs(%arg3 : tensor<16x48x32x5x96xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<16x48x32x5x96xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, promote_operands = [0, 1], reduction = [0, 0, 0, 0, 0, 3], subgroup = [1, 4, 1, 1, 2, 0], subgroup_m_count = 1 : i64, subgroup_n_count = 1 : i64, workgroup = [1, 4, 16, 1, 32, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [64, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @"module-8731856866266521511_apply_op_config"(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_16x48x32x480_nhwc_128x1x1x480_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_24576x128x480_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<24576x480xbf16>, %arg2: tensor<128x480xbf16>, %arg3: tensor<24576x128xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<24576x480xbf16>, tensor<128x480xbf16>) outs(%arg3 : tensor<24576x128xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<24576x128xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_BF16>, promote_operands = [0, 1], reduction = [0, 0, 4], subgroup = [1, 4, 0], subgroup_m_count = 2 : i64, subgroup_n_count = 1 : i64, workgroup = [64, 128, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [128, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module2642271885679815804_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @module5408025256171335551_match_conv_2d_bfloat16_forward_16x96x64x48_nhwc_48x5x5x48_fhwc_nhwf_1x1s_8x8p_4x4d_1g$async_dispatch_0_conv_16x96x64x48x5x5x48_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<16x112x80x48xbf16>, %arg2: tensor<48x5x5x48xbf16>, %arg3: tensor<16x96x64x48xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d4 * 4, d2 + d5 * 4, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d3, d4, d5, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<16x112x80x48xbf16>, tensor<48x5x5x48xbf16>) outs(%arg3 : tensor<16x96x64x48xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<16x96x64x48xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_BF16>, padding = [2, 3, 32, 64, 24], promote_operands = [0, 1, 2], reduction = [0, 0, 0, 0, 3], subgroup = [2, 1, 1, 2, 0], subgroup_m_count = 3 : i64, subgroup_n_count = 1 : i64, workgroup = [2, 3, 32, 64, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [192, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module4635708189346161356_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @"module-7530347551705388041_match_conv_2d_bfloat16_forward_16x96x64x48_nhwc_48x3x3x48_fhwc_nhwf_1x1s_4x4p_4x4d_1g$async_dispatch_0_conv_16x96x64x48x3x3x48_bf16xbf16xf32"(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<16x104x72x48xbf16>, %arg2: tensor<48x3x3x48xbf16>, %arg3: tensor<16x96x64x48xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d4 * 4, d2 + d5 * 4, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d3, d4, d5, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<16x104x72x48xbf16>, tensor<48x3x3x48xbf16>) outs(%arg3 : tensor<16x96x64x48xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<16x96x64x48xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_BF16>, padding = [1, 2, 32, 64, 48], promote_operands = [0, 1, 2], reduction = [0, 0, 0, 0, 6], subgroup = [1, 2, 1, 1, 0], subgroup_m_count = 1 : i64, subgroup_n_count = 2 : i64, workgroup = [1, 2, 32, 64, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [128, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module5175609052454433875_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @"module-3100570618954513192_match_conv_2d_bfloat16_forward_16x96x64x48_nhwc_48x3x3x48_fhwc_nhwf_1x1s_2x2p_2x2d_1g$async_dispatch_0_conv_16x96x64x48x3x3x48_bf16xbf16xf32"(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<16x100x68x48xbf16>, %arg2: tensor<48x3x3x48xbf16>, %arg3: tensor<16x96x64x48xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d4 * 2, d2 + d5 * 2, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d3, d4, d5, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<16x100x68x48xbf16>, tensor<48x3x3x48xbf16>) outs(%arg3 : tensor<16x96x64x48xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<16x96x64x48xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_BF16>, padding = [1, 1, 64, 64, 48], promote_operands = [0, 1, 2], reduction = [0, 0, 0, 0, 6], subgroup = [1, 1, 2, 1, 0], subgroup_m_count = 1 : i64, subgroup_n_count = 2 : i64, workgroup = [1, 1, 64, 64, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [128, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @"module-7367110414070804444_apply_op_config"(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @module2066699225626929295_match_conv_2d_bfloat16_forward_16x96x64x48_nhwc_48x3x3x48_fhwc_nhwf_1x1s_1x1p_1x1d_1g$async_dispatch_0_conv_16x96x64x48x3x3x48_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<16x98x66x48xbf16>, %arg2: tensor<48x3x3x48xbf16>, %arg3: tensor<16x96x64x48xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d4, d2 + d5, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d3, d4, d5, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<16x98x66x48xbf16>, tensor<48x3x3x48xbf16>) outs(%arg3 : tensor<16x96x64x48xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<16x96x64x48xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_BF16>, padding = [1, 1, 64, 64, 48], promote_operands = [0, 1, 2], reduction = [0, 0, 0, 0, 6], subgroup = [1, 1, 1, 2, 0], subgroup_m_count = 2 : i64, subgroup_n_count = 1 : i64, workgroup = [1, 1, 64, 64, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [128, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @"module-690138260419776630_apply_op_config"(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @"module-5011352483966054532_match_conv_2d_bfloat16_forward_16x48x32x48_nhwc_48x5x5x48_fhwc_nhwf_1x1s_8x8p_4x4d_1g$async_dispatch_0_conv_16x48x32x48x5x5x48_bf16xbf16xf32"(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<16x64x48x48xbf16>, %arg2: tensor<48x5x5x48xbf16>, %arg3: tensor<16x48x32x48xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d4 * 4, d2 + d5 * 4, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d3, d4, d5, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<16x64x48x48xbf16>, tensor<48x5x5x48xbf16>) outs(%arg3 : tensor<16x48x32x48xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<16x48x32x48xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_BF16>, padding = [1, 1, 32, 64, 48], promote_operands = [0, 1, 2], reduction = [0, 0, 0, 0, 6], subgroup = [1, 1, 1, 2, 0], subgroup_m_count = 1 : i64, subgroup_n_count = 1 : i64, workgroup = [1, 1, 32, 64, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [64, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @"module-7142624883020280426_apply_op_config"(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_16x48x32x48_nhwc_48x3x3x48_fhwc_nhwf_1x1s_2x2p_2x2d_1g$async_dispatch_0_conv_16x48x32x48x3x3x48_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<16x52x36x48xbf16>, %arg2: tensor<48x3x3x48xbf16>, %arg3: tensor<16x48x32x48xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d4 * 2, d2 + d5 * 2, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d3, d4, d5, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<16x52x36x48xbf16>, tensor<48x3x3x48xbf16>) outs(%arg3 : tensor<16x48x32x48xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<16x48x32x48xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_BF16>, padding = [1, 4, 32, 64, 16], promote_operands = [0, 1, 2], reduction = [0, 0, 0, 0, 2], subgroup = [1, 4, 1, 1, 0], subgroup_m_count = 1 : i64, subgroup_n_count = 2 : i64, workgroup = [1, 4, 32, 64, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [128, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @"module-5878059958949092251_apply_op_config"(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_16x24x16x48_nhwc_192x1x1x48_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_6144x192x48_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<6144x48xbf16>, %arg2: tensor<192x48xbf16>, %arg3: tensor<6144x192xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<6144x48xbf16>, tensor<192x48xbf16>) outs(%arg3 : tensor<6144x192xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<6144x192xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_BF16>, promote_operands = [0, 1], reduction = [0, 0, 2], subgroup = [2, 3, 0], subgroup_m_count = 1 : i64, subgroup_n_count = 1 : i64, workgroup = [64, 96, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [64, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @"module-3236419692971477372_apply_op_config"(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_16x96x64x40_nhwc_40x1x1x40_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_98304x40x40_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<98304x40xbf16>, %arg2: tensor<40x40xbf16>, %arg3: tensor<98304x40xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<98304x40xbf16>, tensor<40x40xbf16>) outs(%arg3 : tensor<98304x40xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<98304x40xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_BF16>, padding = [64, 64, 8], promote_operands = [0, 1, 2], reduction = [0, 0, 1], subgroup = [2, 2, 0], subgroup_m_count = 1 : i64, subgroup_n_count = 1 : i64, workgroup = [64, 64, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [64, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @"module-9078255276051304928_apply_op_config"(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @module2054123257679205068_match_conv_2d_bfloat16_forward_16x192x128x40_nhwc_40x3x3x40_fhwc_nhwf_2x2s_1x1p_1x1d_1g$async_dispatch_0_conv_16x96x64x40x3x3x40_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<16x194x130x40xbf16>, %arg2: tensor<40x3x3x40xbf16>, %arg3: tensor<16x96x64x40xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 * 2 + d4, d2 * 2 + d5, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d3, d4, d5, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<16x194x130x40xbf16>, tensor<40x3x3x40xbf16>) outs(%arg3 : tensor<16x96x64x40xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<16x96x64x40xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_BF16>, padding = [2, 3, 32, 64, 24], promote_operands = [0, 1, 2], reduction = [0, 0, 0, 0, 3], subgroup = [2, 1, 1, 2, 0], subgroup_m_count = 3 : i64, subgroup_n_count = 1 : i64, workgroup = [2, 3, 32, 64, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [192, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @"module-4706140772942856255_apply_op_config"(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_16x48x32x384_nhwc_384x1x1x384_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_24576x384x384_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<24576x384xbf16>, %arg2: tensor<384x384xbf16>, %arg3: tensor<24576x384xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<24576x384xbf16>, tensor<384x384xbf16>) outs(%arg3 : tensor<24576x384xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<24576x384xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, promote_operands = [0, 1], reduction = [0, 0, 4], subgroup = [1, 8, 0], subgroup_m_count = 4 : i64, subgroup_n_count = 1 : i64, workgroup = [64, 128, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [256, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module3494839312045714732_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_16x12x8x384_nhwc_192x1x1x384_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_1536x192x384_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<1536x384xbf16>, %arg2: tensor<192x384xbf16>, %arg3: tensor<1536x192xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<1536x384xbf16>, tensor<192x384xbf16>) outs(%arg3 : tensor<1536x192xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<1536x192xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, promote_operands = [0, 1], reduction = [0, 0, 12], subgroup = [3, 1, 0], subgroup_m_count = 1 : i64, subgroup_n_count = 3 : i64, workgroup = [48, 48, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [192, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module813434365613521345_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_16x8x32x288_nhwc_384x1x1x288_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_4096x384x288_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<4096x288xbf16>, %arg2: tensor<384x288xbf16>, %arg3: tensor<4096x384xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<4096x288xbf16>, tensor<384x288xbf16>) outs(%arg3 : tensor<4096x384xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<4096x384xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, promote_operands = [0, 1], reduction = [0, 0, 6], subgroup = [2, 2, 0], subgroup_m_count = 2 : i64, subgroup_n_count = 3 : i64, workgroup = [64, 96, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [384, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module2933426064943781440_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_16x48x32x288_nhwc_96x1x1x288_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_24576x96x288_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<24576x288xbf16>, %arg2: tensor<96x288xbf16>, %arg3: tensor<24576x96xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<24576x288xbf16>, tensor<96x288xbf16>) outs(%arg3 : tensor<24576x96xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<24576x96xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, promote_operands = [0, 1], reduction = [0, 0, 6], subgroup = [1, 6, 0], subgroup_m_count = 2 : i64, subgroup_n_count = 1 : i64, workgroup = [32, 96, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [128, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module6152709004796445680_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_16x48x32x288_nhwc_384x1x1x288_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_24576x384x288_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<24576x288xbf16>, %arg2: tensor<384x288xbf16>, %arg3: tensor<24576x384xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<24576x288xbf16>, tensor<384x288xbf16>) outs(%arg3 : tensor<24576x384xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<24576x384xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, promote_operands = [0, 1], reduction = [0, 0, 2], subgroup = [8, 3, 0], subgroup_m_count = 1 : i64, subgroup_n_count = 8 : i64, workgroup = [128, 384, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [512, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module8250078879978817073_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_16x48x32x288_nhwc_288x3x3x96_fhwc_nhwf_2x2s_1x1p_1x1d_3g$async_dispatch_0_conv_16x24x16x3x96x3x3x96_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<16x50x34x3x96xbf16>, %arg2: tensor<3x96x3x3x96xbf16>, %arg3: tensor<16x24x16x3x96xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1 * 2 + d5, d2 * 2 + d6, d3, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d3, d4, d5, d6, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<16x50x34x3x96xbf16>, tensor<3x96x3x3x96xbf16>) outs(%arg3 : tensor<16x24x16x3x96xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<16x24x16x3x96xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, promote_operands = [0, 1], reduction = [0, 0, 0, 0, 0, 2], subgroup = [2, 1, 1, 1, 2, 0], subgroup_m_count = 2 : i64, subgroup_n_count = 1 : i64, workgroup = [4, 1, 16, 1, 32, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [128, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module4044126642927762173_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @module8204484670706657071_match_conv_2d_bfloat16_forward_16x48x32x288_nhwc_288x3x3x288_fhwc_nhwf_1x1s_1x1p_1x1d_1g$async_dispatch_0_conv_16x48x32x288x3x3x288_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<16x50x34x288xbf16>, %arg2: tensor<288x3x3x288xbf16>, %arg3: tensor<16x48x32x288xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d4, d2 + d5, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d3, d4, d5, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<16x50x34x288xbf16>, tensor<288x3x3x288xbf16>) outs(%arg3 : tensor<16x48x32x288xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<16x48x32x288xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, promote_operands = [0, 1], reduction = [0, 0, 0, 0, 2], subgroup = [1, 3, 1, 6, 0], subgroup_m_count = 2 : i64, subgroup_n_count = 3 : i64, workgroup = [1, 6, 16, 288, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [384, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @"module-1439035349042860153_apply_op_config"(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_16x24x16x288_nhwc_96x1x1x288_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_6144x96x288_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<6144x288xbf16>, %arg2: tensor<96x288xbf16>, %arg3: tensor<6144x96xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<6144x288xbf16>, tensor<96x288xbf16>) outs(%arg3 : tensor<6144x96xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<6144x96xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, promote_operands = [0, 1], reduction = [0, 0, 6], subgroup = [1, 2, 0], subgroup_m_count = 1 : i64, subgroup_n_count = 3 : i64, workgroup = [16, 96, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [192, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @"module-4383060726791129170_apply_op_config"(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_16x24x16x288_nhwc_288x3x3x288_fhwc_nhwf_1x1s_1x1p_1x1d_1g$async_dispatch_0_conv_16x24x16x288x3x3x288_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<16x26x18x288xbf16>, %arg2: tensor<288x3x3x288xbf16>, %arg3: tensor<16x24x16x288xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d4, d2 + d5, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d3, d4, d5, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<16x26x18x288xbf16>, tensor<288x3x3x288xbf16>) outs(%arg3 : tensor<16x24x16x288xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<16x24x16x288xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, promote_operands = [0, 1], reduction = [0, 0, 0, 0, 6], subgroup = [2, 2, 1, 1, 0], subgroup_m_count = 3 : i64, subgroup_n_count = 2 : i64, workgroup = [2, 6, 16, 32, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [384, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @"module-6247610171180181805_apply_op_config"(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @"module-8838592581327294770_match_conv_2d_bfloat16_forward_16x24x16x288_nhwc_288x3x1x96_fhwc_nhwf_1x1s_1x0p_1x1d_3g$async_dispatch_0_conv_16x24x16x3x96x3x96_bf16xbf16xf32"(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<16x26x16x3x96xbf16>, %arg2: tensor<3x96x3x96xbf16>, %arg3: tensor<16x24x16x3x96xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d5, d2, d3, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d3, d4, d5, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<16x26x16x3x96xbf16>, tensor<3x96x3x96xbf16>) outs(%arg3 : tensor<16x24x16x3x96xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<16x24x16x3x96xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, promote_operands = [0, 1], reduction = [0, 0, 0, 0, 0, 6], subgroup = [2, 1, 1, 1, 2, 0], subgroup_m_count = 3 : i64, subgroup_n_count = 3 : i64, workgroup = [2, 3, 16, 1, 96, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [576, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module6677447418464053593_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_16x24x16x288_nhwc_288x1x3x96_fhwc_nhwf_1x1s_0x1p_1x1d_3g$async_dispatch_0_conv_16x24x16x3x96x3x96_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<16x24x18x3x96xbf16>, %arg2: tensor<3x96x3x96xbf16>, %arg3: tensor<16x24x16x3x96xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2 + d5, d3, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d3, d4, d5, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<16x24x18x3x96xbf16>, tensor<3x96x3x96xbf16>) outs(%arg3 : tensor<16x24x16x3x96xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<16x24x16x3x96xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, promote_operands = [0, 1], reduction = [0, 0, 0, 0, 0, 6], subgroup = [8, 1, 1, 1, 1, 0], subgroup_m_count = 1 : i64, subgroup_n_count = 6 : i64, workgroup = [8, 1, 16, 1, 96, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [384, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @"module-5087663706872769327_apply_op_config"(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_16x24x16x288_nhwc_288x1x1x96_fhwc_nhwf_1x1s_0x0p_1x1d_3g$async_dispatch_0_matmul_like_16x24x16x3x96x96_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<16x24x16x3x96xbf16>, %arg2: tensor<3x96x96xbf16>, %arg3: tensor<16x24x16x3x96xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d4, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<16x24x16x3x96xbf16>, tensor<3x96x96xbf16>) outs(%arg3 : tensor<16x24x16x3x96xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<16x24x16x3x96xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, promote_operands = [0, 1], reduction = [0, 0, 0, 0, 0, 6], subgroup = [1, 3, 1, 1, 6, 0], subgroup_m_count = 2 : i64, subgroup_n_count = 1 : i64, workgroup = [2, 3, 16, 1, 96, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [128, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @"module-6980181089137153937_apply_op_config"(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_16x24x16x288_nhwc_288x1x1x288_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_6144x288x288_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<6144x288xbf16>, %arg2: tensor<288x288xbf16>, %arg3: tensor<6144x288xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<6144x288xbf16>, tensor<288x288xbf16>) outs(%arg3 : tensor<6144x288xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<6144x288xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_BF16>, promote_operands = [0, 1], reduction = [0, 0, 4], subgroup = [1, 3, 0], subgroup_m_count = 3 : i64, subgroup_n_count = 1 : i64, workgroup = [96, 96, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [192, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module5596125379591511372_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_16x24x16x288_nhwc_144x1x1x288_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_6144x144x288_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<6144x288xbf16>, %arg2: tensor<144x288xbf16>, %arg3: tensor<6144x144xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<6144x288xbf16>, tensor<144x288xbf16>) outs(%arg3 : tensor<6144x144xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<6144x144xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, promote_operands = [0, 1], reduction = [0, 0, 2], subgroup = [3, 1, 0], subgroup_m_count = 1 : i64, subgroup_n_count = 1 : i64, workgroup = [48, 16, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [64, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module4283728422656460134_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @"module-4567873843891099811_match_conv_3d_bfloat16_forward_16x8x48x32x288_ndhwc_288x3x1x1x96_fdhwc_ndhwf_1x1x1s_1x0x0p_1x1x1d_3g$async_dispatch_0_conv_16x8x48x32x3x96x3x96_bf16xbf16xf32"(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<16x10x48x32x3x96xbf16>, %arg2: tensor<3x96x3x96xbf16>, %arg3: tensor<16x8x48x32x3x96xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1 + d6, d2, d3, d4, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d4, d5, d6, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<16x10x48x32x3x96xbf16>, tensor<3x96x3x96xbf16>) outs(%arg3 : tensor<16x8x48x32x3x96xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<16x8x48x32x3x96xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_BF16>, promote_operands = [0, 1], reduction = [0, 0, 0, 0, 0, 0, 6], subgroup = [1, 2, 3, 1, 1, 1, 0], subgroup_m_count = 1 : i64, subgroup_n_count = 3 : i64, workgroup = [1, 2, 3, 32, 1, 96, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [192, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module9097057252924939381_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @"module-2183437910029537272_match_conv_3d_bfloat16_forward_16x8x48x32x288_ndhwc_288x2x1x1x288_fdhwc_ndhwf_2x1x1s_0x0x0p_1x1x1d_1g$async_dispatch_0_conv_16x4x48x32x288x2x288_bf16xbf16xf32"(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<16x8x48x32x288xbf16>, %arg2: tensor<288x2x288xbf16>, %arg3: tensor<16x4x48x32x288xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 * 2 + d5, d2, d3, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d5, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<16x8x48x32x288xbf16>, tensor<288x2x288xbf16>) outs(%arg3 : tensor<16x4x48x32x288xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<16x4x48x32x288xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_BF16>, promote_operands = [0, 1], reduction = [0, 0, 0, 0, 0, 8], subgroup = [1, 1, 2, 1, 3, 0], subgroup_m_count = 2 : i64, subgroup_n_count = 1 : i64, workgroup = [1, 2, 2, 32, 96, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [128, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module205055630836799240_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @module2310130817867977318_match_conv_3d_bfloat16_forward_16x8x48x32x288_ndhwc_288x1x3x3x96_fdhwc_ndhwf_1x1x1s_0x1x1p_1x1x1d_3g$async_dispatch_0_conv_16x8x48x32x3x96x3x3x96_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<16x8x50x34x3x96xbf16>, %arg2: tensor<3x96x3x3x96xbf16>, %arg3: tensor<16x8x48x32x3x96xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d2 + d6, d3 + d7, d4, d8)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d4, d5, d6, d7, d8)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<16x8x50x34x3x96xbf16>, tensor<3x96x3x3x96xbf16>) outs(%arg3 : tensor<16x8x48x32x3x96xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<16x8x48x32x3x96xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, promote_operands = [0, 1], reduction = [0, 0, 0, 0, 0, 0, 2], subgroup = [1, 2, 2, 2, 1, 3, 0], subgroup_m_count = 1 : i64, subgroup_n_count = 2 : i64, workgroup = [1, 2, 2, 32, 1, 96, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [128, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @"module-5319777781164961521_apply_op_config"(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @"module-5385892577352669266_match_conv_3d_bfloat16_forward_16x4x48x32x288_ndhwc_288x3x1x1x96_fdhwc_ndhwf_1x1x1s_1x0x0p_1x1x1d_3g$async_dispatch_0_conv_16x4x48x32x3x96x3x96_bf16xbf16xf32"(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<16x6x48x32x3x96xbf16>, %arg2: tensor<3x96x3x96xbf16>, %arg3: tensor<16x4x48x32x3x96xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1 + d6, d2, d3, d4, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d4, d5, d6, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<16x6x48x32x3x96xbf16>, tensor<3x96x3x96xbf16>) outs(%arg3 : tensor<16x4x48x32x3x96xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<16x4x48x32x3x96xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, promote_operands = [0, 1], reduction = [0, 0, 0, 0, 0, 0, 2], subgroup = [2, 2, 3, 1, 1, 2, 0], subgroup_m_count = 1 : i64, subgroup_n_count = 3 : i64, workgroup = [2, 2, 3, 16, 1, 96, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [192, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module1835695904987310839_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @"module-2215561608523455322_match_conv_3d_bfloat16_forward_16x4x48x32x288_ndhwc_288x2x1x1x288_fdhwc_ndhwf_2x1x1s_0x0x0p_1x1x1d_1g$async_dispatch_0_conv_16x2x48x32x288x2x288_bf16xbf16xf32"(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<16x4x48x32x288xbf16>, %arg2: tensor<288x2x288xbf16>, %arg3: tensor<16x2x48x32x288xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 * 2 + d5, d2, d3, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d5, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<16x4x48x32x288xbf16>, tensor<288x2x288xbf16>) outs(%arg3 : tensor<16x2x48x32x288xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<16x2x48x32x288xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, promote_operands = [0, 1], reduction = [0, 0, 0, 0, 0, 4], subgroup = [1, 1, 2, 1, 3, 0], subgroup_m_count = 2 : i64, subgroup_n_count = 2 : i64, workgroup = [1, 2, 2, 16, 96, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [256, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @"module-2958846526441287144_apply_op_config"(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @"module-2973162512074653305_match_conv_3d_bfloat16_forward_16x4x48x32x288_ndhwc_288x1x3x3x96_fdhwc_ndhwf_1x1x1s_0x1x1p_1x1x1d_3g$async_dispatch_0_conv_16x4x48x32x3x96x3x3x96_bf16xbf16xf32"(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<16x4x50x34x3x96xbf16>, %arg2: tensor<3x96x3x3x96xbf16>, %arg3: tensor<16x4x48x32x3x96xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d2 + d6, d3 + d7, d4, d8)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d4, d5, d6, d7, d8)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<16x4x50x34x3x96xbf16>, tensor<3x96x3x3x96xbf16>) outs(%arg3 : tensor<16x4x48x32x3x96xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<16x4x48x32x3x96xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, promote_operands = [0, 1], reduction = [0, 0, 0, 0, 0, 0, 2], subgroup = [1, 1, 8, 1, 1, 6, 0], subgroup_m_count = 2 : i64, subgroup_n_count = 1 : i64, workgroup = [1, 1, 16, 16, 1, 96, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [128, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @"module-252229267495831757_apply_op_config"(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @"module-1589592913381196107_match_conv_3d_bfloat16_forward_16x2x48x32x288_ndhwc_288x3x1x1x96_fdhwc_ndhwf_1x1x1s_1x0x0p_1x1x1d_3g$async_dispatch_0_conv_16x2x48x32x3x96x3x96_bf16xbf16xf32"(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<16x4x48x32x3x96xbf16>, %arg2: tensor<3x96x3x96xbf16>, %arg3: tensor<16x2x48x32x3x96xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1 + d6, d2, d3, d4, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d4, d5, d6, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<16x4x48x32x3x96xbf16>, tensor<3x96x3x96xbf16>) outs(%arg3 : tensor<16x2x48x32x3x96xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<16x2x48x32x3x96xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_BF16>, promote_operands = [0, 1], reduction = [0, 0, 0, 0, 0, 0, 12], subgroup = [1, 1, 1, 1, 1, 3, 0], subgroup_m_count = 6 : i64, subgroup_n_count = 1 : i64, workgroup = [2, 1, 3, 32, 1, 96, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [384, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module8217597828895093057_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @module3244247317313473054_match_conv_3d_bfloat16_forward_16x2x48x32x288_ndhwc_288x2x1x1x288_fdhwc_ndhwf_2x1x1s_0x0x0p_1x1x1d_1g$async_dispatch_0_matmul_like_16x48x32x288x2x288_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<16x2x48x32x288xbf16>, %arg2: tensor<288x2x288xbf16>, %arg3: tensor<16x48x32x288xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d4, d1, d2, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d4, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<16x2x48x32x288xbf16>, tensor<288x2x288xbf16>) outs(%arg3 : tensor<16x48x32x288xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<16x48x32x288xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, promote_operands = [0, 1], reduction = [0, 0, 0, 0, 1, 2], subgroup = [2, 1, 1, 3, 0, 0], subgroup_m_count = 3 : i64, subgroup_n_count = 2 : i64, workgroup = [2, 3, 16, 96, 0, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [384, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module1362604622253719980_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @"module-6865705440217477667_match_conv_3d_bfloat16_forward_16x2x48x32x288_ndhwc_288x1x3x3x96_fdhwc_ndhwf_1x1x1s_0x1x1p_1x1x1d_3g$async_dispatch_0_conv_16x2x48x32x3x96x3x3x96_bf16xbf16xf32"(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<16x2x50x34x3x96xbf16>, %arg2: tensor<3x96x3x3x96xbf16>, %arg3: tensor<16x2x48x32x3x96xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d2 + d6, d3 + d7, d4, d8)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d4, d5, d6, d7, d8)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<16x2x50x34x3x96xbf16>, tensor<3x96x3x3x96xbf16>) outs(%arg3 : tensor<16x2x48x32x3x96xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<16x2x48x32x3x96xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, promote_operands = [0, 1], reduction = [0, 0, 0, 0, 0, 0, 2], subgroup = [2, 1, 4, 1, 1, 3, 0], subgroup_m_count = 1 : i64, subgroup_n_count = 2 : i64, workgroup = [2, 1, 4, 16, 1, 96, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [128, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @"module-7088503340171957963_apply_op_config"(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_16x8x32x2048_nhwc_576x1x1x2048_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_4096x576x2048_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<4096x2048xbf16>, %arg2: tensor<576x2048xbf16>, %arg3: tensor<4096x576xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<4096x2048xbf16>, tensor<576x2048xbf16>) outs(%arg3 : tensor<4096x576xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<4096x576xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, promote_operands = [0, 1], reduction = [0, 0, 4], subgroup = [2, 4, 0], subgroup_m_count = 4 : i64, subgroup_n_count = 1 : i64, workgroup = [128, 64, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [256, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module2280249657558078362_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @module3994070714556536465_match_conv_2d_bfloat16_forward_16x8x32x2048_nhwc_288x1x1x2048_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_4096x288x2048_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<4096x2048xbf16>, %arg2: tensor<288x2048xbf16>, %arg3: tensor<4096x288xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<4096x2048xbf16>, tensor<288x2048xbf16>) outs(%arg3 : tensor<4096x288xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<4096x288xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, promote_operands = [0, 1], reduction = [0, 0, 8], subgroup = [1, 3, 0], subgroup_m_count = 4 : i64, subgroup_n_count = 2 : i64, workgroup = [64, 96, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [512, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @"module-4712683707803685992_apply_op_config"(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @module6559857971860772791_match_conv_2d_bfloat16_forward_16x48x32x2048_nhwc_2048x3x1x2048_fhwc_nhwf_3x1s_1x0p_1x1d_1g$async_dispatch_0_conv_16x16x32x2048x3x2048_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<16x50x32x2048xbf16>, %arg2: tensor<2048x3x2048xbf16>, %arg3: tensor<16x16x32x2048xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1 * 3 + d4, d2, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d4, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<16x50x32x2048xbf16>, tensor<2048x3x2048xbf16>) outs(%arg3 : tensor<16x16x32x2048xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<16x16x32x2048xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, promote_operands = [0, 1], reduction = [0, 0, 0, 0, 2], subgroup = [4, 1, 2, 4, 0], subgroup_m_count = 2 : i64, subgroup_n_count = 4 : i64, workgroup = [8, 1, 32, 256, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [512, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module4870784003488566835_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_16x1x512x2048_nhwc_2048x1x2x2048_fhwc_nhwf_1x2s_0x0p_1x1d_1g$async_dispatch_0_conv_16x256x2048x2x2048_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<16x512x2048xbf16>, %arg2: tensor<2048x2x2048xbf16>, %arg3: tensor<16x256x2048xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1 * 2 + d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d2, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<16x512x2048xbf16>, tensor<2048x2x2048xbf16>) outs(%arg3 : tensor<16x256x2048xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<16x256x2048xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_BF16>, promote_operands = [0, 1], reduction = [0, 0, 0, 8], subgroup = [4, 1, 1, 0], subgroup_m_count = 2 : i64, subgroup_n_count = 4 : i64, workgroup = [8, 32, 128, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [512, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module7219792799496758047_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @module7350505846474122665_match_conv_2d_bfloat16_forward_16x96x64x192_nhwc_96x1x1x192_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_98304x96x192_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<98304x192xbf16>, %arg2: tensor<96x192xbf16>, %arg3: tensor<98304x96xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<98304x192xbf16>, tensor<96x192xbf16>) outs(%arg3 : tensor<98304x96xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<98304x96xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, promote_operands = [0, 1], reduction = [0, 0, 4], subgroup = [8, 3, 0], subgroup_m_count = 3 : i64, subgroup_n_count = 2 : i64, workgroup = [384, 96, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [384, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module8623344520257799845_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_16x48x32x192_nhwc_96x1x1x192_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_24576x96x192_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<24576x192xbf16>, %arg2: tensor<96x192xbf16>, %arg3: tensor<24576x96xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<24576x192xbf16>, tensor<96x192xbf16>) outs(%arg3 : tensor<24576x96xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<24576x96xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_BF16>, promote_operands = [0, 1], reduction = [0, 0, 2], subgroup = [3, 1, 0], subgroup_m_count = 1 : i64, subgroup_n_count = 3 : i64, workgroup = [96, 96, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [192, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module2923642059297261885_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_16x23x1x192_nhwc_192x3x1x192_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_conv_16x21x192x3x192_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<16x23x192xbf16>, %arg2: tensor<192x3x192xbf16>, %arg3: tensor<16x21x192xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1 + d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d2, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<16x23x192xbf16>, tensor<192x3x192xbf16>) outs(%arg3 : tensor<16x21x192xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<16x21x192xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, padding = [1, 16, 16, 64], promote_operands = [0, 1, 2], reduction = [0, 0, 0, 4], subgroup = [1, 1, 1, 0], subgroup_m_count = 1 : i64, subgroup_n_count = 1 : i64, workgroup = [1, 16, 16, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [64, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module354875760083644398_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_16x12x8x192_nhwc_384x1x1x192_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_1536x384x192_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<1536x192xbf16>, %arg2: tensor<384x192xbf16>, %arg3: tensor<1536x384xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<1536x192xbf16>, tensor<384x192xbf16>) outs(%arg3 : tensor<1536x384xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<1536x384xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, promote_operands = [0, 1], reduction = [0, 0, 4], subgroup = [2, 1, 0], subgroup_m_count = 1 : i64, subgroup_n_count = 2 : i64, workgroup = [32, 32, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [128, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module4419538543564309907_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_16x1x21x192_nhwc_384x1x1x192_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_16x21x384x192_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<16x21x192xbf16>, %arg2: tensor<384x192xbf16>, %arg3: tensor<16x21x384xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<16x21x192xbf16>, tensor<384x192xbf16>) outs(%arg3 : tensor<16x21x384xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<16x21x384xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, padding = [2, 16, 32, 192], promote_operands = [0, 1, 2], reduction = [0, 0, 0, 12], subgroup = [1, 1, 2, 0], subgroup_m_count = 2 : i64, subgroup_n_count = 1 : i64, workgroup = [2, 16, 32, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [128, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @"module-271961346453902565_apply_op_config"(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_16x48x32x128_nhwc_64x1x1x128_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_24576x64x128_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<24576x128xbf16>, %arg2: tensor<64x128xbf16>, %arg3: tensor<24576x64xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<24576x128xbf16>, tensor<64x128xbf16>) outs(%arg3 : tensor<24576x64xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<24576x64xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, promote_operands = [0, 1], reduction = [0, 0, 2], subgroup = [2, 1, 0], subgroup_m_count = 1 : i64, subgroup_n_count = 2 : i64, workgroup = [32, 32, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [128, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module8040801911246838659_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_16x48x32x128_nhwc_128x2x2x128_fhwc_nhwf_2x2s_0x0p_1x1d_1g$async_dispatch_0_conv_16x24x16x128x2x2x128_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<16x48x32x128xbf16>, %arg2: tensor<128x2x2x128xbf16>, %arg3: tensor<16x24x16x128xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 * 2 + d4, d2 * 2 + d5, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d3, d4, d5, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<16x48x32x128xbf16>, tensor<128x2x2x128xbf16>) outs(%arg3 : tensor<16x24x16x128xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<16x24x16x128xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, promote_operands = [0, 1], reduction = [0, 0, 0, 0, 2], subgroup = [1, 1, 1, 4, 0], subgroup_m_count = 2 : i64, subgroup_n_count = 1 : i64, workgroup = [1, 2, 16, 64, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [128, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @"module-7653633243858028790_apply_op_config"(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_16x24x16x128_nhwc_192x2x2x128_fhwc_nhwf_2x2s_0x0p_1x1d_1g$async_dispatch_0_conv_16x12x8x192x2x2x128_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<16x24x16x128xbf16>, %arg2: tensor<192x2x2x128xbf16>, %arg3: tensor<16x12x8x192xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 * 2 + d4, d2 * 2 + d5, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d3, d4, d5, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<16x24x16x128xbf16>, tensor<192x2x2x128xbf16>) outs(%arg3 : tensor<16x12x8x192xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<16x12x8x192xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, padding = [1, 1, 16, 32, 128], promote_operands = [0, 1], reduction = [0, 0, 0, 0, 8], subgroup = [1, 1, 1, 1, 0], subgroup_m_count = 1 : i64, subgroup_n_count = 2 : i64, workgroup = [1, 1, 16, 32, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [128, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @"module-7602620474587856351_apply_op_config"(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_16x1x6x1024_nhwc_576x1x1x1024_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_16x6x576x1024_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<16x6x1024xbf16>, %arg2: tensor<576x1024xbf16>, %arg3: tensor<16x6x576xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<16x6x1024xbf16>, tensor<576x1024xbf16>) outs(%arg3 : tensor<16x6x576xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<16x6x576xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, padding = [2, 16, 64, 256], promote_operands = [0, 1], reduction = [0, 0, 0, 16], subgroup = [1, 1, 1, 0], subgroup_m_count = 2 : i64, subgroup_n_count = 4 : i64, workgroup = [2, 16, 64, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [512, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @"module-4289354440547745606_apply_op_config"(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @"module-7508047808273090963_match_conv_2d_bfloat16_forward_128x24x48x480_nhwc_384x1x1x480_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_147456x384x480_bf16xbf16xf32"(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<147456x480xbf16>, %arg2: tensor<384x480xbf16>, %arg3: tensor<147456x384xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<147456x480xbf16>, tensor<384x480xbf16>) outs(%arg3 : tensor<147456x384xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<147456x384xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, promote_operands = [0, 1], reduction = [0, 0, 2], subgroup = [2, 12, 0], subgroup_m_count = 4 : i64, subgroup_n_count = 2 : i64, workgroup = [128, 384, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [512, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module6151868810080441787_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_128x48x32x448_nhwc_384x1x1x448_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_196608x384x448_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<196608x448xbf16>, %arg2: tensor<384x448xbf16>, %arg3: tensor<196608x384xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<196608x448xbf16>, tensor<384x448xbf16>) outs(%arg3 : tensor<196608x384xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<196608x384xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_BF16>, promote_operands = [0, 1], reduction = [0, 0, 4], subgroup = [1, 2, 0], subgroup_m_count = 4 : i64, subgroup_n_count = 2 : i64, workgroup = [128, 128, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [512, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module5741017693719714659_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @"module-1507712216101663294_match_conv_2d_bfloat16_forward_128x48x32x384_nhwc_96x1x1x384_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_196608x96x384_bf16xbf16xf32"(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<196608x384xbf16>, %arg2: tensor<96x384xbf16>, %arg3: tensor<196608x96xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<196608x384xbf16>, tensor<96x384xbf16>) outs(%arg3 : tensor<196608x96xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<196608x96xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, promote_operands = [0, 1], reduction = [0, 0, 2], subgroup = [4, 3, 0], subgroup_m_count = 3 : i64, subgroup_n_count = 2 : i64, workgroup = [192, 96, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [384, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module3933653812497562654_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_128x48x32x384_nhwc_384x3x3x64_fhwc_nhwf_1x1s_1x1p_1x1d_6g$async_dispatch_0_conv_128x48x32x6x64x3x3x64_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<128x50x34x6x64xbf16>, %arg2: tensor<6x64x3x3x64xbf16>, %arg3: tensor<128x48x32x6x64xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1 + d5, d2 + d6, d3, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d3, d4, d5, d6, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<128x50x34x6x64xbf16>, tensor<6x64x3x3x64xbf16>) outs(%arg3 : tensor<128x48x32x6x64xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<128x48x32x6x64xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, promote_operands = [0, 1], reduction = [0, 0, 0, 0, 0, 4], subgroup = [2, 1, 1, 1, 4, 0], subgroup_m_count = 4 : i64, subgroup_n_count = 1 : i64, workgroup = [4, 2, 16, 1, 64, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [256, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @"module-3096228896624724732_apply_op_config"(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_128x48x32x384_nhwc_384x1x1x384_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_196608x384x384_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<196608x384xbf16>, %arg2: tensor<384x384xbf16>, %arg3: tensor<196608x384xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<196608x384xbf16>, tensor<384x384xbf16>) outs(%arg3 : tensor<196608x384xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<196608x384xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_BF16>, promote_operands = [0, 1], reduction = [0, 0, 4], subgroup = [2, 1, 0], subgroup_m_count = 2 : i64, subgroup_n_count = 4 : i64, workgroup = [128, 128, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [512, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module5265578897403436880_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @"module-4092354048484454229_match_conv_2d_bfloat16_forward_128x24x48x384_nhwc_512x1x1x384_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_147456x512x384_bf16xbf16xf32"(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<147456x384xbf16>, %arg2: tensor<512x384xbf16>, %arg3: tensor<147456x512xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<147456x384xbf16>, tensor<512x384xbf16>) outs(%arg3 : tensor<147456x512xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<147456x512xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_BF16>, promote_operands = [0, 1], reduction = [0, 0, 4], subgroup = [1, 4, 0], subgroup_m_count = 4 : i64, subgroup_n_count = 2 : i64, workgroup = [128, 256, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [512, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @"module-5713853901338332798_apply_op_config"(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @module7834922049502397065_match_conv_2d_bfloat16_forward_128x24x48x384_nhwc_384x3x1x128_fhwc_nhwf_1x1s_1x0p_1x1d_3g$async_dispatch_0_conv_128x24x48x3x128x3x128_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<128x26x48x3x128xbf16>, %arg2: tensor<3x128x3x128xbf16>, %arg3: tensor<128x24x48x3x128xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d5, d2, d3, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d3, d4, d5, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<128x26x48x3x128xbf16>, tensor<3x128x3x128xbf16>) outs(%arg3 : tensor<128x24x48x3x128xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<128x24x48x3x128xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, promote_operands = [0, 1], reduction = [0, 0, 0, 0, 0, 2], subgroup = [1, 3, 1, 1, 8, 0], subgroup_m_count = 8 : i64, subgroup_n_count = 1 : i64, workgroup = [8, 3, 16, 1, 128, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [512, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @"module-3307929099139602236_apply_op_config"(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_128x24x48x384_nhwc_384x1x3x128_fhwc_nhwf_1x1s_0x1p_1x1d_3g$async_dispatch_0_conv_128x24x48x3x128x3x128_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<128x24x50x3x128xbf16>, %arg2: tensor<3x128x3x128xbf16>, %arg3: tensor<128x24x48x3x128xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2 + d5, d3, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d3, d4, d5, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<128x24x50x3x128xbf16>, tensor<3x128x3x128xbf16>) outs(%arg3 : tensor<128x24x48x3x128xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<128x24x48x3x128xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, promote_operands = [0, 1], reduction = [0, 0, 0, 0, 0, 2], subgroup = [1, 6, 1, 1, 4, 0], subgroup_m_count = 2 : i64, subgroup_n_count = 2 : i64, workgroup = [1, 12, 16, 1, 128, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [256, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module896857893101354896_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @"module-4520889304652789583_match_conv_2d_bfloat16_forward_128x24x48x384_nhwc_384x1x1x384_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_147456x384x384_bf16xbf16xf32"(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<147456x384xbf16>, %arg2: tensor<384x384xbf16>, %arg3: tensor<147456x384xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<147456x384xbf16>, tensor<384x384xbf16>) outs(%arg3 : tensor<147456x384xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<147456x384xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_BF16>, promote_operands = [0, 1], reduction = [0, 0, 4], subgroup = [2, 3, 0], subgroup_m_count = 2 : i64, subgroup_n_count = 4 : i64, workgroup = [128, 384, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [512, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @"module-522984183083508851_apply_op_config"(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_128x24x48x128_nhwc_384x1x1x128_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_147456x384x128_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<147456x128xbf16>, %arg2: tensor<384x128xbf16>, %arg3: tensor<147456x384xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<147456x128xbf16>, tensor<384x128xbf16>) outs(%arg3 : tensor<147456x384xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<147456x384xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_BF16>, promote_operands = [0, 1], reduction = [0, 0, 4], subgroup = [1, 4, 0], subgroup_m_count = 8 : i64, subgroup_n_count = 1 : i64, workgroup = [256, 128, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [512, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module4071637556353897261_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @module5265578897403436880_match_conv_2d_bfloat16_forward_128x48x32x384_nhwc_384x1x1x384_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_196608x384x384_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<196608x384xbf16>, %arg2: tensor<384x384xbf16>, %arg3: tensor<196608x384xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<196608x384xbf16>, tensor<384x384xbf16>) outs(%arg3 : tensor<196608x384xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<196608x384xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_BF16>, promote_operands = [0, 1], reduction = [0, 0, 4], subgroup = [4, 1, 0], subgroup_m_count = 1 : i64, subgroup_n_count = 4 : i64, workgroup = [128, 128, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [256, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @"module-4936958348168552443_apply_op_config"(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_b_16x48x32x2048_nhwc_2048x3x1x2048_fhwc_nhwf_3x1s_1x0p_1x1d_1g$async_dispatch_0_conv_16x16x32x2048x3x2048_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<16x50x32x2048xbf16>, %arg2: tensor<2048x3x2048xbf16>, %arg3: tensor<16x16x32x2048xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1 * 3 + d4, d2, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d4, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<16x50x32x2048xbf16>, tensor<2048x3x2048xbf16>) outs(%arg3 : tensor<16x16x32x2048xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<16x16x32x2048xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, promote_operands = [0, 1], reduction = [0, 0, 0, 0, 2], subgroup = [4, 1, 1, 8, 0], subgroup_m_count = 4 : i64, subgroup_n_count = 2 : i64, workgroup = [16, 1, 16, 256, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [512, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @"module-101477417419537001_apply_op_config"(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @"module-6795424626653063444_match_conv_2d_bfloat16_forward_128x24x48x128_nhwc_384x1x1x128_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_147456x384x128_bf16xbf16xf32"(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<147456x128xbf16>, %arg2: tensor<384x128xbf16>, %arg3: tensor<147456x384xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<147456x128xbf16>, tensor<384x128xbf16>) outs(%arg3 : tensor<147456x384xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<147456x384xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_BF16>, promote_operands = [0, 1], reduction = [0, 0, 4], subgroup = [2, 4, 0], subgroup_m_count = 8 : i64, subgroup_n_count = 1 : i64, workgroup = [512, 128, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [512, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module1730547502948057187_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @"module-3096228896624724732_match_conv_2d_bfloat16_forward_128x48x32x384_nhwc_384x3x3x64_fhwc_nhwf_1x1s_1x1p_1x1d_6g$async_dispatch_0_conv_128x48x32x6x64x3x3x64_bf16xbf16xf32"(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<128x50x34x6x64xbf16>, %arg2: tensor<6x64x3x3x64xbf16>, %arg3: tensor<128x48x32x6x64xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1 + d5, d2 + d6, d3, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d3, d4, d5, d6, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<128x50x34x6x64xbf16>, tensor<6x64x3x3x64xbf16>) outs(%arg3 : tensor<128x48x32x6x64xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<128x48x32x6x64xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_BF16>, promote_operands = [0, 1], reduction = [0, 0, 0, 0, 0, 4], subgroup = [1, 2, 1, 1, 2, 0], subgroup_m_count = 4 : i64, subgroup_n_count = 1 : i64, workgroup = [4, 2, 32, 1, 64, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [256, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @"module-6050217721080824708_apply_op_config"(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @"module-3610251752505085311_match_conv_2d_bfloat16_forward_16x48x32x576_nhwc_576x3x3x576_fhwc_nhwf_1x1s_1x1p_1x1d_1g$async_dispatch_0_conv_16x48x32x576x3x3x576_bf16xbf16xf32"(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<16x50x34x576xbf16>, %arg2: tensor<576x3x3x576xbf16>, %arg3: tensor<16x48x32x576xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d4, d2 + d5, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d3, d4, d5, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<16x50x34x576xbf16>, tensor<576x3x3x576xbf16>) outs(%arg3 : tensor<16x48x32x576xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<16x48x32x576xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, promote_operands = [0, 1], reduction = [0, 0, 0, 0, 4], subgroup = [8, 1, 1, 3, 0], subgroup_m_count = 2 : i64, subgroup_n_count = 4 : i64, workgroup = [16, 1, 16, 192, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [512, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module7250346489746609524_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @module4826834601099627801_match_conv_2d_bfloat16_forward_16x48x32x768_nhwc_2048x3x3x768_fhwc_nhwf_1x1s_1x1p_1x1d_1g$async_dispatch_0_conv_16x48x32x2048x3x3x768_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
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
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, promote_operands = [0, 1], reduction = [0, 0, 0, 0, 2], subgroup = [4, 2, 1, 4, 0], subgroup_m_count = 2 : i64, subgroup_n_count = 4 : i64, workgroup = [8, 2, 16, 256, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [512, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @"module-6880784253496819418_apply_op_config"(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @module6151868810080441787_match_conv_2d_bfloat16_forward_128x24x48x480_nhwc_384x1x1x480_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_147456x384x480_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<147456x480xbf16>, %arg2: tensor<384x480xbf16>, %arg3: tensor<147456x384xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<147456x480xbf16>, tensor<384x480xbf16>) outs(%arg3 : tensor<147456x384xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<147456x384xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_BF16>, promote_operands = [0, 1], reduction = [0, 0, 4], subgroup = [4, 3, 0], subgroup_m_count = 1 : i64, subgroup_n_count = 2 : i64, workgroup = [128, 192, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [128, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @"module-5484525665775309725_apply_op_config"(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @"module-5713853901338332798_match_conv_2d_bfloat16_forward_128x24x48x384_nhwc_512x1x1x384_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_147456x512x384_bf16xbf16xf32"(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<147456x384xbf16>, %arg2: tensor<512x384xbf16>, %arg3: tensor<147456x512xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<147456x384xbf16>, tensor<512x384xbf16>) outs(%arg3 : tensor<147456x512xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<147456x512xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, promote_operands = [0, 1], reduction = [0, 0, 4], subgroup = [2, 2, 0], subgroup_m_count = 2 : i64, subgroup_n_count = 4 : i64, workgroup = [64, 128, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [512, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @"module-5578073272369513136_apply_op_config"(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @"module-522984183083508851_match_conv_2d_bfloat16_forward_128x24x48x384_nhwc_384x1x1x384_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_147456x384x384_bf16xbf16xf32"(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<147456x384xbf16>, %arg2: tensor<384x384xbf16>, %arg3: tensor<147456x384xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<147456x384xbf16>, tensor<384x384xbf16>) outs(%arg3 : tensor<147456x384xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<147456x384xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_BF16>, promote_operands = [0, 1], reduction = [0, 0, 4], subgroup = [1, 2, 0], subgroup_m_count = 4 : i64, subgroup_n_count = 2 : i64, workgroup = [128, 128, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [512, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module8850980168311619132_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @"module-3307929099139602236_match_conv_2d_bfloat16_forward_128x24x48x384_nhwc_384x3x1x128_fhwc_nhwf_1x1s_1x0p_1x1d_3g$async_dispatch_0_conv_128x24x48x3x128x3x128_bf16xbf16xf32"(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<128x26x48x3x128xbf16>, %arg2: tensor<3x128x3x128xbf16>, %arg3: tensor<128x24x48x3x128xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d5, d2, d3, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d3, d4, d5, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<128x26x48x3x128xbf16>, tensor<3x128x3x128xbf16>) outs(%arg3 : tensor<128x24x48x3x128xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<128x24x48x3x128xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, promote_operands = [0, 1], reduction = [0, 0, 0, 0, 0, 2], subgroup = [8, 1, 1, 1, 4, 0], subgroup_m_count = 4 : i64, subgroup_n_count = 2 : i64, workgroup = [32, 1, 16, 1, 128, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [512, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @"module-911083449296662588_apply_op_config"(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @module896857893101354896_match_conv_2d_bfloat16_forward_128x24x48x384_nhwc_384x1x3x128_fhwc_nhwf_1x1s_0x1p_1x1d_3g$async_dispatch_0_conv_128x24x48x3x128x3x128_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<128x24x50x3x128xbf16>, %arg2: tensor<3x128x3x128xbf16>, %arg3: tensor<128x24x48x3x128xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2 + d5, d3, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d3, d4, d5, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<128x24x50x3x128xbf16>, tensor<3x128x3x128xbf16>) outs(%arg3 : tensor<128x24x48x3x128xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<128x24x48x3x128xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, promote_operands = [0, 1], reduction = [0, 0, 0, 0, 0, 2], subgroup = [2, 2, 1, 1, 4, 0], subgroup_m_count = 2 : i64, subgroup_n_count = 2 : i64, workgroup = [2, 4, 16, 1, 128, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [256, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module4742795371244143238_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_16x8x32x576_nhwc_576x1x1x576_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_4096x576x576_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<4096x576xbf16>, %arg2: tensor<576x576xbf16>, %arg3: tensor<4096x576xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<4096x576xbf16>, tensor<576x576xbf16>) outs(%arg3 : tensor<4096x576xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<4096x576xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, promote_operands = [0, 1], reduction = [0, 0, 12], subgroup = [1, 3, 0], subgroup_m_count = 2 : i64, subgroup_n_count = 2 : i64, workgroup = [32, 96, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [256, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module6851109957392407656_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @module7014507521830355182_match_conv_3d_bfloat16_forward_b_16x2x48x32x288_ndhwc_288x3x1x1x96_fdhwc_ndhwf_1x1x1s_1x0x0p_1x1x1d_3g$async_dispatch_0_conv_16x2x48x32x3x96x3x96_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<16x4x48x32x3x96xbf16>, %arg2: tensor<3x96x3x96xbf16>, %arg3: tensor<16x2x48x32x3x96xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1 + d6, d2, d3, d4, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d4, d5, d6, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<16x4x48x32x3x96xbf16>, tensor<3x96x3x96xbf16>) outs(%arg3 : tensor<16x2x48x32x3x96xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<16x2x48x32x3x96xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_BF16>, promote_operands = [0, 1], reduction = [0, 0, 0, 0, 0, 0, 4], subgroup = [2, 1, 3, 1, 1, 1, 0], subgroup_m_count = 1 : i64, subgroup_n_count = 3 : i64, workgroup = [2, 1, 3, 32, 1, 96, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [192, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @"module-8084250599234373333_apply_op_config"(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @module3276315160131382713_match_conv_3d_bfloat16_forward_b_16x2x48x32x288_ndhwc_288x1x3x3x96_fdhwc_ndhwf_1x1x1s_0x1x1p_1x1x1d_3g$async_dispatch_0_conv_16x2x48x32x3x96x3x3x96_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<16x2x50x34x3x96xbf16>, %arg2: tensor<3x96x3x3x96xbf16>, %arg3: tensor<16x2x48x32x3x96xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d2 + d6, d3 + d7, d4, d8)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d4, d5, d6, d7, d8)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<16x2x50x34x3x96xbf16>, tensor<3x96x3x3x96xbf16>) outs(%arg3 : tensor<16x2x48x32x3x96xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<16x2x48x32x3x96xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_BF16>, promote_operands = [0, 1], reduction = [0, 0, 0, 0, 0, 0, 4], subgroup = [1, 1, 4, 1, 1, 3, 0], subgroup_m_count = 2 : i64, subgroup_n_count = 1 : i64, workgroup = [2, 1, 4, 32, 1, 96, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [128, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @"module-998895541307183783_apply_op_config"(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @"module-1970581355946039598_match_conv_3d_bfloat16_forward_b_16x4x48x32x288_ndhwc_288x3x1x1x96_fdhwc_ndhwf_1x1x1s_1x0x0p_1x1x1d_3g$async_dispatch_0_conv_16x4x48x32x3x96x3x96_bf16xbf16xf32"(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<16x6x48x32x3x96xbf16>, %arg2: tensor<3x96x3x96xbf16>, %arg3: tensor<16x4x48x32x3x96xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1 + d6, d2, d3, d4, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d4, d5, d6, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<16x6x48x32x3x96xbf16>, tensor<3x96x3x96xbf16>) outs(%arg3 : tensor<16x4x48x32x3x96xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<16x4x48x32x3x96xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_BF16>, promote_operands = [0, 1], reduction = [0, 0, 0, 0, 0, 0, 4], subgroup = [1, 4, 1, 1, 1, 3, 0], subgroup_m_count = 2 : i64, subgroup_n_count = 1 : i64, workgroup = [2, 4, 1, 32, 1, 96, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [128, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module8795648018915444682_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @module7216368369122952845_match_conv_3d_bfloat16_forward_b_16x4x48x32x288_ndhwc_288x1x3x3x96_fdhwc_ndhwf_1x1x1s_0x1x1p_1x1x1d_3g$async_dispatch_0_conv_16x4x48x32x3x96x3x3x96_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<16x4x50x34x3x96xbf16>, %arg2: tensor<3x96x3x3x96xbf16>, %arg3: tensor<16x4x48x32x3x96xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d2 + d6, d3 + d7, d4, d8)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d4, d5, d6, d7, d8)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<16x4x50x34x3x96xbf16>, tensor<3x96x3x3x96xbf16>) outs(%arg3 : tensor<16x4x48x32x3x96xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<16x4x48x32x3x96xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, promote_operands = [0, 1], reduction = [0, 0, 0, 0, 0, 0, 2], subgroup = [1, 2, 2, 2, 1, 3, 0], subgroup_m_count = 1 : i64, subgroup_n_count = 2 : i64, workgroup = [1, 2, 2, 32, 1, 96, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [128, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @"module-317436726597910684_apply_op_config"(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @module6909153255479619441_match_conv_3d_bfloat16_forward_b_16x8x48x32x288_ndhwc_288x3x1x1x96_fdhwc_ndhwf_1x1x1s_1x0x0p_1x1x1d_3g$async_dispatch_0_conv_16x8x48x32x3x96x3x96_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<16x10x48x32x3x96xbf16>, %arg2: tensor<3x96x3x96xbf16>, %arg3: tensor<16x8x48x32x3x96xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1 + d6, d2, d3, d4, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d4, d5, d6, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<16x10x48x32x3x96xbf16>, tensor<3x96x3x96xbf16>) outs(%arg3 : tensor<16x8x48x32x3x96xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<16x8x48x32x3x96xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, promote_operands = [0, 1], reduction = [0, 0, 0, 0, 0, 0, 2], subgroup = [1, 1, 1, 2, 1, 6, 0], subgroup_m_count = 3 : i64, subgroup_n_count = 1 : i64, workgroup = [1, 1, 3, 32, 1, 96, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [192, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module8983234031431727099_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @module3515167887415924444_match_conv_3d_bfloat16_forward_b_16x8x48x32x288_ndhwc_288x1x3x3x96_fdhwc_ndhwf_1x1x1s_0x1x1p_1x1x1d_3g$async_dispatch_0_conv_16x8x48x32x3x96x3x3x96_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<16x8x50x34x3x96xbf16>, %arg2: tensor<3x96x3x3x96xbf16>, %arg3: tensor<16x8x48x32x3x96xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d2 + d6, d3 + d7, d4, d8)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d4, d5, d6, d7, d8)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<16x8x50x34x3x96xbf16>, tensor<3x96x3x3x96xbf16>) outs(%arg3 : tensor<16x8x48x32x3x96xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<16x8x48x32x3x96xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, promote_operands = [0, 1], reduction = [0, 0, 0, 0, 0, 0, 2], subgroup = [1, 1, 2, 1, 1, 3, 0], subgroup_m_count = 3 : i64, subgroup_n_count = 2 : i64, workgroup = [1, 1, 6, 16, 1, 96, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [384, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @module8396357738012441017_apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_conv_2d_bfloat16_forward_16x225x225x13_nhwc_64x3x3x13_fhwc_nhwf_1x1s_1x1p_1x1d_1g$async_dispatch_0_conv_16x225x225x64x3x3x13_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<16x227x227x13xbf16>, %arg2: tensor<64x3x3x13xbf16>, %arg3: tensor<16x225x225x64xf32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d4, d2 + d5, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d3, d4, d5, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<16x227x227x13xbf16>, tensor<64x3x3x13xbf16>) outs(%arg3 : tensor<16x225x225x64xf32>) {
      ^bb0(%in: bf16, %in_0: bf16, %out: f32):
        %2 = arith.extf %in : bf16 to f32
        %3 = arith.extf %in_0 : bf16 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<16x225x225x64xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, padding = [2, 1, 16, 64, 16], promote_operands = [0, 1, 2], reduction = [0, 0, 0, 0, 1], subgroup = [1, 1, 1, 1, 0], subgroup_m_count = 2 : i64, subgroup_n_count = 4 : i64, workgroup = [2, 1, 16, 64, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [512, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
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
  transform.named_sequence @__kernel_config(%arg0: !transform.any_op {transform.consumed}) -> !transform.any_op attributes {iree_codegen.tuning_spec_entrypoint} {
    %updated_root = transform.foreach_match in %arg0
        @match_conv_2d_bfloat16_forward_3x1x1x112_nhwc_448x1x1x112_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_3x448x112_bf16xbf16xf32 -> @apply_op_config,
        @match_conv_2d_bfloat16_forward_1x118x182x448_nhwc_896x1x1x448_fhwc_nhwf_2x2s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_59x91x896x448_bf16xbf16xf32 -> @module4559739451968120734_apply_op_config,
        @match_conv_2d_bfloat16_forward_2x235x363x224_nhwc_448x1x1x224_fhwc_nhwf_2x2s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_2x118x182x448x224_bf16xbf16xf32 -> @module7882439876625634314_apply_op_config,
        @match_conv_2d_bfloat16_forward_7x1x1x2016_nhwc_224x1x1x2016_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_7x224x2016_bf16xbf16xf32 -> @"module-2995968965573836068_apply_op_config",
        @match_conv_2d_bfloat16_forward_10x1x1x2016_nhwc_224x1x1x2016_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_10x224x2016_bf16xbf16xf32 -> @"module-1383557259038100597_apply_op_config",
        @match_conv_2d_bfloat16_forward_12x1x1x2016_nhwc_224x1x1x2016_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_12x224x2016_bf16xbf16xf32 -> @module2820520407618095204_apply_op_config,
        @match_conv_2d_bfloat16_forward_6x1x1x2016_nhwc_224x1x1x2016_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_6x224x2016_bf16xbf16xf32 -> @module3713156600714724778_apply_op_config,
        @match_conv_2d_bfloat16_forward_1x59x91x896_nhwc_2016x1x1x896_fhwc_nhwf_2x2s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_30x46x2016x896_bf16xbf16xf32 -> @module8560219421819835492_apply_op_config,
        @match_conv_2d_bfloat16_forward_1x235x363x224_nhwc_448x1x1x224_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_235x363x448x224_bf16xbf16xf32 -> @module5028598877358614442_apply_op_config,
        @match_conv_2d_bfloat16_forward_5x1x1x2016_nhwc_224x1x1x2016_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_5x224x2016_bf16xbf16xf32 -> @module2675858531162868376_apply_op_config,
        @match_conv_2d_bfloat16_forward_1x30x46x2016_nhwc_2048x1x1x2016_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_30x46x2048x2016_bf16xbf16xf32 -> @module8272023259564752550_apply_op_config,
        @match_conv_2d_bfloat16_forward_2x30x46x2016_nhwc_1536x1x1x2016_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_2760x1536x2016_bf16xbf16xf32 -> @module3773559851783729558_apply_op_config,
        @match_conv_2d_bfloat16_forward_1x30x46x2016_nhwc_2016x1x1x2016_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_30x46x2016x2016_bf16xbf16xf32 -> @module6802500929324229951_apply_op_config,
        @match_conv_2d_bfloat16_forward_4x1x1x112_nhwc_896x1x1x112_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_4x896x112_bf16xbf16xf32 -> @"module-7001239958652609188_apply_op_config",
        @match_conv_2d_bfloat16_forward_1x118x182x448_nhwc_896x1x1x448_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_118x182x896x448_bf16xbf16xf32 -> @"module-993111789093238903_apply_op_config",
        @match_conv_2d_bfloat16_forward_4x118x182x448_nhwc_896x1x1x448_fhwc_nhwf_2x2s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_4x59x91x896x448_bf16xbf16xf32 -> @module5885234969857106962_apply_op_config,
        @match_conv_2d_bfloat16_forward_2x1x1x224_nhwc_2016x1x1x224_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_2x2016x224_bf16xbf16xf32 -> @module4586823122037315142_apply_op_config,
        @match_conv_2d_bfloat16_forward_4x1x1x224_nhwc_2016x1x1x224_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_4x2016x224_bf16xbf16xf32 -> @module356981495459478889_apply_op_config,
        @match_conv_2d_bfloat16_forward_4x30x46x2016_nhwc_1536x1x1x2016_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_5520x1536x2016_bf16xbf16xf32 -> @"module-8170860178698619477_apply_op_config",
        @match_conv_2d_bfloat16_forward_3x1x1x224_nhwc_2016x1x1x224_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_3x2016x224_bf16xbf16xf32 -> @module7994357265823291861_apply_op_config,
        @match_conv_2d_bfloat16_forward_5x30x46x2016_nhwc_768x1x1x2016_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_6900x768x2016_bf16xbf16xf32 -> @module7431102081689966800_apply_op_config,
        @match_conv_2d_bfloat16_forward_1x59x91x896_nhwc_2016x1x1x896_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_59x91x2016x896_bf16xbf16xf32 -> @module4332867699959355936_apply_op_config,
        @match_conv_2d_bfloat16_forward_4x1x1x112_nhwc_448x1x1x112_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_4x448x112_bf16xbf16xf32 -> @"module-806051145099938734_apply_op_config",
        @match_conv_2d_bfloat16_forward_2x59x91x896_nhwc_2016x1x1x896_fhwc_nhwf_2x2s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_2x30x46x2016x896_bf16xbf16xf32 -> @"module-7676363064080114619_apply_op_config",
        @match_conv_2d_bfloat16_forward_3x59x91x896_nhwc_2016x1x1x896_fhwc_nhwf_2x2s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_3x30x46x2016x896_bf16xbf16xf32 -> @module6588666152883867170_apply_op_config,
        @match_conv_2d_bfloat16_forward_3x59x91x2016_nhwc_2016x3x3x56_fhwc_nhwf_2x2s_1x1p_1x1d_36g$async_dispatch_0_conv_3x30x46x36x56x3x3x56_bf16xbf16xf32 -> @"module-8030569889332935828_apply_op_config",
        @match_conv_2d_bfloat16_forward_2x59x91x896_nhwc_2016x1x1x896_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_10738x2016x896_bf16xbf16xf32 -> @module5363443944264822312_apply_op_config,
        @match_conv_2d_bfloat16_forward_4x59x91x896_nhwc_2016x1x1x896_fhwc_nhwf_2x2s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_4x30x46x2016x896_bf16xbf16xf32 -> @module2458017827010063645_apply_op_config,
        @match_conv_2d_bfloat16_forward_4x59x91x2016_nhwc_2016x3x3x56_fhwc_nhwf_2x2s_1x1p_1x1d_36g$async_dispatch_0_conv_4x30x46x36x56x3x3x56_bf16xbf16xf32 -> @"module-8605844983155302868_apply_op_config",
        @match_conv_2d_bfloat16_forward_4x30x46x2016_nhwc_2048x1x1x2016_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_5520x2048x2016_bf16xbf16xf32 -> @module4543929421270441657_apply_op_config,
        @match_conv_2d_bfloat16_forward_5x30x46x2016_nhwc_1536x1x1x2016_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_6900x1536x2016_bf16xbf16xf32 -> @module7946456294075948850_apply_op_config,
        @match_conv_2d_bfloat16_forward_4x30x46x2016_nhwc_2016x1x1x2016_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_5520x2016x2016_bf16xbf16xf32 -> @module855441149224401939_apply_op_config,
        @match_conv_2d_bfloat16_forward_4x118x182x448_nhwc_896x1x1x448_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_85904x896x448_bf16xbf16xf32 -> @"module-7843621978812482176_apply_op_config",
        @match_conv_2d_bfloat16_forward_5x59x91x896_nhwc_2016x1x1x896_fhwc_nhwf_2x2s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_5x30x46x2016x896_bf16xbf16xf32 -> @"module-2373990324862608393_apply_op_config",
        @match_conv_2d_bfloat16_forward_1x235x363x448_nhwc_448x3x3x56_fhwc_nhwf_2x2s_1x1p_1x1d_8g$async_dispatch_0_conv_118x182x8x56x3x3x56_bf16xbf16xf32 -> @module8288620826883058744_apply_op_config,
        @match_conv_2d_bfloat16_forward_2x1x1x224_nhwc_896x1x1x224_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_2x896x224_bf16xbf16xf32 -> @module2686472097007468535_apply_op_config,
        @match_conv_2d_bfloat16_forward_3x1x1x224_nhwc_896x1x1x224_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_3x896x224_bf16xbf16xf32 -> @module7325674527831915629_apply_op_config,
        @match_conv_2d_bfloat16_forward_10x30x46x2016_nhwc_768x1x1x2016_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_13800x768x2016_bf16xbf16xf32 -> @"module-6003646264170329989_apply_op_config",
        @match_conv_2d_bfloat16_forward_1x59x91x896_nhwc_896x1x1x896_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_59x91x896x896_bf16xbf16xf32 -> @"module-4360576302824214298_apply_op_config",
        @match_conv_2d_bfloat16_forward_12x30x46x2016_nhwc_1536x1x1x2016_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_16560x1536x2016_bf16xbf16xf32 -> @module8668743119509608937_apply_op_config,
        @match_conv_2d_bfloat16_forward_6x59x91x896_nhwc_2016x1x1x896_fhwc_nhwf_2x2s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_6x30x46x2016x896_bf16xbf16xf32 -> @"module-6940871244302246445_apply_op_config",
        @match_conv_2d_bfloat16_forward_1x235x363x224_nhwc_224x3x3x56_fhwc_nhwf_1x1s_1x1p_1x1d_4g$async_dispatch_0_conv_235x363x4x56x3x3x56_bf16xbf16xf32 -> @"module-1821426178339838639_apply_op_config",
        @match_conv_2d_bfloat16_forward_12x30x46x2016_nhwc_768x1x1x2016_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_16560x768x2016_bf16xbf16xf32 -> @module4439023163120043581_apply_op_config,
        @match_conv_2d_bfloat16_forward_7x59x91x896_nhwc_2016x1x1x896_fhwc_nhwf_2x2s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_7x30x46x2016x896_bf16xbf16xf32 -> @"module-1401953935242321448_apply_op_config",
        @match_conv_2d_bfloat16_forward_1x470x725x224_nhwc_224x3x3x56_fhwc_nhwf_2x2s_1x1p_1x1d_4g$async_dispatch_0_conv_235x363x4x56x3x3x56_bf16xbf16xf32 -> @module973092433758697018_apply_op_config,
        @match_conv_2d_bfloat16_forward_12x118x182x448_nhwc_896x1x1x448_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_257712x896x448_bf16xbf16xf32 -> @module6365494400897409164_apply_op_config,
        @match_conv_2d_bfloat16_forward_2x235x363x448_nhwc_448x3x3x56_fhwc_nhwf_2x2s_1x1p_1x1d_8g$async_dispatch_0_conv_2x118x182x8x56x3x3x56_bf16xbf16xf32 -> @module4462116553785334358_apply_op_config,
        @match_conv_2d_bfloat16_forward_4x118x182x448_nhwc_448x1x1x448_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_85904x448x448_bf16xbf16xf32 -> @module3453032289433897304_apply_op_config,
        @match_conv_2d_bfloat16_forward_10x59x91x896_nhwc_2016x1x1x896_fhwc_nhwf_2x2s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_10x30x46x2016x896_bf16xbf16xf32 -> @module4367576303115157843_apply_op_config,
        @match_conv_2d_bfloat16_forward_2x118x182x896_nhwc_896x3x3x56_fhwc_nhwf_2x2s_1x1p_1x1d_16g$async_dispatch_0_conv_2x59x91x16x56x3x3x56_bf16xbf16xf32 -> @"module-4906234458156126534_apply_op_config",
        @match_conv_2d_bfloat16_forward_1x118x182x448_nhwc_448x3x3x56_fhwc_nhwf_1x1s_1x1p_1x1d_8g$async_dispatch_0_conv_118x182x8x56x3x3x56_bf16xbf16xf32 -> @module8109018211679045215_apply_op_config,
        @match_conv_2d_bfloat16_forward_3x235x363x448_nhwc_448x3x3x56_fhwc_nhwf_2x2s_1x1p_1x1d_8g$async_dispatch_0_conv_3x118x182x8x56x3x3x56_bf16xbf16xf32 -> @module4014810933135631207_apply_op_config,
        @match_conv_2d_bfloat16_forward_12x30x46x2016_nhwc_2016x1x1x2016_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_16560x2016x2016_bf16xbf16xf32 -> @"module-8383162261389945333_apply_op_config",
        @match_conv_2d_bfloat16_forward_2x1x1x2016_nhwc_224x1x1x2016_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_2x224x2016_bf16xbf16xf32 -> @module1133905960676286034_apply_op_config,
        @match_conv_2d_bfloat16_forward_4x1x1x2016_nhwc_224x1x1x2016_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_4x224x2016_bf16xbf16xf32 -> @"module-282786097804479605_apply_op_config",
        @match_conv_2d_bfloat16_forward_12x59x91x896_nhwc_2016x1x1x896_fhwc_nhwf_2x2s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_12x30x46x2016x896_bf16xbf16xf32 -> @module300891171000512914_apply_op_config,
        @match_conv_2d_bfloat16_forward_4x1x1x224_nhwc_896x1x1x224_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_4x896x224_bf16xbf16xf32 -> @"module-4816075757461376162_apply_op_config",
        @match_conv_2d_bfloat16_forward_4x235x363x448_nhwc_448x3x3x56_fhwc_nhwf_2x2s_1x1p_1x1d_8g$async_dispatch_0_conv_4x118x182x8x56x3x3x56_bf16xbf16xf32 -> @module9111302107637546479_apply_op_config,
        @match_conv_2d_bfloat16_forward_3x1x1x2016_nhwc_224x1x1x2016_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_3x224x2016_bf16xbf16xf32 -> @module7616000247919958829_apply_op_config,
        @match_conv_2d_bfloat16_forward_4x118x182x896_nhwc_896x3x3x56_fhwc_nhwf_2x2s_1x1p_1x1d_16g$async_dispatch_0_conv_4x59x91x16x56x3x3x56_bf16xbf16xf32 -> @module5155454980444068234_apply_op_config,
        @match_conv_2d_bfloat16_forward_6x118x182x896_nhwc_896x3x3x56_fhwc_nhwf_2x2s_1x1p_1x1d_16g$async_dispatch_0_conv_6x59x91x16x56x3x3x56_bf16xbf16xf32 -> @"module-8844875856657637720_apply_op_config",
        @match_conv_2d_bfloat16_forward_5x235x363x448_nhwc_448x3x3x56_fhwc_nhwf_2x2s_1x1p_1x1d_8g$async_dispatch_0_conv_5x118x182x8x56x3x3x56_bf16xbf16xf32 -> @module4471347617703089637_apply_op_config,
        @match_conv_2d_bfloat16_forward_10x118x182x896_nhwc_896x3x3x56_fhwc_nhwf_2x2s_1x1p_1x1d_16g$async_dispatch_0_conv_10x59x91x16x56x3x3x56_bf16xbf16xf32 -> @"module-507249471843353843_apply_op_config",
        @match_conv_2d_bfloat16_forward_2x118x182x448_nhwc_448x3x3x56_fhwc_nhwf_1x1s_1x1p_1x1d_8g$async_dispatch_0_conv_2x118x182x8x56x3x3x56_bf16xbf16xf32 -> @"module-7376692197419214668_apply_op_config",
        @match_conv_2d_bfloat16_forward_12x118x182x896_nhwc_896x3x3x56_fhwc_nhwf_2x2s_1x1p_1x1d_16g$async_dispatch_0_conv_12x59x91x16x56x3x3x56_bf16xbf16xf32 -> @module2954593435895805179_apply_op_config,
        @match_conv_2d_bfloat16_forward_6x235x363x448_nhwc_448x3x3x56_fhwc_nhwf_2x2s_1x1p_1x1d_8g$async_dispatch_0_conv_6x118x182x8x56x3x3x56_bf16xbf16xf32 -> @module1498352111523621947_apply_op_config,
        @match_conv_2d_bfloat16_forward_7x235x363x448_nhwc_448x3x3x56_fhwc_nhwf_2x2s_1x1p_1x1d_8g$async_dispatch_0_conv_7x118x182x8x56x3x3x56_bf16xbf16xf32 -> @"module-4591256468792391059_apply_op_config",
        @match_conv_2d_bfloat16_forward_10x235x363x448_nhwc_448x3x3x56_fhwc_nhwf_2x2s_1x1p_1x1d_8g$async_dispatch_0_conv_10x118x182x8x56x3x3x56_bf16xbf16xf32 -> @"module-3494160608888899646_apply_op_config",
        @match_conv_2d_bfloat16_forward_2x59x91x896_nhwc_896x1x1x896_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_10738x896x896_bf16xbf16xf32 -> @"module-5609055561883516000_apply_op_config",
        @match_conv_2d_bfloat16_forward_12x118x182x448_nhwc_448x1x1x448_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_257712x448x448_bf16xbf16xf32 -> @module7151347827273063051_apply_op_config,
        @match_conv_2d_bfloat16_forward_3x118x182x448_nhwc_448x3x3x56_fhwc_nhwf_1x1s_1x1p_1x1d_8g$async_dispatch_0_conv_3x118x182x8x56x3x3x56_bf16xbf16xf32 -> @"module-6365402289902751883_apply_op_config",
        @match_conv_2d_bfloat16_forward_4x118x182x448_nhwc_448x3x3x56_fhwc_nhwf_1x1s_1x1p_1x1d_8g$async_dispatch_0_conv_4x118x182x8x56x3x3x56_bf16xbf16xf32 -> @"module-2802817007326691698_apply_op_config",
        @match_conv_2d_bfloat16_forward_3x59x91x896_nhwc_896x1x1x896_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_16107x896x896_bf16xbf16xf32 -> @"module-8847663828986506085_apply_op_config",
        @match_conv_2d_bfloat16_forward_2x470x725x224_nhwc_224x3x3x56_fhwc_nhwf_2x2s_1x1p_1x1d_4g$async_dispatch_0_conv_2x235x363x4x56x3x3x56_bf16xbf16xf32 -> @"module-1484268700394101154_apply_op_config",
        @match_conv_2d_bfloat16_forward_2x235x363x224_nhwc_224x3x3x56_fhwc_nhwf_1x1s_1x1p_1x1d_4g$async_dispatch_0_conv_2x235x363x4x56x3x3x56_bf16xbf16xf32 -> @"module-2678622183367858380_apply_op_config",
        @match_conv_2d_bfloat16_forward_3x470x725x224_nhwc_224x3x3x56_fhwc_nhwf_2x2s_1x1p_1x1d_4g$async_dispatch_0_conv_3x235x363x4x56x3x3x56_bf16xbf16xf32 -> @module1357578566083234864_apply_op_config,
        @match_conv_2d_bfloat16_forward_4x59x91x896_nhwc_896x3x3x56_fhwc_nhwf_1x1s_1x1p_1x1d_16g$async_dispatch_0_conv_4x59x91x16x56x3x3x56_bf16xbf16xf32 -> @"module-7272548824195366221_apply_op_config",
        @match_conv_2d_bfloat16_forward_12x235x363x448_nhwc_448x3x3x56_fhwc_nhwf_2x2s_1x1p_1x1d_8g$async_dispatch_0_conv_12x118x182x8x56x3x3x56_bf16xbf16xf32 -> @module3767595398975031806_apply_op_config,
        @match_conv_2d_bfloat16_forward_4x59x91x896_nhwc_896x1x1x896_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_21476x896x896_bf16xbf16xf32 -> @module4768356428233192197_apply_op_config,
        @match_conv_2d_bfloat16_forward_5x118x182x448_nhwc_448x3x3x56_fhwc_nhwf_1x1s_1x1p_1x1d_8g$async_dispatch_0_conv_5x118x182x8x56x3x3x56_bf16xbf16xf32 -> @"module-8175233066150774447_apply_op_config",
        @match_conv_2d_bfloat16_forward_3x235x363x224_nhwc_224x3x3x56_fhwc_nhwf_1x1s_1x1p_1x1d_4g$async_dispatch_0_conv_3x235x363x4x56x3x3x56_bf16xbf16xf32 -> @module5997569232756307197_apply_op_config,
        @match_conv_2d_bfloat16_forward_4x470x725x224_nhwc_224x3x3x56_fhwc_nhwf_2x2s_1x1p_1x1d_4g$async_dispatch_0_conv_4x235x363x4x56x3x3x56_bf16xbf16xf32 -> @"module-9157235887781922563_apply_op_config",
        @match_conv_2d_bfloat16_forward_6x118x182x448_nhwc_448x3x3x56_fhwc_nhwf_1x1s_1x1p_1x1d_8g$async_dispatch_0_conv_6x118x182x8x56x3x3x56_bf16xbf16xf32 -> @module6208162720997979308_apply_op_config,
        @match_conv_2d_bfloat16_forward_5x59x91x896_nhwc_896x1x1x896_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_26845x896x896_bf16xbf16xf32 -> @"module-8008214301341210612_apply_op_config",
        @match_conv_2d_bfloat16_forward_5x470x725x224_nhwc_224x3x3x56_fhwc_nhwf_2x2s_1x1p_1x1d_4g$async_dispatch_0_conv_5x235x363x4x56x3x3x56_bf16xbf16xf32 -> @"module-4277437435457599128_apply_op_config",
        @match_conv_2d_bfloat16_forward_4x235x363x224_nhwc_224x3x3x56_fhwc_nhwf_1x1s_1x1p_1x1d_4g$async_dispatch_0_conv_4x235x363x4x56x3x3x56_bf16xbf16xf32 -> @"module-5567820896292441710_apply_op_config",
        @match_conv_2d_bfloat16_forward_7x118x182x448_nhwc_448x3x3x56_fhwc_nhwf_1x1s_1x1p_1x1d_8g$async_dispatch_0_conv_7x118x182x8x56x3x3x56_bf16xbf16xf32 -> @"module-4426083179023641467_apply_op_config",
        @match_conv_2d_bfloat16_forward_6x59x91x896_nhwc_896x3x3x56_fhwc_nhwf_1x1s_1x1p_1x1d_16g$async_dispatch_0_conv_6x59x91x16x56x3x3x56_bf16xbf16xf32 -> @module7624307817249334277_apply_op_config,
        @match_conv_2d_bfloat16_forward_6x470x725x224_nhwc_224x3x3x56_fhwc_nhwf_2x2s_1x1p_1x1d_4g$async_dispatch_0_conv_6x235x363x4x56x3x3x56_bf16xbf16xf32 -> @module8629613322876569104_apply_op_config,
        @match_conv_2d_bfloat16_forward_5x235x363x224_nhwc_224x3x3x56_fhwc_nhwf_1x1s_1x1p_1x1d_4g$async_dispatch_0_conv_5x235x363x4x56x3x3x56_bf16xbf16xf32 -> @module5183625046126744491_apply_op_config,
        @match_conv_2d_bfloat16_forward_7x470x725x224_nhwc_224x3x3x56_fhwc_nhwf_2x2s_1x1p_1x1d_4g$async_dispatch_0_conv_7x235x363x4x56x3x3x56_bf16xbf16xf32 -> @module4859907193630060386_apply_op_config,
        @match_conv_2d_bfloat16_forward_10x118x182x448_nhwc_448x3x3x56_fhwc_nhwf_1x1s_1x1p_1x1d_8g$async_dispatch_0_conv_10x118x182x8x56x3x3x56_bf16xbf16xf32 -> @module7886996018887940867_apply_op_config,
        @match_conv_2d_bfloat16_forward_6x235x363x224_nhwc_224x3x3x56_fhwc_nhwf_1x1s_1x1p_1x1d_4g$async_dispatch_0_conv_6x235x363x4x56x3x3x56_bf16xbf16xf32 -> @module1542749415825119683_apply_op_config,
        @match_conv_2d_bfloat16_forward_10x470x725x224_nhwc_224x3x3x56_fhwc_nhwf_2x2s_1x1p_1x1d_4g$async_dispatch_0_conv_10x235x363x4x56x3x3x56_bf16xbf16xf32 -> @module2279074681818677748_apply_op_config,
        @match_conv_2d_bfloat16_forward_12x470x725x224_nhwc_224x3x3x56_fhwc_nhwf_2x2s_1x1p_1x1d_4g$async_dispatch_0_conv_12x235x363x4x56x3x3x56_bf16xbf16xf32 -> @module3648899047666359245_apply_op_config,
        @match_conv_2d_bfloat16_forward_7x235x363x224_nhwc_224x3x3x56_fhwc_nhwf_1x1s_1x1p_1x1d_4g$async_dispatch_0_conv_7x235x363x4x56x3x3x56_bf16xbf16xf32 -> @"module-5380161393819990948_apply_op_config",
        @match_conv_2d_bfloat16_forward_12x118x182x448_nhwc_448x3x3x56_fhwc_nhwf_1x1s_1x1p_1x1d_8g$async_dispatch_0_conv_12x118x182x8x56x3x3x56_bf16xbf16xf32 -> @"module-3129989938171215199_apply_op_config",
        @match_conv_2d_bfloat16_forward_10x235x363x224_nhwc_224x3x3x56_fhwc_nhwf_1x1s_1x1p_1x1d_4g$async_dispatch_0_conv_10x235x363x4x56x3x3x56_bf16xbf16xf32 -> @module5805262495997135865_apply_op_config,
        @match_conv_2d_bfloat16_forward_12x235x363x224_nhwc_224x3x3x56_fhwc_nhwf_1x1s_1x1p_1x1d_4g$async_dispatch_0_conv_12x235x363x4x56x3x3x56_bf16xbf16xf32 -> @"module-1992552252842254751_apply_op_config",
        @match_conv_2d_bfloat16_forward_7x59x91x896_nhwc_896x1x1x896_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_37583x896x896_bf16xbf16xf32 -> @module950986832807159606_apply_op_config,
        @match_conv_2d_bfloat16_forward_10x59x91x896_nhwc_896x3x3x56_fhwc_nhwf_1x1s_1x1p_1x1d_16g$async_dispatch_0_conv_10x59x91x16x56x3x3x56_bf16xbf16xf32 -> @"module-5482690283594198306_apply_op_config",
        @match_conv_2d_bfloat16_forward_12x59x91x896_nhwc_896x3x3x56_fhwc_nhwf_1x1s_1x1p_1x1d_16g$async_dispatch_0_conv_12x59x91x16x56x3x3x56_bf16xbf16xf32 -> @module4494627814173524847_apply_op_config,
        @match_conv_2d_bfloat16_forward_10x59x91x896_nhwc_896x1x1x896_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_53690x896x896_bf16xbf16xf32 -> @"module-7113439279041897089_apply_op_config",
        @match_conv_2d_bfloat16_forward_12x59x91x896_nhwc_896x1x1x896_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_64428x896x896_bf16xbf16xf32 -> @module6158197132717631226_apply_op_config,
        @match_conv_2d_bfloat16_forward_16x225x225x13_nhwc_64x3x3x13_fhwc_nhwf_1x1s_1x1p_1x1d_1g$async_dispatch_0_conv_16x225x225x64x3x3x13_bf16xbf16xf32 -> @module8396357738012441017_apply_op_config,
        @match_conv_2d_bfloat16_forward_16x450x450x2_nhwc_2x1x1x2_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_3240000x2x2_bf16xbf16xf32 -> @module2677224730547331610_apply_op_config,
        @match_conv_2d_bfloat16_forward_16x24x16x288_nhwc_288x3x1x96_fhwc_nhwf_1x1s_1x0p_1x1d_3g$async_dispatch_0_conv_16x24x16x3x96x3x96_bf16xbf16xf32 -> @"module-5610834851670606643_apply_op_config",
        @match_conv_2d_bfloat16_forward_128x24x48x384_nhwc_512x1x1x384_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_147456x512x384_bf16xbf16xf32 -> @"module-8838592581327294770_apply_op_config",
        @match_conv_2d_bfloat16_forward_16x96x64x96_nhwc_96x1x1x96_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_98304x96x96_bf16xbf16xf32 -> @"module-4092354048484454229_apply_op_config",
        @match_conv_2d_bfloat16_forward_16x48x32x48_nhwc_48x5x5x48_fhwc_nhwf_1x1s_8x8p_4x4d_1g$async_dispatch_0_conv_16x48x32x48x5x5x48_bf16xbf16xf32 -> @module2689198279833294263_apply_op_config,
        @match_conv_2d_bfloat16_forward_16x48x32x576_nhwc_576x1x1x576_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_24576x576x576_bf16xbf16xf32 -> @module3994070714556536465_apply_op_config,
        @match_conv_2d_bfloat16_forward_16x48x32x480_nhwc_480x1x1x96_fhwc_nhwf_1x1s_0x0p_1x1d_5g$async_dispatch_0_matmul_like_16x48x32x5x96x96_bf16xbf16xf32 -> @module5276668663876473507_apply_op_config,
        @match_conv_2d_bfloat16_forward_128x24x48x480_nhwc_384x1x1x480_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_147456x384x480_bf16xbf16xf32 -> @"module-4767593926805822391_apply_op_config",
        @match_conv_3d_bfloat16_forward_b_16x2x48x32x288_ndhwc_288x3x1x1x96_fdhwc_ndhwf_1x1x1s_1x0x0p_1x1x1d_3g$async_dispatch_0_conv_16x2x48x32x3x96x3x96_bf16xbf16xf32 -> @"module-1608284571499770249_apply_op_config",
        @match_conv_2d_bfloat16_forward_16x48x32x672_nhwc_576x1x1x672_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_24576x576x672_bf16xbf16xf32 -> @"module-464855148765289941_apply_op_config",
        @match_conv_2d_bfloat16_forward_16x96x64x96_nhwc_96x3x1x96_fhwc_nhwf_1x1s_1x0p_1x1d_1g$async_dispatch_0_conv_16x96x64x96x3x96_bf16xbf16xf32 -> @module4378881155291204668_apply_op_config,
        @match_conv_2d_bfloat16_forward_16x96x64x48_nhwc_48x3x3x48_fhwc_nhwf_1x1s_1x1p_1x1d_1g$async_dispatch_0_conv_16x96x64x48x3x3x48_bf16xbf16xf32 -> @"module-763959733402495440_apply_op_config",
        @match_conv_2d_bfloat16_forward_16x96x64x192_nhwc_96x1x1x192_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_98304x96x192_bf16xbf16xf32 -> @module2066699225626929295_apply_op_config,
        @match_conv_2d_bfloat16_forward_16x96x64x48_nhwc_48x3x3x48_fhwc_nhwf_1x1s_4x4p_4x4d_1g$async_dispatch_0_conv_16x96x64x48x3x3x48_bf16xbf16xf32 -> @module7350505846474122665_apply_op_config,
        @match_conv_3d_bfloat16_forward_b_16x4x48x32x288_ndhwc_288x3x1x1x96_fdhwc_ndhwf_1x1x1s_1x0x0p_1x1x1d_3g$async_dispatch_0_conv_16x4x48x32x3x96x3x96_bf16xbf16xf32 -> @"module-7530347551705388041_apply_op_config",
        @match_conv_3d_bfloat16_forward_b_16x2x48x32x288_ndhwc_288x1x3x3x96_fdhwc_ndhwf_1x1x1s_0x1x1p_1x1x1d_3g$async_dispatch_0_conv_16x2x48x32x3x96x3x3x96_bf16xbf16xf32 -> @"module-1970581355946039598_apply_op_config",
        @match_conv_2d_bfloat16_forward_16x96x64x48_nhwc_48x3x3x48_fhwc_nhwf_1x1s_2x2p_2x2d_1g$async_dispatch_0_conv_16x96x64x48x3x3x48_bf16xbf16xf32 -> @module3276315160131382713_apply_op_config,
        @match_conv_3d_bfloat16_forward_16x2x48x32x288_ndhwc_288x2x1x1x288_fdhwc_ndhwf_2x1x1s_0x0x0p_1x1x1d_1g$async_dispatch_0_matmul_like_16x48x32x288x2x288_bf16xbf16xf32 -> @"module-3100570618954513192_apply_op_config",
        @match_conv_2d_bfloat16_forward_16x96x64x96_nhwc_96x3x3x96_fhwc_nhwf_1x1s_1x1p_1x1d_1g$async_dispatch_0_conv_16x96x64x96x3x3x96_bf16xbf16xf32 -> @"module-5016717600198604863_apply_op_config",
        @match_conv_2d_bfloat16_forward_128x24x48x384_nhwc_384x3x1x128_fhwc_nhwf_1x1s_1x0p_1x1d_3g$async_dispatch_0_conv_128x24x48x3x128x3x128_bf16xbf16xf32 -> @module4561946264403035130_apply_op_config,
        @match_conv_3d_bfloat16_forward_b_16x8x48x32x288_ndhwc_288x3x1x1x96_fdhwc_ndhwf_1x1x1s_1x0x0p_1x1x1d_3g$async_dispatch_0_conv_16x8x48x32x3x96x3x96_bf16xbf16xf32 -> @"module-6425628932703912647_apply_op_config",
        @match_conv_3d_bfloat16_forward_16x2x48x32x288_ndhwc_288x3x1x1x96_fdhwc_ndhwf_1x1x1s_1x0x0p_1x1x1d_3g$async_dispatch_0_conv_16x2x48x32x3x96x3x96_bf16xbf16xf32 -> @module6909153255479619441_apply_op_config,
        @match_conv_3d_bfloat16_forward_b_16x4x48x32x288_ndhwc_288x1x3x3x96_fdhwc_ndhwf_1x1x1s_0x1x1p_1x1x1d_3g$async_dispatch_0_conv_16x4x48x32x3x96x3x3x96_bf16xbf16xf32 -> @module397248974270796247_apply_op_config,
        @match_conv_2d_bfloat16_forward_128x24x48x384_nhwc_384x1x1x384_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_147456x384x384_bf16xbf16xf32 -> @module7216368369122952845_apply_op_config,
        @match_conv_2d_bfloat16_forward_16x96x64x48_nhwc_48x5x5x48_fhwc_nhwf_1x1s_8x8p_4x4d_1g$async_dispatch_0_conv_16x96x64x48x5x5x48_bf16xbf16xf32 -> @"module-4520889304652789583_apply_op_config",
        @match_conv_3d_bfloat16_forward_16x4x48x32x288_ndhwc_288x2x1x1x288_fdhwc_ndhwf_2x1x1s_0x0x0p_1x1x1d_1g$async_dispatch_0_conv_16x2x48x32x288x2x288_bf16xbf16xf32 -> @module5408025256171335551_apply_op_config,
        @match_conv_3d_bfloat16_forward_16x4x48x32x288_ndhwc_288x3x1x1x96_fdhwc_ndhwf_1x1x1s_1x0x0p_1x1x1d_3g$async_dispatch_0_conv_16x4x48x32x3x96x3x96_bf16xbf16xf32 -> @"module-2215561608523455322_apply_op_config",
        @match_conv_3d_bfloat16_forward_b_16x8x48x32x288_ndhwc_288x1x3x3x96_fdhwc_ndhwf_1x1x1s_0x1x1p_1x1x1d_3g$async_dispatch_0_conv_16x8x48x32x3x96x3x3x96_bf16xbf16xf32 -> @"module-5385892577352669266_apply_op_config",
        @match_conv_2d_bfloat16_forward_16x48x32x576_nhwc_576x3x3x576_fhwc_nhwf_1x1s_1x1p_1x1d_1g$async_dispatch_0_conv_16x48x32x576x3x3x576_bf16xbf16xf32 -> @module3515167887415924444_apply_op_config,
        @match_conv_2d_bfloat16_forward_128x48x32x384_nhwc_96x1x1x384_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_196608x96x384_bf16xbf16xf32 -> @module1669037280268302384_apply_op_config,
        @match_conv_3d_bfloat16_forward_16x2x48x32x288_ndhwc_288x1x3x3x96_fdhwc_ndhwf_1x1x1s_0x1x1p_1x1x1d_3g$async_dispatch_0_conv_16x2x48x32x3x96x3x3x96_bf16xbf16xf32 -> @module1291937590125293741_apply_op_config,
        @match_conv_2d_bfloat16_forward_16x48x32x288_nhwc_288x3x3x288_fhwc_nhwf_1x1s_1x1p_1x1d_1g$async_dispatch_0_conv_16x48x32x288x3x3x288_bf16xbf16xf32 -> @"module-6865705440217477667_apply_op_config",
        @match_conv_2d_bfloat16_forward_16x48x32x768_nhwc_2048x3x3x768_fhwc_nhwf_1x1s_1x1p_1x1d_1g$async_dispatch_0_conv_16x48x32x2048x3x3x768_bf16xbf16xf32 -> @module8204484670706657071_apply_op_config,
        @match_conv_3d_bfloat16_forward_16x8x48x32x288_ndhwc_288x2x1x1x288_fdhwc_ndhwf_2x1x1s_0x0x0p_1x1x1d_1g$async_dispatch_0_conv_16x4x48x32x288x2x288_bf16xbf16xf32 -> @module3663013193463791761_apply_op_config,
        @match_conv_3d_bfloat16_forward_16x8x48x32x288_ndhwc_288x3x1x1x96_fdhwc_ndhwf_1x1x1s_1x0x0p_1x1x1d_3g$async_dispatch_0_conv_16x8x48x32x3x96x3x96_bf16xbf16xf32 -> @"module-2183437910029537272_apply_op_config",
        @match_conv_3d_bfloat16_forward_16x4x48x32x288_ndhwc_288x1x3x3x96_fdhwc_ndhwf_1x1x1s_0x1x1p_1x1x1d_3g$async_dispatch_0_conv_16x4x48x32x3x96x3x3x96_bf16xbf16xf32 -> @"module-4567873843891099811_apply_op_config",
        @match_conv_2d_bfloat16_forward_16x48x32x2048_nhwc_2048x3x1x2048_fhwc_nhwf_3x1s_1x0p_1x1d_1g$async_dispatch_0_conv_16x16x32x2048x3x2048_bf16xbf16xf32 -> @"module-2973162512074653305_apply_op_config",
        @match_conv_3d_bfloat16_forward_16x8x48x32x288_ndhwc_288x1x3x3x96_fdhwc_ndhwf_1x1x1s_0x1x1p_1x1x1d_3g$async_dispatch_0_conv_16x8x48x32x3x96x3x3x96_bf16xbf16xf32 -> @module6559857971860772791_apply_op_config,
        @match_conv_2d_bfloat16_forward_16x192x128x40_nhwc_40x3x3x40_fhwc_nhwf_2x2s_1x1p_1x1d_1g$async_dispatch_0_conv_16x96x64x40x3x3x40_bf16xbf16xf32 -> @module2310130817867977318_apply_op_config,
        @module3663013193463791761_match_conv_2d_bfloat16_forward_16x48x32x768_nhwc_2048x3x3x768_fhwc_nhwf_1x1s_1x1p_1x1d_1g$async_dispatch_0_conv_16x48x32x2048x3x3x768_bf16xbf16xf32 -> @module2054123257679205068_apply_op_config,
        @"module-6425628932703912647_match_conv_2d_bfloat16_forward_128x24x48x384_nhwc_384x3x1x128_fhwc_nhwf_1x1s_1x0p_1x1d_3g$async_dispatch_0_conv_128x24x48x3x128x3x128_bf16xbf16xf32" -> @module7426350141531559482_apply_op_config,
        @module1669037280268302384_match_conv_2d_bfloat16_forward_16x48x32x576_nhwc_576x3x3x576_fhwc_nhwf_1x1s_1x1p_1x1d_1g$async_dispatch_0_conv_16x48x32x576x3x3x576_bf16xbf16xf32 -> @module7834922049502397065_apply_op_config,
        @match_conv_2d_bfloat16_forward_16x48x32x96_nhwc_96x1x3x96_fhwc_nhwf_1x1s_0x2p_2x2d_1g$async_dispatch_0_conv_16x48x32x96x3x96_bf16xbf16xf32 -> @"module-5346593499180439123_apply_op_config",
        @match_conv_2d_bfloat16_forward_16x48x32x96_nhwc_96x1x3x96_fhwc_nhwf_1x1s_0x1p_1x1d_1g$async_dispatch_0_conv_16x48x32x96x3x96_bf16xbf16xf32 -> @module5802777366728024462_apply_op_config,
        @match_conv_2d_bfloat16_forward_16x48x32x96_nhwc_48x1x1x96_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_24576x48x96_bf16xbf16xf32 -> @module6979160480657067220_apply_op_config,
        @match_conv_2d_bfloat16_forward_16x48x32x96_nhwc_288x1x1x96_fhwc_nhwf_2x2s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_16x24x16x288x96_bf16xbf16xf32 -> @"module-1545887152392129862_apply_op_config",
        @match_conv_2d_bfloat16_forward_16x48x32x96_nhwc_288x1x1x96_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_24576x288x96_bf16xbf16xf32 -> @module1140498713111542990_apply_op_config,
        @match_conv_2d_bfloat16_forward_16x24x16x96_nhwc_96x3x1x96_fhwc_nhwf_1x1s_4x0p_4x4d_1g$async_dispatch_0_conv_16x24x16x96x3x96_bf16xbf16xf32 -> @"module-8950407434555634540_apply_op_config",
        @match_conv_2d_bfloat16_forward_16x24x16x96_nhwc_96x3x1x96_fhwc_nhwf_1x1s_2x0p_2x2d_1g$async_dispatch_0_conv_16x24x16x96x3x96_bf16xbf16xf32 -> @module4308463292893212318_apply_op_config,
        @match_conv_2d_bfloat16_forward_16x24x16x96_nhwc_96x1x3x96_fhwc_nhwf_1x1s_0x2p_2x2d_1g$async_dispatch_0_conv_16x24x16x96x3x96_bf16xbf16xf32 -> @module6755737478891846325_apply_op_config,
        @match_conv_2d_bfloat16_forward_16x24x16x96_nhwc_96x1x3x96_fhwc_nhwf_1x1s_0x1p_1x1d_1g$async_dispatch_0_conv_16x24x16x96x3x96_bf16xbf16xf32 -> @module7294564384001904877_apply_op_config,
        @module7426350141531559482_match_conv_2d_bfloat16_forward_16x48x32x768_nhwc_2048x3x3x768_fhwc_nhwf_1x1s_1x1p_1x1d_1g$async_dispatch_0_conv_16x48x32x2048x3x3x768_bf16xbf16xf32 -> @module6549106516543380535_apply_op_config,
        @module4378881155291204668_match_conv_2d_bfloat16_forward_16x48x32x672_nhwc_576x1x1x672_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_24576x576x672_bf16xbf16xf32 -> @module4826834601099627801_apply_op_config,
        @match_conv_2d_bfloat16_forward_16x38x38x64_nhwc_64x3x3x64_fhwc_nhwf_1x1s_1x1p_1x1d_1g$async_dispatch_0_conv_16x38x38x64x3x3x64_bf16xbf16xf32 -> @"module-8700669116357993800_apply_op_config",
        @match_conv_2d_bfloat16_forward_16x38x38x64_nhwc_64x3x1x64_fhwc_nhwf_1x1s_1x0p_1x1d_1g$async_dispatch_0_conv_16x38x38x64x3x64_bf16xbf16xf32 -> @module8379457896070944627_apply_op_config,
        @match_conv_2d_bfloat16_forward_16x38x38x64_nhwc_64x1x3x64_fhwc_nhwf_1x2s_0x1p_1x1d_1g$async_dispatch_0_conv_16x38x19x64x3x64_bf16xbf16xf32 -> @"module-5039491333864607957_apply_op_config",
        @match_conv_2d_bfloat16_forward_16x1x32x64_nhwc_64x1x1x64_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_16x32x64x64_bf16xbf16xf32 -> @module1970200436349607352_apply_op_config,
        @"module-5346593499180439123_match_conv_2d_bfloat16_forward_16x48x32x576_nhwc_576x3x3x576_fhwc_nhwf_1x1s_1x1p_1x1d_1g$async_dispatch_0_conv_16x48x32x576x3x3x576_bf16xbf16xf32" -> @module7070744609584329381_apply_op_config,
        @module8615966579730679787_match_conv_2d_bfloat16_forward_16x48x32x576_nhwc_576x1x1x576_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_24576x576x576_bf16xbf16xf32 -> @"module-3610251752505085311_apply_op_config",
        @match_conv_2d_bfloat16_forward_16x48x32x576_nhwc_512x1x1x576_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_24576x512x576_bf16xbf16xf32 -> @"module-5863701520949056473_apply_op_config",
        @match_conv_2d_bfloat16_forward_16x24x16x576_nhwc_288x1x1x576_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_6144x288x576_bf16xbf16xf32 -> @"module-3870568879285452189_apply_op_config",
        @match_conv_2d_bfloat16_forward_16x1x30x576_nhwc_8192x1x1x576_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_16x30x8192x576_bf16xbf16xf32 -> @"module-5617105841929747007_apply_op_config",
        @match_conv_2d_bfloat16_forward_16x1x30x576_nhwc_3x1x1x576_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_16x30x3x576_bf16xbf16xf32 -> @module1623047442867551033_apply_op_config,
        @match_conv_2d_bfloat16_forward_16x1x30x576_nhwc_1024x1x1x576_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_16x30x1024x576_bf16xbf16xf32 -> @module3504644795186292955_apply_op_config,
        @match_conv_2d_bfloat16_forward_16x1x2x576_nhwc_192x1x1x576_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_16x2x192x576_bf16xbf16xf32 -> @module8970479007554311969_apply_op_config,
        @match_conv_2d_bfloat16_forward_16x1x2x576_nhwc_1024x1x1x576_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_16x2x1024x576_bf16xbf16xf32 -> @"module-5443390639632347215_apply_op_config",
        @match_conv_2d_bfloat16_forward_16x1x1x576_nhwc_576x1x1x576_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_16x576x576_bf16xbf16xf32 -> @module2614628224725838856_apply_op_config,
        @"module-4767593926805822391_match_conv_2d_bfloat16_forward_16x48x32x480_nhwc_480x1x1x96_fhwc_nhwf_1x1s_0x0p_1x1d_5g$async_dispatch_0_matmul_like_16x48x32x5x96x96_bf16xbf16xf32" -> @module891295249512997994_apply_op_config,
        @match_conv_2d_bfloat16_forward_16x48x32x480_nhwc_128x1x1x480_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_24576x128x480_bf16xbf16xf32 -> @"module-8731856866266521511_apply_op_config",
        @module5408025256171335551_match_conv_2d_bfloat16_forward_16x96x64x48_nhwc_48x5x5x48_fhwc_nhwf_1x1s_8x8p_4x4d_1g$async_dispatch_0_conv_16x96x64x48x5x5x48_bf16xbf16xf32 -> @module2642271885679815804_apply_op_config,
        @"module-7530347551705388041_match_conv_2d_bfloat16_forward_16x96x64x48_nhwc_48x3x3x48_fhwc_nhwf_1x1s_4x4p_4x4d_1g$async_dispatch_0_conv_16x96x64x48x3x3x48_bf16xbf16xf32" -> @module4635708189346161356_apply_op_config,
        @"module-3100570618954513192_match_conv_2d_bfloat16_forward_16x96x64x48_nhwc_48x3x3x48_fhwc_nhwf_1x1s_2x2p_2x2d_1g$async_dispatch_0_conv_16x96x64x48x3x3x48_bf16xbf16xf32" -> @module5175609052454433875_apply_op_config,
        @module2066699225626929295_match_conv_2d_bfloat16_forward_16x96x64x48_nhwc_48x3x3x48_fhwc_nhwf_1x1s_1x1p_1x1d_1g$async_dispatch_0_conv_16x96x64x48x3x3x48_bf16xbf16xf32 -> @"module-7367110414070804444_apply_op_config",
        @"module-5011352483966054532_match_conv_2d_bfloat16_forward_16x48x32x48_nhwc_48x5x5x48_fhwc_nhwf_1x1s_8x8p_4x4d_1g$async_dispatch_0_conv_16x48x32x48x5x5x48_bf16xbf16xf32" -> @"module-690138260419776630_apply_op_config",
        @match_conv_2d_bfloat16_forward_16x48x32x48_nhwc_48x3x3x48_fhwc_nhwf_1x1s_2x2p_2x2d_1g$async_dispatch_0_conv_16x48x32x48x3x3x48_bf16xbf16xf32 -> @"module-7142624883020280426_apply_op_config",
        @match_conv_2d_bfloat16_forward_16x24x16x48_nhwc_192x1x1x48_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_6144x192x48_bf16xbf16xf32 -> @"module-5878059958949092251_apply_op_config",
        @match_conv_2d_bfloat16_forward_16x96x64x40_nhwc_40x1x1x40_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_98304x40x40_bf16xbf16xf32 -> @"module-3236419692971477372_apply_op_config",
        @module2054123257679205068_match_conv_2d_bfloat16_forward_16x192x128x40_nhwc_40x3x3x40_fhwc_nhwf_2x2s_1x1p_1x1d_1g$async_dispatch_0_conv_16x96x64x40x3x3x40_bf16xbf16xf32 -> @"module-9078255276051304928_apply_op_config",
        @match_conv_2d_bfloat16_forward_16x48x32x384_nhwc_384x1x1x384_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_24576x384x384_bf16xbf16xf32 -> @"module-4706140772942856255_apply_op_config",
        @match_conv_2d_bfloat16_forward_16x12x8x384_nhwc_192x1x1x384_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_1536x192x384_bf16xbf16xf32 -> @module3494839312045714732_apply_op_config,
        @match_conv_2d_bfloat16_forward_16x8x32x288_nhwc_384x1x1x288_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_4096x384x288_bf16xbf16xf32 -> @module813434365613521345_apply_op_config,
        @match_conv_2d_bfloat16_forward_16x48x32x288_nhwc_96x1x1x288_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_24576x96x288_bf16xbf16xf32 -> @module2933426064943781440_apply_op_config,
        @match_conv_2d_bfloat16_forward_16x48x32x288_nhwc_384x1x1x288_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_24576x384x288_bf16xbf16xf32 -> @module6152709004796445680_apply_op_config,
        @match_conv_2d_bfloat16_forward_16x48x32x288_nhwc_288x3x3x96_fhwc_nhwf_2x2s_1x1p_1x1d_3g$async_dispatch_0_conv_16x24x16x3x96x3x3x96_bf16xbf16xf32 -> @module8250078879978817073_apply_op_config,
        @module8204484670706657071_match_conv_2d_bfloat16_forward_16x48x32x288_nhwc_288x3x3x288_fhwc_nhwf_1x1s_1x1p_1x1d_1g$async_dispatch_0_conv_16x48x32x288x3x3x288_bf16xbf16xf32 -> @module4044126642927762173_apply_op_config,
        @match_conv_2d_bfloat16_forward_16x24x16x288_nhwc_96x1x1x288_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_6144x96x288_bf16xbf16xf32 -> @"module-1439035349042860153_apply_op_config",
        @match_conv_2d_bfloat16_forward_16x24x16x288_nhwc_288x3x3x288_fhwc_nhwf_1x1s_1x1p_1x1d_1g$async_dispatch_0_conv_16x24x16x288x3x3x288_bf16xbf16xf32 -> @"module-4383060726791129170_apply_op_config",
        @"module-8838592581327294770_match_conv_2d_bfloat16_forward_16x24x16x288_nhwc_288x3x1x96_fhwc_nhwf_1x1s_1x0p_1x1d_3g$async_dispatch_0_conv_16x24x16x3x96x3x96_bf16xbf16xf32" -> @"module-6247610171180181805_apply_op_config",
        @match_conv_2d_bfloat16_forward_16x24x16x288_nhwc_288x1x3x96_fhwc_nhwf_1x1s_0x1p_1x1d_3g$async_dispatch_0_conv_16x24x16x3x96x3x96_bf16xbf16xf32 -> @module6677447418464053593_apply_op_config,
        @match_conv_2d_bfloat16_forward_16x24x16x288_nhwc_288x1x1x96_fhwc_nhwf_1x1s_0x0p_1x1d_3g$async_dispatch_0_matmul_like_16x24x16x3x96x96_bf16xbf16xf32 -> @"module-5087663706872769327_apply_op_config",
        @match_conv_2d_bfloat16_forward_16x24x16x288_nhwc_288x1x1x288_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_6144x288x288_bf16xbf16xf32 -> @"module-6980181089137153937_apply_op_config",
        @match_conv_2d_bfloat16_forward_16x24x16x288_nhwc_144x1x1x288_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_6144x144x288_bf16xbf16xf32 -> @module5596125379591511372_apply_op_config,
        @"module-4567873843891099811_match_conv_3d_bfloat16_forward_16x8x48x32x288_ndhwc_288x3x1x1x96_fdhwc_ndhwf_1x1x1s_1x0x0p_1x1x1d_3g$async_dispatch_0_conv_16x8x48x32x3x96x3x96_bf16xbf16xf32" -> @module4283728422656460134_apply_op_config,
        @"module-2183437910029537272_match_conv_3d_bfloat16_forward_16x8x48x32x288_ndhwc_288x2x1x1x288_fdhwc_ndhwf_2x1x1s_0x0x0p_1x1x1d_1g$async_dispatch_0_conv_16x4x48x32x288x2x288_bf16xbf16xf32" -> @module9097057252924939381_apply_op_config,
        @module2310130817867977318_match_conv_3d_bfloat16_forward_16x8x48x32x288_ndhwc_288x1x3x3x96_fdhwc_ndhwf_1x1x1s_0x1x1p_1x1x1d_3g$async_dispatch_0_conv_16x8x48x32x3x96x3x3x96_bf16xbf16xf32 -> @module205055630836799240_apply_op_config,
        @"module-5385892577352669266_match_conv_3d_bfloat16_forward_16x4x48x32x288_ndhwc_288x3x1x1x96_fdhwc_ndhwf_1x1x1s_1x0x0p_1x1x1d_3g$async_dispatch_0_conv_16x4x48x32x3x96x3x96_bf16xbf16xf32" -> @"module-5319777781164961521_apply_op_config",
        @"module-2215561608523455322_match_conv_3d_bfloat16_forward_16x4x48x32x288_ndhwc_288x2x1x1x288_fdhwc_ndhwf_2x1x1s_0x0x0p_1x1x1d_1g$async_dispatch_0_conv_16x2x48x32x288x2x288_bf16xbf16xf32" -> @module1835695904987310839_apply_op_config,
        @"module-2973162512074653305_match_conv_3d_bfloat16_forward_16x4x48x32x288_ndhwc_288x1x3x3x96_fdhwc_ndhwf_1x1x1s_0x1x1p_1x1x1d_3g$async_dispatch_0_conv_16x4x48x32x3x96x3x3x96_bf16xbf16xf32" -> @"module-2958846526441287144_apply_op_config",
        @"module-1589592913381196107_match_conv_3d_bfloat16_forward_16x2x48x32x288_ndhwc_288x3x1x1x96_fdhwc_ndhwf_1x1x1s_1x0x0p_1x1x1d_3g$async_dispatch_0_conv_16x2x48x32x3x96x3x96_bf16xbf16xf32" -> @"module-252229267495831757_apply_op_config",
        @module3244247317313473054_match_conv_3d_bfloat16_forward_16x2x48x32x288_ndhwc_288x2x1x1x288_fdhwc_ndhwf_2x1x1s_0x0x0p_1x1x1d_1g$async_dispatch_0_matmul_like_16x48x32x288x2x288_bf16xbf16xf32 -> @module8217597828895093057_apply_op_config,
        @"module-6865705440217477667_match_conv_3d_bfloat16_forward_16x2x48x32x288_ndhwc_288x1x3x3x96_fdhwc_ndhwf_1x1x1s_0x1x1p_1x1x1d_3g$async_dispatch_0_conv_16x2x48x32x3x96x3x3x96_bf16xbf16xf32" -> @module1362604622253719980_apply_op_config,
        @match_conv_2d_bfloat16_forward_16x8x32x2048_nhwc_576x1x1x2048_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_4096x576x2048_bf16xbf16xf32 -> @"module-7088503340171957963_apply_op_config",
        @module3994070714556536465_match_conv_2d_bfloat16_forward_16x8x32x2048_nhwc_288x1x1x2048_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_4096x288x2048_bf16xbf16xf32 -> @module2280249657558078362_apply_op_config,
        @module6559857971860772791_match_conv_2d_bfloat16_forward_16x48x32x2048_nhwc_2048x3x1x2048_fhwc_nhwf_3x1s_1x0p_1x1d_1g$async_dispatch_0_conv_16x16x32x2048x3x2048_bf16xbf16xf32 -> @"module-4712683707803685992_apply_op_config",
        @match_conv_2d_bfloat16_forward_16x1x512x2048_nhwc_2048x1x2x2048_fhwc_nhwf_1x2s_0x0p_1x1d_1g$async_dispatch_0_conv_16x256x2048x2x2048_bf16xbf16xf32 -> @module4870784003488566835_apply_op_config,
        @module7350505846474122665_match_conv_2d_bfloat16_forward_16x96x64x192_nhwc_96x1x1x192_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_98304x96x192_bf16xbf16xf32 -> @module7219792799496758047_apply_op_config,
        @match_conv_2d_bfloat16_forward_16x48x32x192_nhwc_96x1x1x192_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_24576x96x192_bf16xbf16xf32 -> @module8623344520257799845_apply_op_config,
        @match_conv_2d_bfloat16_forward_16x23x1x192_nhwc_192x3x1x192_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_conv_16x21x192x3x192_bf16xbf16xf32 -> @module2923642059297261885_apply_op_config,
        @match_conv_2d_bfloat16_forward_16x12x8x192_nhwc_384x1x1x192_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_1536x384x192_bf16xbf16xf32 -> @module354875760083644398_apply_op_config,
        @match_conv_2d_bfloat16_forward_16x1x21x192_nhwc_384x1x1x192_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_16x21x384x192_bf16xbf16xf32 -> @module4419538543564309907_apply_op_config,
        @match_conv_2d_bfloat16_forward_16x48x32x128_nhwc_64x1x1x128_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_24576x64x128_bf16xbf16xf32 -> @"module-271961346453902565_apply_op_config",
        @match_conv_2d_bfloat16_forward_16x48x32x128_nhwc_128x2x2x128_fhwc_nhwf_2x2s_0x0p_1x1d_1g$async_dispatch_0_conv_16x24x16x128x2x2x128_bf16xbf16xf32 -> @module8040801911246838659_apply_op_config,
        @match_conv_2d_bfloat16_forward_16x24x16x128_nhwc_192x2x2x128_fhwc_nhwf_2x2s_0x0p_1x1d_1g$async_dispatch_0_conv_16x12x8x192x2x2x128_bf16xbf16xf32 -> @"module-7653633243858028790_apply_op_config",
        @match_conv_2d_bfloat16_forward_16x1x6x1024_nhwc_576x1x1x1024_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_16x6x576x1024_bf16xbf16xf32 -> @"module-7602620474587856351_apply_op_config",
        @"module-7508047808273090963_match_conv_2d_bfloat16_forward_128x24x48x480_nhwc_384x1x1x480_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_147456x384x480_bf16xbf16xf32" -> @"module-4289354440547745606_apply_op_config",
        @match_conv_2d_bfloat16_forward_128x48x32x448_nhwc_384x1x1x448_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_196608x384x448_bf16xbf16xf32 -> @module6151868810080441787_apply_op_config,
        @"module-1507712216101663294_match_conv_2d_bfloat16_forward_128x48x32x384_nhwc_96x1x1x384_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_196608x96x384_bf16xbf16xf32" -> @module5741017693719714659_apply_op_config,
        @match_conv_2d_bfloat16_forward_128x48x32x384_nhwc_384x3x3x64_fhwc_nhwf_1x1s_1x1p_1x1d_6g$async_dispatch_0_conv_128x48x32x6x64x3x3x64_bf16xbf16xf32 -> @module3933653812497562654_apply_op_config,
        @match_conv_2d_bfloat16_forward_128x48x32x384_nhwc_384x1x1x384_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_196608x384x384_bf16xbf16xf32 -> @"module-3096228896624724732_apply_op_config",
        @"module-4092354048484454229_match_conv_2d_bfloat16_forward_128x24x48x384_nhwc_512x1x1x384_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_147456x512x384_bf16xbf16xf32" -> @module5265578897403436880_apply_op_config,
        @module7834922049502397065_match_conv_2d_bfloat16_forward_128x24x48x384_nhwc_384x3x1x128_fhwc_nhwf_1x1s_1x0p_1x1d_3g$async_dispatch_0_conv_128x24x48x3x128x3x128_bf16xbf16xf32 -> @"module-5713853901338332798_apply_op_config",
        @match_conv_2d_bfloat16_forward_128x24x48x384_nhwc_384x1x3x128_fhwc_nhwf_1x1s_0x1p_1x1d_3g$async_dispatch_0_conv_128x24x48x3x128x3x128_bf16xbf16xf32 -> @"module-3307929099139602236_apply_op_config",
        @"module-4520889304652789583_match_conv_2d_bfloat16_forward_128x24x48x384_nhwc_384x1x1x384_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_147456x384x384_bf16xbf16xf32" -> @module896857893101354896_apply_op_config,
        @match_conv_2d_bfloat16_forward_128x24x48x128_nhwc_384x1x1x128_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_147456x384x128_bf16xbf16xf32 -> @"module-522984183083508851_apply_op_config",
        @module5265578897403436880_match_conv_2d_bfloat16_forward_128x48x32x384_nhwc_384x1x1x384_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_196608x384x384_bf16xbf16xf32 -> @module4071637556353897261_apply_op_config,
        @match_conv_2d_bfloat16_forward_b_16x48x32x2048_nhwc_2048x3x1x2048_fhwc_nhwf_3x1s_1x0p_1x1d_1g$async_dispatch_0_conv_16x16x32x2048x3x2048_bf16xbf16xf32 -> @"module-4936958348168552443_apply_op_config",
        @"module-6795424626653063444_match_conv_2d_bfloat16_forward_128x24x48x128_nhwc_384x1x1x128_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_147456x384x128_bf16xbf16xf32" -> @"module-101477417419537001_apply_op_config",
        @"module-3096228896624724732_match_conv_2d_bfloat16_forward_128x48x32x384_nhwc_384x3x3x64_fhwc_nhwf_1x1s_1x1p_1x1d_6g$async_dispatch_0_conv_128x48x32x6x64x3x3x64_bf16xbf16xf32" -> @module1730547502948057187_apply_op_config,
        @"module-3610251752505085311_match_conv_2d_bfloat16_forward_16x48x32x576_nhwc_576x3x3x576_fhwc_nhwf_1x1s_1x1p_1x1d_1g$async_dispatch_0_conv_16x48x32x576x3x3x576_bf16xbf16xf32" -> @"module-6050217721080824708_apply_op_config",
        @module4826834601099627801_match_conv_2d_bfloat16_forward_16x48x32x768_nhwc_2048x3x3x768_fhwc_nhwf_1x1s_1x1p_1x1d_1g$async_dispatch_0_conv_16x48x32x2048x3x3x768_bf16xbf16xf32 -> @module7250346489746609524_apply_op_config,
        @module6151868810080441787_match_conv_2d_bfloat16_forward_128x24x48x480_nhwc_384x1x1x480_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_147456x384x480_bf16xbf16xf32 -> @"module-6880784253496819418_apply_op_config",
        @"module-5713853901338332798_match_conv_2d_bfloat16_forward_128x24x48x384_nhwc_512x1x1x384_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_147456x512x384_bf16xbf16xf32" -> @"module-5484525665775309725_apply_op_config",
        @"module-522984183083508851_match_conv_2d_bfloat16_forward_128x24x48x384_nhwc_384x1x1x384_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_147456x384x384_bf16xbf16xf32" -> @"module-5578073272369513136_apply_op_config",
        @"module-3307929099139602236_match_conv_2d_bfloat16_forward_128x24x48x384_nhwc_384x3x1x128_fhwc_nhwf_1x1s_1x0p_1x1d_3g$async_dispatch_0_conv_128x24x48x3x128x3x128_bf16xbf16xf32" -> @module8850980168311619132_apply_op_config,
        @module896857893101354896_match_conv_2d_bfloat16_forward_128x24x48x384_nhwc_384x1x3x128_fhwc_nhwf_1x1s_0x1p_1x1d_3g$async_dispatch_0_conv_128x24x48x3x128x3x128_bf16xbf16xf32 -> @"module-911083449296662588_apply_op_config",
        @match_conv_2d_bfloat16_forward_16x8x32x576_nhwc_576x1x1x576_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_4096x576x576_bf16xbf16xf32 -> @module4742795371244143238_apply_op_config,
        @module7014507521830355182_match_conv_3d_bfloat16_forward_b_16x2x48x32x288_ndhwc_288x3x1x1x96_fdhwc_ndhwf_1x1x1s_1x0x0p_1x1x1d_3g$async_dispatch_0_conv_16x2x48x32x3x96x3x96_bf16xbf16xf32 -> @module6851109957392407656_apply_op_config,
        @module3276315160131382713_match_conv_3d_bfloat16_forward_b_16x2x48x32x288_ndhwc_288x1x3x3x96_fdhwc_ndhwf_1x1x1s_0x1x1p_1x1x1d_3g$async_dispatch_0_conv_16x2x48x32x3x96x3x3x96_bf16xbf16xf32 -> @"module-8084250599234373333_apply_op_config",
        @"module-1970581355946039598_match_conv_3d_bfloat16_forward_b_16x4x48x32x288_ndhwc_288x3x1x1x96_fdhwc_ndhwf_1x1x1s_1x0x0p_1x1x1d_3g$async_dispatch_0_conv_16x4x48x32x3x96x3x96_bf16xbf16xf32" -> @"module-998895541307183783_apply_op_config",
        @module7216368369122952845_match_conv_3d_bfloat16_forward_b_16x4x48x32x288_ndhwc_288x1x3x3x96_fdhwc_ndhwf_1x1x1s_0x1x1p_1x1x1d_3g$async_dispatch_0_conv_16x4x48x32x3x96x3x3x96_bf16xbf16xf32 -> @module8795648018915444682_apply_op_config,
        @module6909153255479619441_match_conv_3d_bfloat16_forward_b_16x8x48x32x288_ndhwc_288x3x1x1x96_fdhwc_ndhwf_1x1x1s_1x0x0p_1x1x1d_3g$async_dispatch_0_conv_16x8x48x32x3x96x3x96_bf16xbf16xf32 -> @"module-317436726597910684_apply_op_config",
        @module3515167887415924444_match_conv_3d_bfloat16_forward_b_16x8x48x32x288_ndhwc_288x1x3x3x96_fdhwc_ndhwf_1x1x1s_0x1x1p_1x1x1d_3g$async_dispatch_0_conv_16x8x48x32x3x96x3x3x96_bf16xbf16xf32 -> @module8983234031431727099_apply_op_config : (!transform.any_op) -> !transform.any_op
    transform.yield %updated_root : !transform.any_op
  }
}
