#translation = #iree_codegen.translation_info<pipeline = None workgroup_size = [256, 1, 1] subgroup_size = 64, {llvm_func_attrs = {"amdgpu-waves-per-eu" = "2", "denormal-fp-math-f32" = "preserve-sign"}}>
module attributes {transform.with_named_sequence} {
  stream.executable private @base_attention {
    stream.executable.export public @base_attention workgroups() -> (index, index, index) {
      %c8 = arith.constant 8 : index
      %c2 = arith.constant 2 : index
      %c32 = arith.constant 32 : index
      stream.return %c8, %c2, %c32 : index, index, index
    }
    builtin.module {
      func.func @base_attention(%arg0: !stream.binding, %arg1: !stream.binding, %arg2: !stream.binding, %arg3: !stream.binding) attributes {translation_info = #translation} {
        %cst = arith.constant dense<1357> : vector<16xindex>
        %cst_0 = arith.constant dense<56> : vector<4xindex>
        %cst_1 = arith.constant dense<48> : vector<4xindex>
        %cst_2 = arith.constant dense<40> : vector<4xindex>
        %cst_3 = arith.constant dense<32> : vector<4xindex>
        %cst_4 = arith.constant dense<24> : vector<4xindex>
        %cst_5 = arith.constant dense<16> : vector<4xindex>
        %cst_6 = arith.constant dense<8> : vector<4xindex>
        %c204 = arith.constant 204 : index
        %c136 = arith.constant 136 : index
        %c272 = arith.constant 272 : index
        %cst_7 = arith.constant dense<0.000000e+00> : vector<8xf16>
        %cst_8 = arith.constant dense<1.000000e+00> : vector<1xf32>
        %c64_i32 = arith.constant 64 : i32
        %c32_i32 = arith.constant 32 : i32
        %cst_9 = arith.constant dense<0> : vector<16xi32>
        %cst_10 = arith.constant dense<[0, 1, 2, 3]> : vector<4xindex>
        %c1073741822 = arith.constant 1073741822 : index
        %c68 = arith.constant 68 : index
        %c1357 = arith.constant 1357 : index
        %c22 = arith.constant 22 : index
        %c120 = arith.constant 120 : index
        %c112 = arith.constant 112 : index
        %c104 = arith.constant 104 : index
        %c96 = arith.constant 96 : index
        %c88 = arith.constant 88 : index
        %c80 = arith.constant 80 : index
        %c72 = arith.constant 72 : index
        %c56 = arith.constant 56 : index
        %c48 = arith.constant 48 : index
        %c40 = arith.constant 40 : index
        %c24 = arith.constant 24 : index
        %c16 = arith.constant 16 : index
        %c8 = arith.constant 8 : index
        %c4 = arith.constant 4 : index
        %c128 = arith.constant 128 : index
        %c64 = arith.constant 64 : index
        %c1 = arith.constant 1 : index
        %c32 = arith.constant 32 : index
        %c0 = arith.constant 0 : index
        %cst_11 = arith.constant dense<-1.000000e+06> : vector<16xf32>
        %cst_12 = arith.constant dense<-1.000000e+06> : vector<1xf32>
        %cst_13 = arith.constant dense<0.000000e+00> : vector<1xf32>
        %cst_14 = arith.constant dense<0.000000e+00> : vector<16xf32>
        %cst_15 = arith.constant dense<1.275630e-01> : vector<4xf16>
        %workgroup_id_0 = stream.dispatch.workgroup.id[0] : index
        %workgroup_id_1 = stream.dispatch.workgroup.id[1] : index
        %workgroup_id_2 = stream.dispatch.workgroup.id[2] : index
        %thread_id_x = gpu.thread_id  x
        %thread_id_y = gpu.thread_id  y
        %thread_id_z = gpu.thread_id  z
        %alloc = memref.alloc() : memref<1x64x1x68xf16, #gpu.address_space<workgroup>>
        %alloc_16 = memref.alloc() : memref<1x1x64x132xf16, #gpu.address_space<workgroup>>
        %0 = stream.binding.subspan %arg0[%c0] : !stream.binding -> memref<1x1024x32x128xf16, strided<[4194304, 4096, 128, 1], offset: ?>>
        %1 = arith.divsi %thread_id_x, %c64 : index
        %2 = arith.muli %1, %c32 overflow<nsw, nuw> : index
        %3 = arith.muli %workgroup_id_0, %c128 overflow<nsw, nuw> : index
        %4 = arith.remsi %thread_id_x, %c32 : index
        %5 = arith.addi %4, %3 overflow<nsw, nuw> : index
        %6 = arith.addi %5, %2 overflow<nsw, nuw> : index
        %7 = arith.remsi %workgroup_id_2, %c32 : index
        %8 = arith.remsi %thread_id_x, %c64 : index
        %9 = arith.divsi %8, %c32 : index
        %10 = arith.muli %9, %c4 overflow<nsw, nuw> : index
        %11 = vector.load %0[%c0, %6, %7, %10] : memref<1x1024x32x128xf16, strided<[4194304, 4096, 128, 1], offset: ?>>, vector<4xf16>
        %12 = arith.addi %10, %c8 overflow<nsw, nuw> : index
        %13 = vector.load %0[%c0, %6, %7, %12] : memref<1x1024x32x128xf16, strided<[4194304, 4096, 128, 1], offset: ?>>, vector<4xf16>
        %14 = arith.addi %10, %c16 overflow<nsw, nuw> : index
        %15 = vector.load %0[%c0, %6, %7, %14] : memref<1x1024x32x128xf16, strided<[4194304, 4096, 128, 1], offset: ?>>, vector<4xf16>
        %16 = arith.addi %10, %c24 overflow<nsw, nuw> : index
        %17 = vector.load %0[%c0, %6, %7, %16] : memref<1x1024x32x128xf16, strided<[4194304, 4096, 128, 1], offset: ?>>, vector<4xf16>
        %18 = arith.addi %10, %c32 overflow<nsw, nuw> : index
        %19 = vector.load %0[%c0, %6, %7, %18] : memref<1x1024x32x128xf16, strided<[4194304, 4096, 128, 1], offset: ?>>, vector<4xf16>
        %20 = arith.addi %10, %c40 overflow<nsw, nuw> : index
        %21 = vector.load %0[%c0, %6, %7, %20] : memref<1x1024x32x128xf16, strided<[4194304, 4096, 128, 1], offset: ?>>, vector<4xf16>
        %22 = arith.addi %10, %c48 overflow<nsw, nuw> : index
        %23 = vector.load %0[%c0, %6, %7, %22] : memref<1x1024x32x128xf16, strided<[4194304, 4096, 128, 1], offset: ?>>, vector<4xf16>
        %24 = arith.addi %10, %c56 overflow<nsw, nuw> : index
        %25 = vector.load %0[%c0, %6, %7, %24] : memref<1x1024x32x128xf16, strided<[4194304, 4096, 128, 1], offset: ?>>, vector<4xf16>
        %26 = arith.addi %10, %c64 overflow<nsw, nuw> : index
        %27 = vector.load %0[%c0, %6, %7, %26] : memref<1x1024x32x128xf16, strided<[4194304, 4096, 128, 1], offset: ?>>, vector<4xf16>
        %28 = arith.addi %10, %c72 overflow<nsw, nuw> : index
        %29 = vector.load %0[%c0, %6, %7, %28] : memref<1x1024x32x128xf16, strided<[4194304, 4096, 128, 1], offset: ?>>, vector<4xf16>
        %30 = arith.addi %10, %c80 overflow<nsw, nuw> : index
        %31 = vector.load %0[%c0, %6, %7, %30] : memref<1x1024x32x128xf16, strided<[4194304, 4096, 128, 1], offset: ?>>, vector<4xf16>
        %32 = arith.addi %10, %c88 overflow<nsw, nuw> : index
        %33 = vector.load %0[%c0, %6, %7, %32] : memref<1x1024x32x128xf16, strided<[4194304, 4096, 128, 1], offset: ?>>, vector<4xf16>
        %34 = arith.addi %10, %c96 overflow<nsw, nuw> : index
        %35 = vector.load %0[%c0, %6, %7, %34] : memref<1x1024x32x128xf16, strided<[4194304, 4096, 128, 1], offset: ?>>, vector<4xf16>
        %36 = arith.addi %10, %c104 overflow<nsw, nuw> : index
        %37 = vector.load %0[%c0, %6, %7, %36] : memref<1x1024x32x128xf16, strided<[4194304, 4096, 128, 1], offset: ?>>, vector<4xf16>
        %38 = arith.addi %10, %c112 overflow<nsw, nuw> : index
        %39 = vector.load %0[%c0, %6, %7, %38] : memref<1x1024x32x128xf16, strided<[4194304, 4096, 128, 1], offset: ?>>, vector<4xf16>
        %40 = arith.addi %10, %c120 overflow<nsw, nuw> : index
        %41 = vector.load %0[%c0, %6, %7, %40] : memref<1x1024x32x128xf16, strided<[4194304, 4096, 128, 1], offset: ?>>, vector<4xf16>
        %42 = stream.binding.subspan %arg2[%c0] : !stream.binding -> memref<1x1357x32x128xf16, strided<[5558272, 4096, 128, 1], offset: ?>>
        %43 = stream.binding.subspan %arg1[%c0] : !stream.binding -> memref<1x1357x32x128xf16, strided<[5558272, 4096, 128, 1], offset: ?>>
        %44 = arith.mulf %11, %cst_15 : vector<4xf16>
        %45 = arith.mulf %13, %cst_15 : vector<4xf16>
        %46 = arith.mulf %15, %cst_15 : vector<4xf16>
        %47 = arith.mulf %17, %cst_15 : vector<4xf16>
        %48 = arith.mulf %19, %cst_15 : vector<4xf16>
        %49 = arith.mulf %21, %cst_15 : vector<4xf16>
        %50 = arith.mulf %23, %cst_15 : vector<4xf16>
        %51 = arith.mulf %25, %cst_15 : vector<4xf16>
        %52 = arith.mulf %27, %cst_15 : vector<4xf16>
        %53 = arith.mulf %29, %cst_15 : vector<4xf16>
        %54 = arith.mulf %31, %cst_15 : vector<4xf16>
        %55 = arith.mulf %33, %cst_15 : vector<4xf16>
        %56 = arith.mulf %35, %cst_15 : vector<4xf16>
        %57 = arith.mulf %37, %cst_15 : vector<4xf16>
        %58 = arith.mulf %39, %cst_15 : vector<4xf16>
        %59 = arith.mulf %41, %cst_15 : vector<4xf16>
        %60 = arith.muli %thread_id_y, %c16 overflow<nsw, nuw> : index
        %61 = arith.muli %thread_id_z, %c16 overflow<nsw, nuw> : index
        %62 = arith.divsi %thread_id_x, %c16 : index
        %63 = arith.addi %62, %61 overflow<nsw, nuw> : index
        %64 = arith.addi %63, %60 overflow<nsw, nuw> : index
        %65 = arith.remsi %64, %c64 : index
        %66 = arith.remsi %thread_id_x, %c16 : index
        %67 = arith.muli %66, %c8 overflow<nsw, nuw> : index
        %68 = arith.addi %64, %c16 overflow<nsw, nuw> : index
        %69 = arith.remsi %68, %c64 : index
        %70 = arith.addi %64, %c32 overflow<nsw, nuw> : index
        %71 = arith.remsi %70, %c64 : index
        %72 = arith.addi %64, %c48 overflow<nsw, nuw> : index
        %73 = arith.remsi %72, %c64 : index
        %74 = arith.muli %thread_id_y, %c32 overflow<nsw, nuw> : index
        %75 = arith.muli %thread_id_z, %c32 overflow<nsw, nuw> : index
        %76 = arith.divsi %thread_id_x, %c8 : index
        %77 = arith.addi %76, %75 overflow<nsw, nuw> : index
        %78 = arith.addi %77, %74 overflow<nsw, nuw> : index
        %79 = arith.remsi %78, %c64 : index
        %80 = arith.remsi %thread_id_x, %c8 : index
        %81 = arith.muli %80, %c8 overflow<nsw, nuw> : index
        %82 = arith.muli %workgroup_id_1, %c64 overflow<nsw, nuw> : index
        %83 = arith.addi %82, %81 overflow<nsw, nuw> : index
        %84 = arith.addi %78, %c32 overflow<nsw, nuw> : index
        %85 = arith.remsi %84, %c64 : index
        %86 = arith.muli %thread_id_y, %c64 overflow<nsw, nuw> : index
        %87 = arith.addi %4, %86 overflow<nsw, nuw> : index
        %88 = arith.muli %9, %c272 overflow<nsw> : index
        %89 = arith.addi %88, %87 overflow<nsw> : index
        %reinterpret_cast = memref.reinterpret_cast %alloc to offset: [%89], sizes: [%c1073741822], strides: [1] : memref<1x64x1x68xf16, #gpu.address_space<workgroup>> to memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>
        %90 = arith.muli %12, %c68 overflow<nsw> : index
        %91 = arith.addi %90, %87 overflow<nsw> : index
        %reinterpret_cast_17 = memref.reinterpret_cast %alloc to offset: [%91], sizes: [%c1073741822], strides: [1] : memref<1x64x1x68xf16, #gpu.address_space<workgroup>> to memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>
        %92 = arith.muli %14, %c68 overflow<nsw> : index
        %93 = arith.addi %92, %87 overflow<nsw> : index
        %reinterpret_cast_18 = memref.reinterpret_cast %alloc to offset: [%93], sizes: [%c1073741822], strides: [1] : memref<1x64x1x68xf16, #gpu.address_space<workgroup>> to memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>
        %94 = arith.muli %16, %c68 overflow<nsw> : index
        %95 = arith.addi %94, %87 overflow<nsw> : index
        %reinterpret_cast_19 = memref.reinterpret_cast %alloc to offset: [%95], sizes: [%c1073741822], strides: [1] : memref<1x64x1x68xf16, #gpu.address_space<workgroup>> to memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>
        %96 = arith.muli %18, %c68 overflow<nsw> : index
        %97 = arith.addi %96, %87 overflow<nsw> : index
        %reinterpret_cast_20 = memref.reinterpret_cast %alloc to offset: [%97], sizes: [%c1073741822], strides: [1] : memref<1x64x1x68xf16, #gpu.address_space<workgroup>> to memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>
        %98 = arith.muli %20, %c68 overflow<nsw> : index
        %99 = arith.addi %98, %87 overflow<nsw> : index
        %reinterpret_cast_21 = memref.reinterpret_cast %alloc to offset: [%99], sizes: [%c1073741822], strides: [1] : memref<1x64x1x68xf16, #gpu.address_space<workgroup>> to memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>
        %100 = arith.muli %22, %c68 overflow<nsw> : index
        %101 = arith.addi %100, %87 overflow<nsw> : index
        %reinterpret_cast_22 = memref.reinterpret_cast %alloc to offset: [%101], sizes: [%c1073741822], strides: [1] : memref<1x64x1x68xf16, #gpu.address_space<workgroup>> to memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>
        %102 = arith.muli %24, %c68 overflow<nsw> : index
        %103 = arith.addi %102, %87 overflow<nsw> : index
        %reinterpret_cast_23 = memref.reinterpret_cast %alloc to offset: [%103], sizes: [%c1073741822], strides: [1] : memref<1x64x1x68xf16, #gpu.address_space<workgroup>> to memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>
        %104 = arith.addi %87, %c32 overflow<nsw, nuw> : index
        %105 = arith.addi %88, %104 overflow<nsw> : index
        %reinterpret_cast_24 = memref.reinterpret_cast %alloc to offset: [%105], sizes: [%c1073741822], strides: [1] : memref<1x64x1x68xf16, #gpu.address_space<workgroup>> to memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>
        %106 = arith.addi %90, %104 overflow<nsw> : index
        %reinterpret_cast_25 = memref.reinterpret_cast %alloc to offset: [%106], sizes: [%c1073741822], strides: [1] : memref<1x64x1x68xf16, #gpu.address_space<workgroup>> to memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>
        %107 = arith.addi %92, %104 overflow<nsw> : index
        %reinterpret_cast_26 = memref.reinterpret_cast %alloc to offset: [%107], sizes: [%c1073741822], strides: [1] : memref<1x64x1x68xf16, #gpu.address_space<workgroup>> to memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>
        %108 = arith.addi %94, %104 overflow<nsw> : index
        %reinterpret_cast_27 = memref.reinterpret_cast %alloc to offset: [%108], sizes: [%c1073741822], strides: [1] : memref<1x64x1x68xf16, #gpu.address_space<workgroup>> to memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>
        %109 = arith.addi %96, %104 overflow<nsw> : index
        %reinterpret_cast_28 = memref.reinterpret_cast %alloc to offset: [%109], sizes: [%c1073741822], strides: [1] : memref<1x64x1x68xf16, #gpu.address_space<workgroup>> to memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>
        %110 = arith.addi %98, %104 overflow<nsw> : index
        %reinterpret_cast_29 = memref.reinterpret_cast %alloc to offset: [%110], sizes: [%c1073741822], strides: [1] : memref<1x64x1x68xf16, #gpu.address_space<workgroup>> to memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>
        %111 = arith.addi %100, %104 overflow<nsw> : index
        %reinterpret_cast_30 = memref.reinterpret_cast %alloc to offset: [%111], sizes: [%c1073741822], strides: [1] : memref<1x64x1x68xf16, #gpu.address_space<workgroup>> to memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>
        %112 = arith.addi %102, %104 overflow<nsw> : index
        %reinterpret_cast_31 = memref.reinterpret_cast %alloc to offset: [%112], sizes: [%c1073741822], strides: [1] : memref<1x64x1x68xf16, #gpu.address_space<workgroup>> to memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>
        %113 = arith.addi %4, %c32 overflow<nsw, nuw> : index
        %114:4 = scf.for %arg4 = %c0 to %c22 step %c1 iter_args(%arg5 = %cst_12, %arg6 = %cst_13, %arg7 = %cst_14, %arg8 = %cst_14) -> (vector<1xf32>, vector<1xf32>, vector<16xf32>, vector<16xf32>) {
          amdgpu.lds_barrier
          %140 = arith.muli %arg4, %c64 overflow<nsw, nuw> : index
          %141 = arith.addi %65, %140 overflow<nsw, nuw> : index
          %142 = arith.cmpi slt, %141, %c1357 : index
          %143 = vector.splat %142 : vector<8xi1>
          %144 = vector.maskedload %43[%c0, %141, %7, %67], %143, %cst_7 : memref<1x1357x32x128xf16, strided<[5558272, 4096, 128, 1], offset: ?>>, vector<8xi1>, vector<8xf16> into vector<8xf16>
          vector.store %144, %alloc_16[%c0, %c0, %65, %67] : memref<1x1x64x132xf16, #gpu.address_space<workgroup>>, vector<8xf16>
          %145 = arith.addi %69, %140 overflow<nsw, nuw> : index
          %146 = arith.cmpi slt, %145, %c1357 : index
          %147 = vector.splat %146 : vector<8xi1>
          %148 = vector.maskedload %43[%c0, %145, %7, %67], %147, %cst_7 : memref<1x1357x32x128xf16, strided<[5558272, 4096, 128, 1], offset: ?>>, vector<8xi1>, vector<8xf16> into vector<8xf16>
          vector.store %148, %alloc_16[%c0, %c0, %69, %67] : memref<1x1x64x132xf16, #gpu.address_space<workgroup>>, vector<8xf16>
          %149 = arith.addi %71, %140 overflow<nsw, nuw> : index
          %150 = arith.cmpi slt, %149, %c1357 : index
          %151 = vector.splat %150 : vector<8xi1>
          %152 = vector.maskedload %43[%c0, %149, %7, %67], %151, %cst_7 : memref<1x1357x32x128xf16, strided<[5558272, 4096, 128, 1], offset: ?>>, vector<8xi1>, vector<8xf16> into vector<8xf16>
          vector.store %152, %alloc_16[%c0, %c0, %71, %67] : memref<1x1x64x132xf16, #gpu.address_space<workgroup>>, vector<8xf16>
          %153 = arith.addi %73, %140 overflow<nsw, nuw> : index
          %154 = arith.cmpi slt, %153, %c1357 : index
          %155 = vector.splat %154 : vector<8xi1>
          %156 = vector.maskedload %43[%c0, %153, %7, %67], %155, %cst_7 : memref<1x1357x32x128xf16, strided<[5558272, 4096, 128, 1], offset: ?>>, vector<8xi1>, vector<8xf16> into vector<8xf16>
          vector.store %156, %alloc_16[%c0, %c0, %73, %67] : memref<1x1x64x132xf16, #gpu.address_space<workgroup>>, vector<8xf16>
          %157 = arith.addi %79, %140 overflow<nsw, nuw> : index
          %158 = arith.cmpi slt, %157, %c1357 : index
          %159 = vector.splat %158 : vector<8xi1>
          %160 = vector.maskedload %42[%c0, %157, %7, %83], %159, %cst_7 : memref<1x1357x32x128xf16, strided<[5558272, 4096, 128, 1], offset: ?>>, vector<8xi1>, vector<8xf16> into vector<8xf16>
          vector.store %160, %alloc[%c0, %79, %c0, %81] : memref<1x64x1x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
          %161 = arith.addi %85, %140 overflow<nsw, nuw> : index
          %162 = arith.cmpi slt, %161, %c1357 : index
          %163 = vector.splat %162 : vector<8xi1>
          %164 = vector.maskedload %42[%c0, %161, %7, %83], %163, %cst_7 : memref<1x1357x32x128xf16, strided<[5558272, 4096, 128, 1], offset: ?>>, vector<8xi1>, vector<8xf16> into vector<8xf16>
          vector.store %164, %alloc[%c0, %85, %c0, %81] : memref<1x64x1x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
          amdgpu.lds_barrier
          %165 = vector.load %reinterpret_cast[%c0] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
          %166 = vector.load %reinterpret_cast[%c68] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
          %167 = vector.load %reinterpret_cast[%c136] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
          %168 = vector.load %reinterpret_cast[%c204] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
          %169 = vector.extract %165[0] : f16 from vector<1xf16>
          %170 = vector.extract %166[0] : f16 from vector<1xf16>
          %171 = vector.extract %167[0] : f16 from vector<1xf16>
          %172 = vector.extract %168[0] : f16 from vector<1xf16>
          %173 = vector.from_elements %169, %170, %171, %172 : vector<4xf16>
          %174 = vector.load %reinterpret_cast_17[%c0] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
          %175 = vector.load %reinterpret_cast_17[%c68] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
          %176 = vector.load %reinterpret_cast_17[%c136] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
          %177 = vector.load %reinterpret_cast_17[%c204] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
          %178 = vector.extract %174[0] : f16 from vector<1xf16>
          %179 = vector.extract %175[0] : f16 from vector<1xf16>
          %180 = vector.extract %176[0] : f16 from vector<1xf16>
          %181 = vector.extract %177[0] : f16 from vector<1xf16>
          %182 = vector.from_elements %178, %179, %180, %181 : vector<4xf16>
          %183 = vector.load %reinterpret_cast_18[%c0] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
          %184 = vector.load %reinterpret_cast_18[%c68] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
          %185 = vector.load %reinterpret_cast_18[%c136] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
          %186 = vector.load %reinterpret_cast_18[%c204] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
          %187 = vector.extract %183[0] : f16 from vector<1xf16>
          %188 = vector.extract %184[0] : f16 from vector<1xf16>
          %189 = vector.extract %185[0] : f16 from vector<1xf16>
          %190 = vector.extract %186[0] : f16 from vector<1xf16>
          %191 = vector.from_elements %187, %188, %189, %190 : vector<4xf16>
          %192 = vector.load %reinterpret_cast_19[%c0] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
          %193 = vector.load %reinterpret_cast_19[%c68] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
          %194 = vector.load %reinterpret_cast_19[%c136] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
          %195 = vector.load %reinterpret_cast_19[%c204] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
          %196 = vector.extract %192[0] : f16 from vector<1xf16>
          %197 = vector.extract %193[0] : f16 from vector<1xf16>
          %198 = vector.extract %194[0] : f16 from vector<1xf16>
          %199 = vector.extract %195[0] : f16 from vector<1xf16>
          %200 = vector.from_elements %196, %197, %198, %199 : vector<4xf16>
          %201 = vector.load %reinterpret_cast_20[%c0] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
          %202 = vector.load %reinterpret_cast_20[%c68] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
          %203 = vector.load %reinterpret_cast_20[%c136] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
          %204 = vector.load %reinterpret_cast_20[%c204] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
          %205 = vector.extract %201[0] : f16 from vector<1xf16>
          %206 = vector.extract %202[0] : f16 from vector<1xf16>
          %207 = vector.extract %203[0] : f16 from vector<1xf16>
          %208 = vector.extract %204[0] : f16 from vector<1xf16>
          %209 = vector.from_elements %205, %206, %207, %208 : vector<4xf16>
          %210 = vector.load %reinterpret_cast_21[%c0] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
          %211 = vector.load %reinterpret_cast_21[%c68] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
          %212 = vector.load %reinterpret_cast_21[%c136] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
          %213 = vector.load %reinterpret_cast_21[%c204] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
          %214 = vector.extract %210[0] : f16 from vector<1xf16>
          %215 = vector.extract %211[0] : f16 from vector<1xf16>
          %216 = vector.extract %212[0] : f16 from vector<1xf16>
          %217 = vector.extract %213[0] : f16 from vector<1xf16>
          %218 = vector.from_elements %214, %215, %216, %217 : vector<4xf16>
          %219 = vector.load %reinterpret_cast_22[%c0] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
          %220 = vector.load %reinterpret_cast_22[%c68] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
          %221 = vector.load %reinterpret_cast_22[%c136] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
          %222 = vector.load %reinterpret_cast_22[%c204] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
          %223 = vector.extract %219[0] : f16 from vector<1xf16>
          %224 = vector.extract %220[0] : f16 from vector<1xf16>
          %225 = vector.extract %221[0] : f16 from vector<1xf16>
          %226 = vector.extract %222[0] : f16 from vector<1xf16>
          %227 = vector.from_elements %223, %224, %225, %226 : vector<4xf16>
          %228 = vector.load %reinterpret_cast_23[%c0] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
          %229 = vector.load %reinterpret_cast_23[%c68] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
          %230 = vector.load %reinterpret_cast_23[%c136] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
          %231 = vector.load %reinterpret_cast_23[%c204] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
          %232 = vector.extract %228[0] : f16 from vector<1xf16>
          %233 = vector.extract %229[0] : f16 from vector<1xf16>
          %234 = vector.extract %230[0] : f16 from vector<1xf16>
          %235 = vector.extract %231[0] : f16 from vector<1xf16>
          %236 = vector.from_elements %232, %233, %234, %235 : vector<4xf16>
          %237 = vector.load %reinterpret_cast_24[%c0] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
          %238 = vector.load %reinterpret_cast_24[%c68] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
          %239 = vector.load %reinterpret_cast_24[%c136] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
          %240 = vector.load %reinterpret_cast_24[%c204] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
          %241 = vector.extract %237[0] : f16 from vector<1xf16>
          %242 = vector.extract %238[0] : f16 from vector<1xf16>
          %243 = vector.extract %239[0] : f16 from vector<1xf16>
          %244 = vector.extract %240[0] : f16 from vector<1xf16>
          %245 = vector.from_elements %241, %242, %243, %244 : vector<4xf16>
          %246 = vector.load %reinterpret_cast_25[%c0] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
          %247 = vector.load %reinterpret_cast_25[%c68] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
          %248 = vector.load %reinterpret_cast_25[%c136] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
          %249 = vector.load %reinterpret_cast_25[%c204] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
          %250 = vector.extract %246[0] : f16 from vector<1xf16>
          %251 = vector.extract %247[0] : f16 from vector<1xf16>
          %252 = vector.extract %248[0] : f16 from vector<1xf16>
          %253 = vector.extract %249[0] : f16 from vector<1xf16>
          %254 = vector.from_elements %250, %251, %252, %253 : vector<4xf16>
          %255 = vector.load %reinterpret_cast_26[%c0] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
          %256 = vector.load %reinterpret_cast_26[%c68] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
          %257 = vector.load %reinterpret_cast_26[%c136] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
          %258 = vector.load %reinterpret_cast_26[%c204] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
          %259 = vector.extract %255[0] : f16 from vector<1xf16>
          %260 = vector.extract %256[0] : f16 from vector<1xf16>
          %261 = vector.extract %257[0] : f16 from vector<1xf16>
          %262 = vector.extract %258[0] : f16 from vector<1xf16>
          %263 = vector.from_elements %259, %260, %261, %262 : vector<4xf16>
          %264 = vector.load %reinterpret_cast_27[%c0] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
          %265 = vector.load %reinterpret_cast_27[%c68] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
          %266 = vector.load %reinterpret_cast_27[%c136] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
          %267 = vector.load %reinterpret_cast_27[%c204] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
          %268 = vector.extract %264[0] : f16 from vector<1xf16>
          %269 = vector.extract %265[0] : f16 from vector<1xf16>
          %270 = vector.extract %266[0] : f16 from vector<1xf16>
          %271 = vector.extract %267[0] : f16 from vector<1xf16>
          %272 = vector.from_elements %268, %269, %270, %271 : vector<4xf16>
          %273 = vector.load %reinterpret_cast_28[%c0] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
          %274 = vector.load %reinterpret_cast_28[%c68] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
          %275 = vector.load %reinterpret_cast_28[%c136] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
          %276 = vector.load %reinterpret_cast_28[%c204] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
          %277 = vector.extract %273[0] : f16 from vector<1xf16>
          %278 = vector.extract %274[0] : f16 from vector<1xf16>
          %279 = vector.extract %275[0] : f16 from vector<1xf16>
          %280 = vector.extract %276[0] : f16 from vector<1xf16>
          %281 = vector.from_elements %277, %278, %279, %280 : vector<4xf16>
          %282 = vector.load %reinterpret_cast_29[%c0] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
          %283 = vector.load %reinterpret_cast_29[%c68] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
          %284 = vector.load %reinterpret_cast_29[%c136] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
          %285 = vector.load %reinterpret_cast_29[%c204] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
          %286 = vector.extract %282[0] : f16 from vector<1xf16>
          %287 = vector.extract %283[0] : f16 from vector<1xf16>
          %288 = vector.extract %284[0] : f16 from vector<1xf16>
          %289 = vector.extract %285[0] : f16 from vector<1xf16>
          %290 = vector.from_elements %286, %287, %288, %289 : vector<4xf16>
          %291 = vector.load %reinterpret_cast_30[%c0] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
          %292 = vector.load %reinterpret_cast_30[%c68] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
          %293 = vector.load %reinterpret_cast_30[%c136] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
          %294 = vector.load %reinterpret_cast_30[%c204] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
          %295 = vector.extract %291[0] : f16 from vector<1xf16>
          %296 = vector.extract %292[0] : f16 from vector<1xf16>
          %297 = vector.extract %293[0] : f16 from vector<1xf16>
          %298 = vector.extract %294[0] : f16 from vector<1xf16>
          %299 = vector.from_elements %295, %296, %297, %298 : vector<4xf16>
          %300 = vector.load %reinterpret_cast_31[%c0] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
          %301 = vector.load %reinterpret_cast_31[%c68] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
          %302 = vector.load %reinterpret_cast_31[%c136] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
          %303 = vector.load %reinterpret_cast_31[%c204] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
          %304 = vector.extract %300[0] : f16 from vector<1xf16>
          %305 = vector.extract %301[0] : f16 from vector<1xf16>
          %306 = vector.extract %302[0] : f16 from vector<1xf16>
          %307 = vector.extract %303[0] : f16 from vector<1xf16>
          %308 = vector.from_elements %304, %305, %306, %307 : vector<4xf16>
          %309 = vector.load %alloc_16[%c0, %c0, %4, %10] : memref<1x1x64x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %310 = vector.load %alloc_16[%c0, %c0, %4, %12] : memref<1x1x64x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %311 = vector.load %alloc_16[%c0, %c0, %4, %14] : memref<1x1x64x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %312 = vector.load %alloc_16[%c0, %c0, %4, %16] : memref<1x1x64x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %313 = vector.load %alloc_16[%c0, %c0, %4, %18] : memref<1x1x64x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %314 = vector.load %alloc_16[%c0, %c0, %4, %20] : memref<1x1x64x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %315 = vector.load %alloc_16[%c0, %c0, %4, %22] : memref<1x1x64x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %316 = vector.load %alloc_16[%c0, %c0, %4, %24] : memref<1x1x64x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %317 = vector.load %alloc_16[%c0, %c0, %4, %26] : memref<1x1x64x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %318 = vector.load %alloc_16[%c0, %c0, %4, %28] : memref<1x1x64x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %319 = vector.load %alloc_16[%c0, %c0, %4, %30] : memref<1x1x64x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %320 = vector.load %alloc_16[%c0, %c0, %4, %32] : memref<1x1x64x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %321 = vector.load %alloc_16[%c0, %c0, %4, %34] : memref<1x1x64x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %322 = vector.load %alloc_16[%c0, %c0, %4, %36] : memref<1x1x64x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %323 = vector.load %alloc_16[%c0, %c0, %4, %38] : memref<1x1x64x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %324 = vector.load %alloc_16[%c0, %c0, %4, %40] : memref<1x1x64x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %325 = vector.load %alloc_16[%c0, %c0, %113, %10] : memref<1x1x64x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %326 = vector.load %alloc_16[%c0, %c0, %113, %12] : memref<1x1x64x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %327 = vector.load %alloc_16[%c0, %c0, %113, %14] : memref<1x1x64x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %328 = vector.load %alloc_16[%c0, %c0, %113, %16] : memref<1x1x64x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %329 = vector.load %alloc_16[%c0, %c0, %113, %18] : memref<1x1x64x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %330 = vector.load %alloc_16[%c0, %c0, %113, %20] : memref<1x1x64x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %331 = vector.load %alloc_16[%c0, %c0, %113, %22] : memref<1x1x64x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %332 = vector.load %alloc_16[%c0, %c0, %113, %24] : memref<1x1x64x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %333 = vector.load %alloc_16[%c0, %c0, %113, %26] : memref<1x1x64x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %334 = vector.load %alloc_16[%c0, %c0, %113, %28] : memref<1x1x64x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %335 = vector.load %alloc_16[%c0, %c0, %113, %30] : memref<1x1x64x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %336 = vector.load %alloc_16[%c0, %c0, %113, %32] : memref<1x1x64x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %337 = vector.load %alloc_16[%c0, %c0, %113, %34] : memref<1x1x64x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %338 = vector.load %alloc_16[%c0, %c0, %113, %36] : memref<1x1x64x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %339 = vector.load %alloc_16[%c0, %c0, %113, %38] : memref<1x1x64x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %340 = vector.load %alloc_16[%c0, %c0, %113, %40] : memref<1x1x64x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %341 = amdgpu.mfma %309 * %44 + %cst_14 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %342 = amdgpu.mfma %310 * %45 + %341 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %343 = amdgpu.mfma %311 * %46 + %342 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %344 = amdgpu.mfma %312 * %47 + %343 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %345 = amdgpu.mfma %313 * %48 + %344 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %346 = amdgpu.mfma %314 * %49 + %345 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %347 = amdgpu.mfma %315 * %50 + %346 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %348 = amdgpu.mfma %316 * %51 + %347 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %349 = amdgpu.mfma %317 * %52 + %348 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %350 = amdgpu.mfma %318 * %53 + %349 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %351 = amdgpu.mfma %319 * %54 + %350 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %352 = amdgpu.mfma %320 * %55 + %351 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %353 = amdgpu.mfma %321 * %56 + %352 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %354 = amdgpu.mfma %322 * %57 + %353 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %355 = amdgpu.mfma %323 * %58 + %354 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %356 = amdgpu.mfma %324 * %59 + %355 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %357 = amdgpu.mfma %325 * %44 + %cst_14 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %358 = amdgpu.mfma %326 * %45 + %357 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %359 = amdgpu.mfma %327 * %46 + %358 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %360 = amdgpu.mfma %328 * %47 + %359 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %361 = amdgpu.mfma %329 * %48 + %360 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %362 = amdgpu.mfma %330 * %49 + %361 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %363 = amdgpu.mfma %331 * %50 + %362 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %364 = amdgpu.mfma %332 * %51 + %363 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %365 = amdgpu.mfma %333 * %52 + %364 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %366 = amdgpu.mfma %334 * %53 + %365 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %367 = amdgpu.mfma %335 * %54 + %366 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %368 = amdgpu.mfma %336 * %55 + %367 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %369 = amdgpu.mfma %337 * %56 + %368 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %370 = amdgpu.mfma %338 * %57 + %369 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %371 = amdgpu.mfma %339 * %58 + %370 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %372 = amdgpu.mfma %340 * %59 + %371 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %373 = arith.addi %140, %10 overflow<nsw, nuw> : index
          %374 = vector.splat %373 : vector<4xindex>
          %375 = arith.addi %374, %cst_10 overflow<nsw, nuw> : vector<4xindex>
          %376 = arith.index_cast %375 : vector<4xindex> to vector<4xi32>
          %377 = arith.addi %375, %cst_6 overflow<nsw, nuw> : vector<4xindex>
          %378 = arith.index_cast %377 : vector<4xindex> to vector<4xi32>
          %379 = arith.addi %375, %cst_5 overflow<nsw, nuw> : vector<4xindex>
          %380 = arith.index_cast %379 : vector<4xindex> to vector<4xi32>
          %381 = arith.addi %375, %cst_4 overflow<nsw, nuw> : vector<4xindex>
          %382 = arith.index_cast %381 : vector<4xindex> to vector<4xi32>
          %383 = vector.insert_strided_slice %376, %cst_9 {offsets = [0], strides = [1]} : vector<4xi32> into vector<16xi32>
          %384 = vector.insert_strided_slice %378, %383 {offsets = [4], strides = [1]} : vector<4xi32> into vector<16xi32>
          %385 = vector.insert_strided_slice %380, %384 {offsets = [8], strides = [1]} : vector<4xi32> into vector<16xi32>
          %386 = vector.insert_strided_slice %382, %385 {offsets = [12], strides = [1]} : vector<4xi32> into vector<16xi32>
          %387 = arith.addi %375, %cst_3 overflow<nsw, nuw> : vector<4xindex>
          %388 = arith.index_cast %387 : vector<4xindex> to vector<4xi32>
          %389 = arith.addi %375, %cst_2 overflow<nsw, nuw> : vector<4xindex>
          %390 = arith.index_cast %389 : vector<4xindex> to vector<4xi32>
          %391 = arith.addi %375, %cst_1 overflow<nsw, nuw> : vector<4xindex>
          %392 = arith.index_cast %391 : vector<4xindex> to vector<4xi32>
          %393 = arith.addi %375, %cst_0 overflow<nsw, nuw> : vector<4xindex>
          %394 = arith.index_cast %393 : vector<4xindex> to vector<4xi32>
          %395 = vector.insert_strided_slice %388, %cst_9 {offsets = [0], strides = [1]} : vector<4xi32> into vector<16xi32>
          %396 = vector.insert_strided_slice %390, %395 {offsets = [4], strides = [1]} : vector<4xi32> into vector<16xi32>
          %397 = vector.insert_strided_slice %392, %396 {offsets = [8], strides = [1]} : vector<4xi32> into vector<16xi32>
          %398 = vector.insert_strided_slice %394, %397 {offsets = [12], strides = [1]} : vector<4xi32> into vector<16xi32>
          %399 = arith.index_cast %386 : vector<16xi32> to vector<16xindex>
          %400 = arith.cmpi slt, %399, %cst : vector<16xindex>
          %401 = arith.index_cast %398 : vector<16xi32> to vector<16xindex>
          %402 = arith.cmpi slt, %401, %cst : vector<16xindex>
          %403 = arith.select %400, %cst_14, %cst_11 : vector<16xi1>, vector<16xf32>
          %404 = arith.select %402, %cst_14, %cst_11 : vector<16xi1>, vector<16xf32>
          %405 = arith.addf %356, %403 : vector<16xf32>
          %406 = arith.addf %372, %404 : vector<16xf32>
          %407 = vector.extract_strided_slice %405 {offsets = [0], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
          %408 = vector.extract_strided_slice %405 {offsets = [1], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
          %409 = arith.maximumf %407, %408 : vector<1xf32>
          %410 = vector.extract_strided_slice %405 {offsets = [2], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
          %411 = arith.maximumf %409, %410 : vector<1xf32>
          %412 = vector.extract_strided_slice %405 {offsets = [3], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
          %413 = arith.maximumf %411, %412 : vector<1xf32>
          %414 = vector.extract_strided_slice %405 {offsets = [4], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
          %415 = arith.maximumf %413, %414 : vector<1xf32>
          %416 = vector.extract_strided_slice %405 {offsets = [5], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
          %417 = arith.maximumf %415, %416 : vector<1xf32>
          %418 = vector.extract_strided_slice %405 {offsets = [6], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
          %419 = arith.maximumf %417, %418 : vector<1xf32>
          %420 = vector.extract_strided_slice %405 {offsets = [7], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
          %421 = arith.maximumf %419, %420 : vector<1xf32>
          %422 = vector.extract_strided_slice %405 {offsets = [8], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
          %423 = arith.maximumf %421, %422 : vector<1xf32>
          %424 = vector.extract_strided_slice %405 {offsets = [9], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
          %425 = arith.maximumf %423, %424 : vector<1xf32>
          %426 = vector.extract_strided_slice %405 {offsets = [10], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
          %427 = arith.maximumf %425, %426 : vector<1xf32>
          %428 = vector.extract_strided_slice %405 {offsets = [11], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
          %429 = arith.maximumf %427, %428 : vector<1xf32>
          %430 = vector.extract_strided_slice %405 {offsets = [12], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
          %431 = arith.maximumf %429, %430 : vector<1xf32>
          %432 = vector.extract_strided_slice %405 {offsets = [13], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
          %433 = arith.maximumf %431, %432 : vector<1xf32>
          %434 = vector.extract_strided_slice %405 {offsets = [14], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
          %435 = arith.maximumf %433, %434 : vector<1xf32>
          %436 = vector.extract_strided_slice %405 {offsets = [15], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
          %437 = arith.maximumf %435, %436 : vector<1xf32>
          %438 = vector.extract_strided_slice %406 {offsets = [0], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
          %439 = vector.extract_strided_slice %406 {offsets = [1], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
          %440 = arith.maximumf %438, %439 : vector<1xf32>
          %441 = vector.extract_strided_slice %406 {offsets = [2], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
          %442 = arith.maximumf %440, %441 : vector<1xf32>
          %443 = vector.extract_strided_slice %406 {offsets = [3], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
          %444 = arith.maximumf %442, %443 : vector<1xf32>
          %445 = vector.extract_strided_slice %406 {offsets = [4], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
          %446 = arith.maximumf %444, %445 : vector<1xf32>
          %447 = vector.extract_strided_slice %406 {offsets = [5], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
          %448 = arith.maximumf %446, %447 : vector<1xf32>
          %449 = vector.extract_strided_slice %406 {offsets = [6], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
          %450 = arith.maximumf %448, %449 : vector<1xf32>
          %451 = vector.extract_strided_slice %406 {offsets = [7], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
          %452 = arith.maximumf %450, %451 : vector<1xf32>
          %453 = vector.extract_strided_slice %406 {offsets = [8], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
          %454 = arith.maximumf %452, %453 : vector<1xf32>
          %455 = vector.extract_strided_slice %406 {offsets = [9], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
          %456 = arith.maximumf %454, %455 : vector<1xf32>
          %457 = vector.extract_strided_slice %406 {offsets = [10], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
          %458 = arith.maximumf %456, %457 : vector<1xf32>
          %459 = vector.extract_strided_slice %406 {offsets = [11], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
          %460 = arith.maximumf %458, %459 : vector<1xf32>
          %461 = vector.extract_strided_slice %406 {offsets = [12], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
          %462 = arith.maximumf %460, %461 : vector<1xf32>
          %463 = vector.extract_strided_slice %406 {offsets = [13], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
          %464 = arith.maximumf %462, %463 : vector<1xf32>
          %465 = vector.extract_strided_slice %406 {offsets = [14], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
          %466 = arith.maximumf %464, %465 : vector<1xf32>
          %467 = vector.extract_strided_slice %406 {offsets = [15], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
          %468 = arith.maximumf %466, %467 : vector<1xf32>
          %469 = arith.maximumf %437, %468 : vector<1xf32>
          %470 = vector.extract %469[0] : f32 from vector<1xf32>
          %shuffleResult, %valid = gpu.shuffle  xor %470, %c32_i32, %c64_i32 : f32
          %471 = vector.broadcast %shuffleResult : f32 to vector<1xf32>
          %472 = arith.maximumf %469, %471 : vector<1xf32>
          %473 = arith.maximumf %arg5, %472 : vector<1xf32>
          %474 = arith.subf %arg5, %473 : vector<1xf32>
          %475 = math.exp2 %474 : vector<1xf32>
          %476 = vector.extract %473[0] : f32 from vector<1xf32>
          %477 = vector.splat %476 : vector<16xf32>
          %478 = arith.subf %405, %477 : vector<16xf32>
          %479 = arith.subf %406, %477 : vector<16xf32>
          %480 = math.exp2 %478 : vector<16xf32>
          %481 = math.exp2 %479 : vector<16xf32>
          %482 = arith.mulf %arg6, %475 : vector<1xf32>
          %483 = arith.addf %480, %481 : vector<16xf32>
          %484 = vector.extract_strided_slice %483 {offsets = [0], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
          %485 = vector.extract_strided_slice %483 {offsets = [1], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
          %486 = arith.addf %484, %485 : vector<1xf32>
          %487 = vector.extract_strided_slice %483 {offsets = [2], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
          %488 = arith.addf %486, %487 : vector<1xf32>
          %489 = vector.extract_strided_slice %483 {offsets = [3], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
          %490 = arith.addf %488, %489 : vector<1xf32>
          %491 = vector.extract_strided_slice %483 {offsets = [4], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
          %492 = arith.addf %490, %491 : vector<1xf32>
          %493 = vector.extract_strided_slice %483 {offsets = [5], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
          %494 = arith.addf %492, %493 : vector<1xf32>
          %495 = vector.extract_strided_slice %483 {offsets = [6], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
          %496 = arith.addf %494, %495 : vector<1xf32>
          %497 = vector.extract_strided_slice %483 {offsets = [7], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
          %498 = arith.addf %496, %497 : vector<1xf32>
          %499 = vector.extract_strided_slice %483 {offsets = [8], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
          %500 = arith.addf %498, %499 : vector<1xf32>
          %501 = vector.extract_strided_slice %483 {offsets = [9], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
          %502 = arith.addf %500, %501 : vector<1xf32>
          %503 = vector.extract_strided_slice %483 {offsets = [10], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
          %504 = arith.addf %502, %503 : vector<1xf32>
          %505 = vector.extract_strided_slice %483 {offsets = [11], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
          %506 = arith.addf %504, %505 : vector<1xf32>
          %507 = vector.extract_strided_slice %483 {offsets = [12], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
          %508 = arith.addf %506, %507 : vector<1xf32>
          %509 = vector.extract_strided_slice %483 {offsets = [13], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
          %510 = arith.addf %508, %509 : vector<1xf32>
          %511 = vector.extract_strided_slice %483 {offsets = [14], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
          %512 = arith.addf %510, %511 : vector<1xf32>
          %513 = vector.extract_strided_slice %483 {offsets = [15], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
          %514 = arith.addf %512, %513 : vector<1xf32>
          %515 = vector.extract %514[0] : f32 from vector<1xf32>
          %shuffleResult_32, %valid_33 = gpu.shuffle  xor %515, %c32_i32, %c64_i32 : f32
          %516 = vector.broadcast %shuffleResult_32 : f32 to vector<1xf32>
          %517 = arith.addf %514, %516 : vector<1xf32>
          %518 = arith.addf %482, %517 : vector<1xf32>
          %519 = arith.truncf %480 : vector<16xf32> to vector<16xf16>
          %520 = arith.truncf %481 : vector<16xf32> to vector<16xf16>
          %521 = vector.extract %475[0] : f32 from vector<1xf32>
          %522 = vector.splat %521 : vector<16xf32>
          %523 = arith.mulf %arg7, %522 : vector<16xf32>
          %524 = arith.mulf %arg8, %522 : vector<16xf32>
          %525 = vector.extract_strided_slice %519 {offsets = [0], sizes = [4], strides = [1]} : vector<16xf16> to vector<4xf16>
          %526 = vector.extract_strided_slice %519 {offsets = [4], sizes = [4], strides = [1]} : vector<16xf16> to vector<4xf16>
          %527 = vector.extract_strided_slice %519 {offsets = [8], sizes = [4], strides = [1]} : vector<16xf16> to vector<4xf16>
          %528 = vector.extract_strided_slice %519 {offsets = [12], sizes = [4], strides = [1]} : vector<16xf16> to vector<4xf16>
          %529 = vector.extract_strided_slice %520 {offsets = [0], sizes = [4], strides = [1]} : vector<16xf16> to vector<4xf16>
          %530 = vector.extract_strided_slice %520 {offsets = [4], sizes = [4], strides = [1]} : vector<16xf16> to vector<4xf16>
          %531 = vector.extract_strided_slice %520 {offsets = [8], sizes = [4], strides = [1]} : vector<16xf16> to vector<4xf16>
          %532 = vector.extract_strided_slice %520 {offsets = [12], sizes = [4], strides = [1]} : vector<16xf16> to vector<4xf16>
          %533 = amdgpu.mfma %173 * %525 + %523 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %534 = amdgpu.mfma %182 * %526 + %533 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %535 = amdgpu.mfma %191 * %527 + %534 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %536 = amdgpu.mfma %200 * %528 + %535 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %537 = amdgpu.mfma %209 * %529 + %536 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %538 = amdgpu.mfma %218 * %530 + %537 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %539 = amdgpu.mfma %227 * %531 + %538 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %540 = amdgpu.mfma %236 * %532 + %539 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %541 = amdgpu.mfma %245 * %525 + %524 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %542 = amdgpu.mfma %254 * %526 + %541 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %543 = amdgpu.mfma %263 * %527 + %542 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %544 = amdgpu.mfma %272 * %528 + %543 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %545 = amdgpu.mfma %281 * %529 + %544 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %546 = amdgpu.mfma %290 * %530 + %545 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %547 = amdgpu.mfma %299 * %531 + %546 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %548 = amdgpu.mfma %308 * %532 + %547 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          scf.yield %473, %518, %540, %548 : vector<1xf32>, vector<1xf32>, vector<16xf32>, vector<16xf32>
        }
        %115 = arith.divf %cst_8, %114#1 : vector<1xf32>
        %116 = vector.extract %115[0] : f32 from vector<1xf32>
        %117 = vector.splat %116 : vector<16xf32>
        %118 = arith.mulf %114#2, %117 : vector<16xf32>
        %119 = arith.mulf %114#3, %117 : vector<16xf32>
        %120 = vector.extract_strided_slice %118 {offsets = [0], sizes = [4], strides = [1]} : vector<16xf32> to vector<4xf32>
        %121 = stream.binding.subspan %arg3[%c0] : !stream.binding -> memref<1x1024x32x128xf32, strided<[4194304, 4096, 128, 1], offset: ?>>
        %122 = arith.muli %thread_id_y, %c64 overflow<nsw, nuw> : index
        %123 = arith.muli %workgroup_id_1, %c64 overflow<nsw, nuw> : index
        %124 = arith.addi %123, %122 overflow<nsw, nuw> : index
        %125 = arith.addi %124, %10 overflow<nsw, nuw> : index
        vector.store %120, %121[%c0, %6, %7, %125] : memref<1x1024x32x128xf32, strided<[4194304, 4096, 128, 1], offset: ?>>, vector<4xf32>
        %126 = vector.extract_strided_slice %118 {offsets = [4], sizes = [4], strides = [1]} : vector<16xf32> to vector<4xf32>
        %127 = arith.addi %125, %c8 overflow<nsw, nuw> : index
        vector.store %126, %121[%c0, %6, %7, %127] : memref<1x1024x32x128xf32, strided<[4194304, 4096, 128, 1], offset: ?>>, vector<4xf32>
        %128 = vector.extract_strided_slice %118 {offsets = [8], sizes = [4], strides = [1]} : vector<16xf32> to vector<4xf32>
        %129 = arith.addi %125, %c16 overflow<nsw, nuw> : index
        vector.store %128, %121[%c0, %6, %7, %129] : memref<1x1024x32x128xf32, strided<[4194304, 4096, 128, 1], offset: ?>>, vector<4xf32>
        %130 = vector.extract_strided_slice %118 {offsets = [12], sizes = [4], strides = [1]} : vector<16xf32> to vector<4xf32>
        %131 = arith.addi %125, %c24 overflow<nsw, nuw> : index
        vector.store %130, %121[%c0, %6, %7, %131] : memref<1x1024x32x128xf32, strided<[4194304, 4096, 128, 1], offset: ?>>, vector<4xf32>
        %132 = vector.extract_strided_slice %119 {offsets = [0], sizes = [4], strides = [1]} : vector<16xf32> to vector<4xf32>
        %133 = arith.addi %125, %c32 overflow<nsw, nuw> : index
        vector.store %132, %121[%c0, %6, %7, %133] : memref<1x1024x32x128xf32, strided<[4194304, 4096, 128, 1], offset: ?>>, vector<4xf32>
        %134 = vector.extract_strided_slice %119 {offsets = [4], sizes = [4], strides = [1]} : vector<16xf32> to vector<4xf32>
        %135 = arith.addi %125, %c40 overflow<nsw, nuw> : index
        vector.store %134, %121[%c0, %6, %7, %135] : memref<1x1024x32x128xf32, strided<[4194304, 4096, 128, 1], offset: ?>>, vector<4xf32>
        %136 = vector.extract_strided_slice %119 {offsets = [8], sizes = [4], strides = [1]} : vector<16xf32> to vector<4xf32>
        %137 = arith.addi %125, %c48 overflow<nsw, nuw> : index
        vector.store %136, %121[%c0, %6, %7, %137] : memref<1x1024x32x128xf32, strided<[4194304, 4096, 128, 1], offset: ?>>, vector<4xf32>
        %138 = vector.extract_strided_slice %119 {offsets = [12], sizes = [4], strides = [1]} : vector<16xf32> to vector<4xf32>
        %139 = arith.addi %125, %c56 overflow<nsw, nuw> : index
        vector.store %138, %121[%c0, %6, %7, %139] : memref<1x1024x32x128xf32, strided<[4194304, 4096, 128, 1], offset: ?>>, vector<4xf32>
        return
      }
    }
  }
  func.func @isolated_benchmark(%arg0: tensor<1x1024x32x128xf16>, %arg1: tensor<1x1357x32x128xf16>, %arg2: tensor<1x1357x32x128xf16>, %arg3: tensor<1x1024x32x128xf32>) -> tensor<1x1024x32x128xf32> {
    %0 = flow.dispatch @base_attention::@base_attention(%arg0, %arg1, %arg2, %arg3) : (tensor<1x1024x32x128xf16>, tensor<1x1357x32x128xf16>, tensor<1x1357x32x128xf16>, tensor<1x1024x32x128xf32>) -> %arg3
    return %0 : tensor<1x1024x32x128xf32>
  }
}
