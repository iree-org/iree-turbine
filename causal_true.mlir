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
        %114 = vector.splat %6 : vector<1xindex>
        %115 = arith.index_cast %114 : vector<1xindex> to vector<1xi32>
        %116 = vector.extract %115[0] : i32 from vector<1xi32>
        %117 = vector.splat %116 : vector<16xi32>
        %118:4 = scf.for %arg4 = %c0 to %c22 step %c1 iter_args(%arg5 = %cst_12, %arg6 = %cst_13, %arg7 = %cst_14, %arg8 = %cst_14) -> (vector<1xf32>, vector<1xf32>, vector<16xf32>, vector<16xf32>) {
          amdgpu.lds_barrier
          %144 = arith.muli %arg4, %c64 overflow<nsw, nuw> : index
          %145 = arith.addi %65, %144 overflow<nsw, nuw> : index
          %146 = arith.cmpi slt, %145, %c1357 : index
          %147 = vector.splat %146 : vector<8xi1>
          %148 = vector.maskedload %43[%c0, %145, %7, %67], %147, %cst_7 : memref<1x1357x32x128xf16, strided<[5558272, 4096, 128, 1], offset: ?>>, vector<8xi1>, vector<8xf16> into vector<8xf16>
          vector.store %148, %alloc_16[%c0, %c0, %65, %67] : memref<1x1x64x132xf16, #gpu.address_space<workgroup>>, vector<8xf16>
          %149 = arith.addi %69, %144 overflow<nsw, nuw> : index
          %150 = arith.cmpi slt, %149, %c1357 : index
          %151 = vector.splat %150 : vector<8xi1>
          %152 = vector.maskedload %43[%c0, %149, %7, %67], %151, %cst_7 : memref<1x1357x32x128xf16, strided<[5558272, 4096, 128, 1], offset: ?>>, vector<8xi1>, vector<8xf16> into vector<8xf16>
          vector.store %152, %alloc_16[%c0, %c0, %69, %67] : memref<1x1x64x132xf16, #gpu.address_space<workgroup>>, vector<8xf16>
          %153 = arith.addi %71, %144 overflow<nsw, nuw> : index
          %154 = arith.cmpi slt, %153, %c1357 : index
          %155 = vector.splat %154 : vector<8xi1>
          %156 = vector.maskedload %43[%c0, %153, %7, %67], %155, %cst_7 : memref<1x1357x32x128xf16, strided<[5558272, 4096, 128, 1], offset: ?>>, vector<8xi1>, vector<8xf16> into vector<8xf16>
          vector.store %156, %alloc_16[%c0, %c0, %71, %67] : memref<1x1x64x132xf16, #gpu.address_space<workgroup>>, vector<8xf16>
          %157 = arith.addi %73, %144 overflow<nsw, nuw> : index
          %158 = arith.cmpi slt, %157, %c1357 : index
          %159 = vector.splat %158 : vector<8xi1>
          %160 = vector.maskedload %43[%c0, %157, %7, %67], %159, %cst_7 : memref<1x1357x32x128xf16, strided<[5558272, 4096, 128, 1], offset: ?>>, vector<8xi1>, vector<8xf16> into vector<8xf16>
          vector.store %160, %alloc_16[%c0, %c0, %73, %67] : memref<1x1x64x132xf16, #gpu.address_space<workgroup>>, vector<8xf16>
          %161 = arith.addi %79, %144 overflow<nsw, nuw> : index
          %162 = arith.cmpi slt, %161, %c1357 : index
          %163 = vector.splat %162 : vector<8xi1>
          %164 = vector.maskedload %42[%c0, %161, %7, %83], %163, %cst_7 : memref<1x1357x32x128xf16, strided<[5558272, 4096, 128, 1], offset: ?>>, vector<8xi1>, vector<8xf16> into vector<8xf16>
          vector.store %164, %alloc[%c0, %79, %c0, %81] : memref<1x64x1x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
          %165 = arith.addi %85, %144 overflow<nsw, nuw> : index
          %166 = arith.cmpi slt, %165, %c1357 : index
          %167 = vector.splat %166 : vector<8xi1>
          %168 = vector.maskedload %42[%c0, %165, %7, %83], %167, %cst_7 : memref<1x1357x32x128xf16, strided<[5558272, 4096, 128, 1], offset: ?>>, vector<8xi1>, vector<8xf16> into vector<8xf16>
          vector.store %168, %alloc[%c0, %85, %c0, %81] : memref<1x64x1x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
          amdgpu.lds_barrier
          %169 = vector.load %reinterpret_cast[%c0] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
          %170 = vector.load %reinterpret_cast[%c68] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
          %171 = vector.load %reinterpret_cast[%c136] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
          %172 = vector.load %reinterpret_cast[%c204] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
          %173 = vector.extract %169[0] : f16 from vector<1xf16>
          %174 = vector.extract %170[0] : f16 from vector<1xf16>
          %175 = vector.extract %171[0] : f16 from vector<1xf16>
          %176 = vector.extract %172[0] : f16 from vector<1xf16>
          %177 = vector.from_elements %173, %174, %175, %176 : vector<4xf16>
          %178 = vector.load %reinterpret_cast_17[%c0] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
          %179 = vector.load %reinterpret_cast_17[%c68] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
          %180 = vector.load %reinterpret_cast_17[%c136] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
          %181 = vector.load %reinterpret_cast_17[%c204] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
          %182 = vector.extract %178[0] : f16 from vector<1xf16>
          %183 = vector.extract %179[0] : f16 from vector<1xf16>
          %184 = vector.extract %180[0] : f16 from vector<1xf16>
          %185 = vector.extract %181[0] : f16 from vector<1xf16>
          %186 = vector.from_elements %182, %183, %184, %185 : vector<4xf16>
          %187 = vector.load %reinterpret_cast_18[%c0] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
          %188 = vector.load %reinterpret_cast_18[%c68] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
          %189 = vector.load %reinterpret_cast_18[%c136] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
          %190 = vector.load %reinterpret_cast_18[%c204] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
          %191 = vector.extract %187[0] : f16 from vector<1xf16>
          %192 = vector.extract %188[0] : f16 from vector<1xf16>
          %193 = vector.extract %189[0] : f16 from vector<1xf16>
          %194 = vector.extract %190[0] : f16 from vector<1xf16>
          %195 = vector.from_elements %191, %192, %193, %194 : vector<4xf16>
          %196 = vector.load %reinterpret_cast_19[%c0] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
          %197 = vector.load %reinterpret_cast_19[%c68] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
          %198 = vector.load %reinterpret_cast_19[%c136] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
          %199 = vector.load %reinterpret_cast_19[%c204] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
          %200 = vector.extract %196[0] : f16 from vector<1xf16>
          %201 = vector.extract %197[0] : f16 from vector<1xf16>
          %202 = vector.extract %198[0] : f16 from vector<1xf16>
          %203 = vector.extract %199[0] : f16 from vector<1xf16>
          %204 = vector.from_elements %200, %201, %202, %203 : vector<4xf16>
          %205 = vector.load %reinterpret_cast_20[%c0] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
          %206 = vector.load %reinterpret_cast_20[%c68] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
          %207 = vector.load %reinterpret_cast_20[%c136] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
          %208 = vector.load %reinterpret_cast_20[%c204] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
          %209 = vector.extract %205[0] : f16 from vector<1xf16>
          %210 = vector.extract %206[0] : f16 from vector<1xf16>
          %211 = vector.extract %207[0] : f16 from vector<1xf16>
          %212 = vector.extract %208[0] : f16 from vector<1xf16>
          %213 = vector.from_elements %209, %210, %211, %212 : vector<4xf16>
          %214 = vector.load %reinterpret_cast_21[%c0] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
          %215 = vector.load %reinterpret_cast_21[%c68] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
          %216 = vector.load %reinterpret_cast_21[%c136] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
          %217 = vector.load %reinterpret_cast_21[%c204] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
          %218 = vector.extract %214[0] : f16 from vector<1xf16>
          %219 = vector.extract %215[0] : f16 from vector<1xf16>
          %220 = vector.extract %216[0] : f16 from vector<1xf16>
          %221 = vector.extract %217[0] : f16 from vector<1xf16>
          %222 = vector.from_elements %218, %219, %220, %221 : vector<4xf16>
          %223 = vector.load %reinterpret_cast_22[%c0] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
          %224 = vector.load %reinterpret_cast_22[%c68] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
          %225 = vector.load %reinterpret_cast_22[%c136] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
          %226 = vector.load %reinterpret_cast_22[%c204] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
          %227 = vector.extract %223[0] : f16 from vector<1xf16>
          %228 = vector.extract %224[0] : f16 from vector<1xf16>
          %229 = vector.extract %225[0] : f16 from vector<1xf16>
          %230 = vector.extract %226[0] : f16 from vector<1xf16>
          %231 = vector.from_elements %227, %228, %229, %230 : vector<4xf16>
          %232 = vector.load %reinterpret_cast_23[%c0] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
          %233 = vector.load %reinterpret_cast_23[%c68] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
          %234 = vector.load %reinterpret_cast_23[%c136] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
          %235 = vector.load %reinterpret_cast_23[%c204] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
          %236 = vector.extract %232[0] : f16 from vector<1xf16>
          %237 = vector.extract %233[0] : f16 from vector<1xf16>
          %238 = vector.extract %234[0] : f16 from vector<1xf16>
          %239 = vector.extract %235[0] : f16 from vector<1xf16>
          %240 = vector.from_elements %236, %237, %238, %239 : vector<4xf16>
          %241 = vector.load %reinterpret_cast_24[%c0] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
          %242 = vector.load %reinterpret_cast_24[%c68] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
          %243 = vector.load %reinterpret_cast_24[%c136] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
          %244 = vector.load %reinterpret_cast_24[%c204] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
          %245 = vector.extract %241[0] : f16 from vector<1xf16>
          %246 = vector.extract %242[0] : f16 from vector<1xf16>
          %247 = vector.extract %243[0] : f16 from vector<1xf16>
          %248 = vector.extract %244[0] : f16 from vector<1xf16>
          %249 = vector.from_elements %245, %246, %247, %248 : vector<4xf16>
          %250 = vector.load %reinterpret_cast_25[%c0] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
          %251 = vector.load %reinterpret_cast_25[%c68] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
          %252 = vector.load %reinterpret_cast_25[%c136] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
          %253 = vector.load %reinterpret_cast_25[%c204] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
          %254 = vector.extract %250[0] : f16 from vector<1xf16>
          %255 = vector.extract %251[0] : f16 from vector<1xf16>
          %256 = vector.extract %252[0] : f16 from vector<1xf16>
          %257 = vector.extract %253[0] : f16 from vector<1xf16>
          %258 = vector.from_elements %254, %255, %256, %257 : vector<4xf16>
          %259 = vector.load %reinterpret_cast_26[%c0] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
          %260 = vector.load %reinterpret_cast_26[%c68] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
          %261 = vector.load %reinterpret_cast_26[%c136] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
          %262 = vector.load %reinterpret_cast_26[%c204] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
          %263 = vector.extract %259[0] : f16 from vector<1xf16>
          %264 = vector.extract %260[0] : f16 from vector<1xf16>
          %265 = vector.extract %261[0] : f16 from vector<1xf16>
          %266 = vector.extract %262[0] : f16 from vector<1xf16>
          %267 = vector.from_elements %263, %264, %265, %266 : vector<4xf16>
          %268 = vector.load %reinterpret_cast_27[%c0] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
          %269 = vector.load %reinterpret_cast_27[%c68] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
          %270 = vector.load %reinterpret_cast_27[%c136] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
          %271 = vector.load %reinterpret_cast_27[%c204] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
          %272 = vector.extract %268[0] : f16 from vector<1xf16>
          %273 = vector.extract %269[0] : f16 from vector<1xf16>
          %274 = vector.extract %270[0] : f16 from vector<1xf16>
          %275 = vector.extract %271[0] : f16 from vector<1xf16>
          %276 = vector.from_elements %272, %273, %274, %275 : vector<4xf16>
          %277 = vector.load %reinterpret_cast_28[%c0] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
          %278 = vector.load %reinterpret_cast_28[%c68] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
          %279 = vector.load %reinterpret_cast_28[%c136] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
          %280 = vector.load %reinterpret_cast_28[%c204] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
          %281 = vector.extract %277[0] : f16 from vector<1xf16>
          %282 = vector.extract %278[0] : f16 from vector<1xf16>
          %283 = vector.extract %279[0] : f16 from vector<1xf16>
          %284 = vector.extract %280[0] : f16 from vector<1xf16>
          %285 = vector.from_elements %281, %282, %283, %284 : vector<4xf16>
          %286 = vector.load %reinterpret_cast_29[%c0] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
          %287 = vector.load %reinterpret_cast_29[%c68] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
          %288 = vector.load %reinterpret_cast_29[%c136] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
          %289 = vector.load %reinterpret_cast_29[%c204] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
          %290 = vector.extract %286[0] : f16 from vector<1xf16>
          %291 = vector.extract %287[0] : f16 from vector<1xf16>
          %292 = vector.extract %288[0] : f16 from vector<1xf16>
          %293 = vector.extract %289[0] : f16 from vector<1xf16>
          %294 = vector.from_elements %290, %291, %292, %293 : vector<4xf16>
          %295 = vector.load %reinterpret_cast_30[%c0] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
          %296 = vector.load %reinterpret_cast_30[%c68] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
          %297 = vector.load %reinterpret_cast_30[%c136] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
          %298 = vector.load %reinterpret_cast_30[%c204] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
          %299 = vector.extract %295[0] : f16 from vector<1xf16>
          %300 = vector.extract %296[0] : f16 from vector<1xf16>
          %301 = vector.extract %297[0] : f16 from vector<1xf16>
          %302 = vector.extract %298[0] : f16 from vector<1xf16>
          %303 = vector.from_elements %299, %300, %301, %302 : vector<4xf16>
          %304 = vector.load %reinterpret_cast_31[%c0] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
          %305 = vector.load %reinterpret_cast_31[%c68] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
          %306 = vector.load %reinterpret_cast_31[%c136] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
          %307 = vector.load %reinterpret_cast_31[%c204] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
          %308 = vector.extract %304[0] : f16 from vector<1xf16>
          %309 = vector.extract %305[0] : f16 from vector<1xf16>
          %310 = vector.extract %306[0] : f16 from vector<1xf16>
          %311 = vector.extract %307[0] : f16 from vector<1xf16>
          %312 = vector.from_elements %308, %309, %310, %311 : vector<4xf16>
          %313 = vector.load %alloc_16[%c0, %c0, %4, %10] : memref<1x1x64x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %314 = vector.load %alloc_16[%c0, %c0, %4, %12] : memref<1x1x64x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %315 = vector.load %alloc_16[%c0, %c0, %4, %14] : memref<1x1x64x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %316 = vector.load %alloc_16[%c0, %c0, %4, %16] : memref<1x1x64x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %317 = vector.load %alloc_16[%c0, %c0, %4, %18] : memref<1x1x64x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %318 = vector.load %alloc_16[%c0, %c0, %4, %20] : memref<1x1x64x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %319 = vector.load %alloc_16[%c0, %c0, %4, %22] : memref<1x1x64x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %320 = vector.load %alloc_16[%c0, %c0, %4, %24] : memref<1x1x64x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %321 = vector.load %alloc_16[%c0, %c0, %4, %26] : memref<1x1x64x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %322 = vector.load %alloc_16[%c0, %c0, %4, %28] : memref<1x1x64x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %323 = vector.load %alloc_16[%c0, %c0, %4, %30] : memref<1x1x64x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %324 = vector.load %alloc_16[%c0, %c0, %4, %32] : memref<1x1x64x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %325 = vector.load %alloc_16[%c0, %c0, %4, %34] : memref<1x1x64x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %326 = vector.load %alloc_16[%c0, %c0, %4, %36] : memref<1x1x64x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %327 = vector.load %alloc_16[%c0, %c0, %4, %38] : memref<1x1x64x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %328 = vector.load %alloc_16[%c0, %c0, %4, %40] : memref<1x1x64x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %329 = vector.load %alloc_16[%c0, %c0, %113, %10] : memref<1x1x64x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %330 = vector.load %alloc_16[%c0, %c0, %113, %12] : memref<1x1x64x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %331 = vector.load %alloc_16[%c0, %c0, %113, %14] : memref<1x1x64x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %332 = vector.load %alloc_16[%c0, %c0, %113, %16] : memref<1x1x64x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %333 = vector.load %alloc_16[%c0, %c0, %113, %18] : memref<1x1x64x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %334 = vector.load %alloc_16[%c0, %c0, %113, %20] : memref<1x1x64x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %335 = vector.load %alloc_16[%c0, %c0, %113, %22] : memref<1x1x64x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %336 = vector.load %alloc_16[%c0, %c0, %113, %24] : memref<1x1x64x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %337 = vector.load %alloc_16[%c0, %c0, %113, %26] : memref<1x1x64x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %338 = vector.load %alloc_16[%c0, %c0, %113, %28] : memref<1x1x64x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %339 = vector.load %alloc_16[%c0, %c0, %113, %30] : memref<1x1x64x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %340 = vector.load %alloc_16[%c0, %c0, %113, %32] : memref<1x1x64x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %341 = vector.load %alloc_16[%c0, %c0, %113, %34] : memref<1x1x64x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %342 = vector.load %alloc_16[%c0, %c0, %113, %36] : memref<1x1x64x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %343 = vector.load %alloc_16[%c0, %c0, %113, %38] : memref<1x1x64x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %344 = vector.load %alloc_16[%c0, %c0, %113, %40] : memref<1x1x64x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %345 = amdgpu.mfma %313 * %44 + %cst_14 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %346 = amdgpu.mfma %314 * %45 + %345 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %347 = amdgpu.mfma %315 * %46 + %346 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %348 = amdgpu.mfma %316 * %47 + %347 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %349 = amdgpu.mfma %317 * %48 + %348 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %350 = amdgpu.mfma %318 * %49 + %349 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %351 = amdgpu.mfma %319 * %50 + %350 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %352 = amdgpu.mfma %320 * %51 + %351 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %353 = amdgpu.mfma %321 * %52 + %352 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %354 = amdgpu.mfma %322 * %53 + %353 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %355 = amdgpu.mfma %323 * %54 + %354 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %356 = amdgpu.mfma %324 * %55 + %355 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %357 = amdgpu.mfma %325 * %56 + %356 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %358 = amdgpu.mfma %326 * %57 + %357 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %359 = amdgpu.mfma %327 * %58 + %358 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %360 = amdgpu.mfma %328 * %59 + %359 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %361 = amdgpu.mfma %329 * %44 + %cst_14 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %362 = amdgpu.mfma %330 * %45 + %361 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %363 = amdgpu.mfma %331 * %46 + %362 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %364 = amdgpu.mfma %332 * %47 + %363 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %365 = amdgpu.mfma %333 * %48 + %364 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %366 = amdgpu.mfma %334 * %49 + %365 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %367 = amdgpu.mfma %335 * %50 + %366 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %368 = amdgpu.mfma %336 * %51 + %367 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %369 = amdgpu.mfma %337 * %52 + %368 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %370 = amdgpu.mfma %338 * %53 + %369 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %371 = amdgpu.mfma %339 * %54 + %370 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %372 = amdgpu.mfma %340 * %55 + %371 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %373 = amdgpu.mfma %341 * %56 + %372 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %374 = amdgpu.mfma %342 * %57 + %373 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %375 = amdgpu.mfma %343 * %58 + %374 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %376 = amdgpu.mfma %344 * %59 + %375 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %377 = arith.addi %144, %10 overflow<nsw, nuw> : index
          %378 = vector.splat %377 : vector<4xindex>
          %379 = arith.addi %378, %cst_10 overflow<nsw, nuw> : vector<4xindex>
          %380 = arith.index_cast %379 : vector<4xindex> to vector<4xi32>
          %381 = arith.addi %379, %cst_6 overflow<nsw, nuw> : vector<4xindex>
          %382 = arith.index_cast %381 : vector<4xindex> to vector<4xi32>
          %383 = arith.addi %379, %cst_5 overflow<nsw, nuw> : vector<4xindex>
          %384 = arith.index_cast %383 : vector<4xindex> to vector<4xi32>
          %385 = arith.addi %379, %cst_4 overflow<nsw, nuw> : vector<4xindex>
          %386 = arith.index_cast %385 : vector<4xindex> to vector<4xi32>
          %387 = vector.insert_strided_slice %380, %cst_9 {offsets = [0], strides = [1]} : vector<4xi32> into vector<16xi32>
          %388 = vector.insert_strided_slice %382, %387 {offsets = [4], strides = [1]} : vector<4xi32> into vector<16xi32>
          %389 = vector.insert_strided_slice %384, %388 {offsets = [8], strides = [1]} : vector<4xi32> into vector<16xi32>
          %390 = vector.insert_strided_slice %386, %389 {offsets = [12], strides = [1]} : vector<4xi32> into vector<16xi32>
          %391 = arith.addi %379, %cst_3 overflow<nsw, nuw> : vector<4xindex>
          %392 = arith.index_cast %391 : vector<4xindex> to vector<4xi32>
          %393 = arith.addi %379, %cst_2 overflow<nsw, nuw> : vector<4xindex>
          %394 = arith.index_cast %393 : vector<4xindex> to vector<4xi32>
          %395 = arith.addi %379, %cst_1 overflow<nsw, nuw> : vector<4xindex>
          %396 = arith.index_cast %395 : vector<4xindex> to vector<4xi32>
          %397 = arith.addi %379, %cst_0 overflow<nsw, nuw> : vector<4xindex>
          %398 = arith.index_cast %397 : vector<4xindex> to vector<4xi32>
          %399 = vector.insert_strided_slice %392, %cst_9 {offsets = [0], strides = [1]} : vector<4xi32> into vector<16xi32>
          %400 = vector.insert_strided_slice %394, %399 {offsets = [4], strides = [1]} : vector<4xi32> into vector<16xi32>
          %401 = vector.insert_strided_slice %396, %400 {offsets = [8], strides = [1]} : vector<4xi32> into vector<16xi32>
          %402 = vector.insert_strided_slice %398, %401 {offsets = [12], strides = [1]} : vector<4xi32> into vector<16xi32>
          %403 = arith.index_cast %390 : vector<16xi32> to vector<16xindex>
          %404 = arith.cmpi slt, %403, %cst : vector<16xindex>
          %405 = arith.cmpi sge, %117, %390 : vector<16xi32>
          %406 = arith.cmpi sge, %117, %402 : vector<16xi32>
          %407 = arith.andi %405, %404 : vector<16xi1>
          %408 = arith.andi %406, %404 : vector<16xi1>
          %409 = arith.select %407, %cst_14, %cst_11 : vector<16xi1>, vector<16xf32>
          %410 = arith.select %408, %cst_14, %cst_11 : vector<16xi1>, vector<16xf32>
          %411 = arith.addf %360, %409 : vector<16xf32>
          %412 = arith.addf %376, %410 : vector<16xf32>
          %413 = vector.extract_strided_slice %411 {offsets = [0], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
          %414 = vector.extract_strided_slice %411 {offsets = [1], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
          %415 = arith.maximumf %413, %414 : vector<1xf32>
          %416 = vector.extract_strided_slice %411 {offsets = [2], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
          %417 = arith.maximumf %415, %416 : vector<1xf32>
          %418 = vector.extract_strided_slice %411 {offsets = [3], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
          %419 = arith.maximumf %417, %418 : vector<1xf32>
          %420 = vector.extract_strided_slice %411 {offsets = [4], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
          %421 = arith.maximumf %419, %420 : vector<1xf32>
          %422 = vector.extract_strided_slice %411 {offsets = [5], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
          %423 = arith.maximumf %421, %422 : vector<1xf32>
          %424 = vector.extract_strided_slice %411 {offsets = [6], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
          %425 = arith.maximumf %423, %424 : vector<1xf32>
          %426 = vector.extract_strided_slice %411 {offsets = [7], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
          %427 = arith.maximumf %425, %426 : vector<1xf32>
          %428 = vector.extract_strided_slice %411 {offsets = [8], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
          %429 = arith.maximumf %427, %428 : vector<1xf32>
          %430 = vector.extract_strided_slice %411 {offsets = [9], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
          %431 = arith.maximumf %429, %430 : vector<1xf32>
          %432 = vector.extract_strided_slice %411 {offsets = [10], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
          %433 = arith.maximumf %431, %432 : vector<1xf32>
          %434 = vector.extract_strided_slice %411 {offsets = [11], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
          %435 = arith.maximumf %433, %434 : vector<1xf32>
          %436 = vector.extract_strided_slice %411 {offsets = [12], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
          %437 = arith.maximumf %435, %436 : vector<1xf32>
          %438 = vector.extract_strided_slice %411 {offsets = [13], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
          %439 = arith.maximumf %437, %438 : vector<1xf32>
          %440 = vector.extract_strided_slice %411 {offsets = [14], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
          %441 = arith.maximumf %439, %440 : vector<1xf32>
          %442 = vector.extract_strided_slice %411 {offsets = [15], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
          %443 = arith.maximumf %441, %442 : vector<1xf32>
          %444 = vector.extract_strided_slice %412 {offsets = [0], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
          %445 = vector.extract_strided_slice %412 {offsets = [1], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
          %446 = arith.maximumf %444, %445 : vector<1xf32>
          %447 = vector.extract_strided_slice %412 {offsets = [2], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
          %448 = arith.maximumf %446, %447 : vector<1xf32>
          %449 = vector.extract_strided_slice %412 {offsets = [3], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
          %450 = arith.maximumf %448, %449 : vector<1xf32>
          %451 = vector.extract_strided_slice %412 {offsets = [4], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
          %452 = arith.maximumf %450, %451 : vector<1xf32>
          %453 = vector.extract_strided_slice %412 {offsets = [5], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
          %454 = arith.maximumf %452, %453 : vector<1xf32>
          %455 = vector.extract_strided_slice %412 {offsets = [6], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
          %456 = arith.maximumf %454, %455 : vector<1xf32>
          %457 = vector.extract_strided_slice %412 {offsets = [7], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
          %458 = arith.maximumf %456, %457 : vector<1xf32>
          %459 = vector.extract_strided_slice %412 {offsets = [8], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
          %460 = arith.maximumf %458, %459 : vector<1xf32>
          %461 = vector.extract_strided_slice %412 {offsets = [9], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
          %462 = arith.maximumf %460, %461 : vector<1xf32>
          %463 = vector.extract_strided_slice %412 {offsets = [10], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
          %464 = arith.maximumf %462, %463 : vector<1xf32>
          %465 = vector.extract_strided_slice %412 {offsets = [11], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
          %466 = arith.maximumf %464, %465 : vector<1xf32>
          %467 = vector.extract_strided_slice %412 {offsets = [12], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
          %468 = arith.maximumf %466, %467 : vector<1xf32>
          %469 = vector.extract_strided_slice %412 {offsets = [13], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
          %470 = arith.maximumf %468, %469 : vector<1xf32>
          %471 = vector.extract_strided_slice %412 {offsets = [14], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
          %472 = arith.maximumf %470, %471 : vector<1xf32>
          %473 = vector.extract_strided_slice %412 {offsets = [15], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
          %474 = arith.maximumf %472, %473 : vector<1xf32>
          %475 = arith.maximumf %443, %474 : vector<1xf32>
          %476 = vector.extract %475[0] : f32 from vector<1xf32>
          %shuffleResult, %valid = gpu.shuffle  xor %476, %c32_i32, %c64_i32 : f32
          %477 = vector.broadcast %shuffleResult : f32 to vector<1xf32>
          %478 = arith.maximumf %475, %477 : vector<1xf32>
          %479 = arith.maximumf %arg5, %478 : vector<1xf32>
          %480 = arith.subf %arg5, %479 : vector<1xf32>
          %481 = math.exp2 %480 : vector<1xf32>
          %482 = vector.extract %479[0] : f32 from vector<1xf32>
          %483 = vector.splat %482 : vector<16xf32>
          %484 = arith.subf %411, %483 : vector<16xf32>
          %485 = arith.subf %412, %483 : vector<16xf32>
          %486 = math.exp2 %484 : vector<16xf32>
          %487 = math.exp2 %485 : vector<16xf32>
          %488 = arith.mulf %arg6, %481 : vector<1xf32>
          %489 = arith.addf %486, %487 : vector<16xf32>
          %490 = vector.extract_strided_slice %489 {offsets = [0], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
          %491 = vector.extract_strided_slice %489 {offsets = [1], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
          %492 = arith.addf %490, %491 : vector<1xf32>
          %493 = vector.extract_strided_slice %489 {offsets = [2], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
          %494 = arith.addf %492, %493 : vector<1xf32>
          %495 = vector.extract_strided_slice %489 {offsets = [3], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
          %496 = arith.addf %494, %495 : vector<1xf32>
          %497 = vector.extract_strided_slice %489 {offsets = [4], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
          %498 = arith.addf %496, %497 : vector<1xf32>
          %499 = vector.extract_strided_slice %489 {offsets = [5], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
          %500 = arith.addf %498, %499 : vector<1xf32>
          %501 = vector.extract_strided_slice %489 {offsets = [6], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
          %502 = arith.addf %500, %501 : vector<1xf32>
          %503 = vector.extract_strided_slice %489 {offsets = [7], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
          %504 = arith.addf %502, %503 : vector<1xf32>
          %505 = vector.extract_strided_slice %489 {offsets = [8], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
          %506 = arith.addf %504, %505 : vector<1xf32>
          %507 = vector.extract_strided_slice %489 {offsets = [9], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
          %508 = arith.addf %506, %507 : vector<1xf32>
          %509 = vector.extract_strided_slice %489 {offsets = [10], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
          %510 = arith.addf %508, %509 : vector<1xf32>
          %511 = vector.extract_strided_slice %489 {offsets = [11], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
          %512 = arith.addf %510, %511 : vector<1xf32>
          %513 = vector.extract_strided_slice %489 {offsets = [12], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
          %514 = arith.addf %512, %513 : vector<1xf32>
          %515 = vector.extract_strided_slice %489 {offsets = [13], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
          %516 = arith.addf %514, %515 : vector<1xf32>
          %517 = vector.extract_strided_slice %489 {offsets = [14], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
          %518 = arith.addf %516, %517 : vector<1xf32>
          %519 = vector.extract_strided_slice %489 {offsets = [15], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
          %520 = arith.addf %518, %519 : vector<1xf32>
          %521 = vector.extract %520[0] : f32 from vector<1xf32>
          %shuffleResult_32, %valid_33 = gpu.shuffle  xor %521, %c32_i32, %c64_i32 : f32
          %522 = vector.broadcast %shuffleResult_32 : f32 to vector<1xf32>
          %523 = arith.addf %520, %522 : vector<1xf32>
          %524 = arith.addf %488, %523 : vector<1xf32>
          %525 = arith.truncf %486 : vector<16xf32> to vector<16xf16>
          %526 = arith.truncf %487 : vector<16xf32> to vector<16xf16>
          %527 = vector.extract %481[0] : f32 from vector<1xf32>
          %528 = vector.splat %527 : vector<16xf32>
          %529 = arith.mulf %arg7, %528 : vector<16xf32>
          %530 = arith.mulf %arg8, %528 : vector<16xf32>
          %531 = vector.extract_strided_slice %525 {offsets = [0], sizes = [4], strides = [1]} : vector<16xf16> to vector<4xf16>
          %532 = vector.extract_strided_slice %525 {offsets = [4], sizes = [4], strides = [1]} : vector<16xf16> to vector<4xf16>
          %533 = vector.extract_strided_slice %525 {offsets = [8], sizes = [4], strides = [1]} : vector<16xf16> to vector<4xf16>
          %534 = vector.extract_strided_slice %525 {offsets = [12], sizes = [4], strides = [1]} : vector<16xf16> to vector<4xf16>
          %535 = vector.extract_strided_slice %526 {offsets = [0], sizes = [4], strides = [1]} : vector<16xf16> to vector<4xf16>
          %536 = vector.extract_strided_slice %526 {offsets = [4], sizes = [4], strides = [1]} : vector<16xf16> to vector<4xf16>
          %537 = vector.extract_strided_slice %526 {offsets = [8], sizes = [4], strides = [1]} : vector<16xf16> to vector<4xf16>
          %538 = vector.extract_strided_slice %526 {offsets = [12], sizes = [4], strides = [1]} : vector<16xf16> to vector<4xf16>
          %539 = amdgpu.mfma %177 * %531 + %529 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %540 = amdgpu.mfma %186 * %532 + %539 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %541 = amdgpu.mfma %195 * %533 + %540 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %542 = amdgpu.mfma %204 * %534 + %541 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %543 = amdgpu.mfma %213 * %535 + %542 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %544 = amdgpu.mfma %222 * %536 + %543 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %545 = amdgpu.mfma %231 * %537 + %544 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %546 = amdgpu.mfma %240 * %538 + %545 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %547 = amdgpu.mfma %249 * %531 + %530 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %548 = amdgpu.mfma %258 * %532 + %547 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %549 = amdgpu.mfma %267 * %533 + %548 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %550 = amdgpu.mfma %276 * %534 + %549 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %551 = amdgpu.mfma %285 * %535 + %550 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %552 = amdgpu.mfma %294 * %536 + %551 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %553 = amdgpu.mfma %303 * %537 + %552 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %554 = amdgpu.mfma %312 * %538 + %553 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          scf.yield %479, %524, %546, %554 : vector<1xf32>, vector<1xf32>, vector<16xf32>, vector<16xf32>
        }
        %119 = arith.divf %cst_8, %118#1 : vector<1xf32>
        %120 = vector.extract %119[0] : f32 from vector<1xf32>
        %121 = vector.splat %120 : vector<16xf32>
        %122 = arith.mulf %118#2, %121 : vector<16xf32>
        %123 = arith.mulf %118#3, %121 : vector<16xf32>
        %124 = vector.extract_strided_slice %122 {offsets = [0], sizes = [4], strides = [1]} : vector<16xf32> to vector<4xf32>
        %125 = stream.binding.subspan %arg3[%c0] : !stream.binding -> memref<1x1024x32x128xf32, strided<[4194304, 4096, 128, 1], offset: ?>>
        %126 = arith.muli %thread_id_y, %c64 overflow<nsw, nuw> : index
        %127 = arith.muli %workgroup_id_1, %c64 overflow<nsw, nuw> : index
        %128 = arith.addi %127, %126 overflow<nsw, nuw> : index
        %129 = arith.addi %128, %10 overflow<nsw, nuw> : index
        vector.store %124, %125[%c0, %6, %7, %129] : memref<1x1024x32x128xf32, strided<[4194304, 4096, 128, 1], offset: ?>>, vector<4xf32>
        %130 = vector.extract_strided_slice %122 {offsets = [4], sizes = [4], strides = [1]} : vector<16xf32> to vector<4xf32>
        %131 = arith.addi %129, %c8 overflow<nsw, nuw> : index
        vector.store %130, %125[%c0, %6, %7, %131] : memref<1x1024x32x128xf32, strided<[4194304, 4096, 128, 1], offset: ?>>, vector<4xf32>
        %132 = vector.extract_strided_slice %122 {offsets = [8], sizes = [4], strides = [1]} : vector<16xf32> to vector<4xf32>
        %133 = arith.addi %129, %c16 overflow<nsw, nuw> : index
        vector.store %132, %125[%c0, %6, %7, %133] : memref<1x1024x32x128xf32, strided<[4194304, 4096, 128, 1], offset: ?>>, vector<4xf32>
        %134 = vector.extract_strided_slice %122 {offsets = [12], sizes = [4], strides = [1]} : vector<16xf32> to vector<4xf32>
        %135 = arith.addi %129, %c24 overflow<nsw, nuw> : index
        vector.store %134, %125[%c0, %6, %7, %135] : memref<1x1024x32x128xf32, strided<[4194304, 4096, 128, 1], offset: ?>>, vector<4xf32>
        %136 = vector.extract_strided_slice %123 {offsets = [0], sizes = [4], strides = [1]} : vector<16xf32> to vector<4xf32>
        %137 = arith.addi %129, %c32 overflow<nsw, nuw> : index
        vector.store %136, %125[%c0, %6, %7, %137] : memref<1x1024x32x128xf32, strided<[4194304, 4096, 128, 1], offset: ?>>, vector<4xf32>
        %138 = vector.extract_strided_slice %123 {offsets = [4], sizes = [4], strides = [1]} : vector<16xf32> to vector<4xf32>
        %139 = arith.addi %129, %c40 overflow<nsw, nuw> : index
        vector.store %138, %125[%c0, %6, %7, %139] : memref<1x1024x32x128xf32, strided<[4194304, 4096, 128, 1], offset: ?>>, vector<4xf32>
        %140 = vector.extract_strided_slice %123 {offsets = [8], sizes = [4], strides = [1]} : vector<16xf32> to vector<4xf32>
        %141 = arith.addi %129, %c48 overflow<nsw, nuw> : index
        vector.store %140, %125[%c0, %6, %7, %141] : memref<1x1024x32x128xf32, strided<[4194304, 4096, 128, 1], offset: ?>>, vector<4xf32>
        %142 = vector.extract_strided_slice %123 {offsets = [12], sizes = [4], strides = [1]} : vector<16xf32> to vector<4xf32>
        %143 = arith.addi %129, %c56 overflow<nsw, nuw> : index
        vector.store %142, %125[%c0, %6, %7, %143] : memref<1x1024x32x128xf32, strided<[4194304, 4096, 128, 1], offset: ?>>, vector<4xf32>
        return
      }
    }
  }
  func.func @isolated_benchmark(%arg0: tensor<1x1024x32x128xf16>, %arg1: tensor<1x1357x32x128xf16>, %arg2: tensor<1x1357x32x128xf16>, %arg3: tensor<1x1024x32x128xf32>) -> tensor<1x1024x32x128xf32> {
    %0 = flow.dispatch @base_attention::@base_attention(%arg0, %arg1, %arg2, %arg3) : (tensor<1x1024x32x128xf16>, tensor<1x1357x32x128xf16>, tensor<1x1357x32x128xf16>, tensor<1x1024x32x128xf32>) -> %arg3
    return %0 : tensor<1x1024x32x128xf32>
  }
}
