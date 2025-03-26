#translation = #iree_codegen.translation_info<pipeline = None workgroup_size = [256, 1, 1] subgroup_size = 64, {llvm_func_attrs = {"amdgpu-waves-per-eu" = "2", "denormal-fp-math-f32" = "preserve-sign"}}>
module attributes {transform.with_named_sequence} {
  stream.executable private @base_attention_custom_mask {
    stream.executable.export public @base_attention_custom_mask workgroups() -> (index, index, index) {
      %c1 = arith.constant 1 : index
      stream.return %c1, %c1, %c1 : index, index, index
    }
    builtin.module {
      func.func @base_attention_custom_mask(%arg0: !stream.binding, %arg1: !stream.binding, %arg2: !stream.binding, %arg3: !stream.binding, %arg4: !stream.binding, %arg5: !stream.binding) attributes {translation_info = #translation} {
        %cst = arith.constant dense<2> : vector<4xindex>
        %cst_0 = arith.constant dense<-1.000000e+06> : vector<4xf32>
        %c204 = arith.constant 204 : index
        %c136 = arith.constant 136 : index
        %c272 = arith.constant 272 : index
        %cst_1 = arith.constant dense<2> : vector<8xindex>
        %cst_2 = arith.constant dense<0.000000e+00> : vector<8xf16>
        %c27 = arith.constant 27 : index
        %c72 = arith.constant 72 : index
        %c6144 = arith.constant 6144 : index
        %cst_3 = arith.constant dense<0.000000e+00> : vector<4xf16>
        %cst_4 = arith.constant dense<1.000000e+00> : vector<1xf32>
        %c64_i32 = arith.constant 64 : i32
        %c32_i32 = arith.constant 32 : i32
        %cst_5 = arith.constant dense<[0, 1, 2, 3]> : vector<4xindex>
        %c56 = arith.constant 56 : index
        %c48 = arith.constant 48 : index
        %c40 = arith.constant 40 : index
        %c24 = arith.constant 24 : index
        %c16 = arith.constant 16 : index
        %c1073741822 = arith.constant 1073741822 : index
        %c68 = arith.constant 68 : index
        %cst_6 = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7]> : vector<8xindex>
        %c2 = arith.constant 2 : index
        %c2048 = arith.constant 2048 : index
        %c3 = arith.constant 3 : index
        %c8 = arith.constant 8 : index
        %c4 = arith.constant 4 : index
        %c128 = arith.constant 128 : index
        %c64 = arith.constant 64 : index
        %c1 = arith.constant 1 : index
        %c32 = arith.constant 32 : index
        %c0 = arith.constant 0 : index
        %cst_7 = arith.constant dense<-1.000000e+06> : vector<16xf32>
        %cst_8 = arith.constant dense<-1.000000e+06> : vector<1xf32>
        %cst_9 = arith.constant dense<0.000000e+00> : vector<1xf32>
        %cst_10 = arith.constant dense<0.000000e+00> : vector<16xf32>
        %cst_11 = arith.constant dense<8.330080e-01> : vector<4xf16>
        %workgroup_id_0 = stream.dispatch.workgroup.id[0] : index
        %workgroup_id_1 = stream.dispatch.workgroup.id[1] : index
        %thread_id_x = gpu.thread_id  x
        %thread_id_y = gpu.thread_id  y
        %thread_id_z = gpu.thread_id  z
        %alloc = memref.alloc() : memref<1x64x1x68xf16, #gpu.address_space<workgroup>>
        %alloc_12 = memref.alloc() : memref<1x1x64x7xf16, #gpu.address_space<workgroup>>
        %0 = stream.binding.subspan %arg0[%c0] : !stream.binding -> memref<1x1x1x3xf16, strided<[3, 3, 3, 1], offset: ?>>
        %1 = arith.divsi %thread_id_x, %c64 : index
        %2 = arith.muli %1, %c32 overflow<nsw, nuw> : index
        %3 = arith.muli %workgroup_id_0, %c128 overflow<nsw, nuw> : index
        %4 = arith.remsi %thread_id_x, %c32 : index
        %5 = arith.addi %4, %3 overflow<nsw, nuw> : index
        %6 = arith.addi %5, %2 overflow<nsw, nuw> : index
        %7 = arith.cmpi slt, %6, %c1 : index
        %8 = vector.splat %7 : vector<4xi1>
        %9 = arith.remsi %thread_id_x, %c64 : index
        %10 = arith.divsi %9, %c32 : index
        %11 = arith.muli %10, %c4 overflow<nsw, nuw> : index
        %12 = vector.maskedload %0[%c0, %6, %c0, %11], %8, %cst_3 : memref<1x1x1x3xf16, strided<[3, 3, 3, 1], offset: ?>>, vector<4xi1>, vector<4xf16> into vector<4xf16>
        %13 = stream.binding.subspan %arg2[%c0] : !stream.binding -> memref<1x2x1x2xf16, strided<[4, 2, 2, 1], offset: ?>>
        %14 = stream.binding.subspan %arg1[%c0] : !stream.binding -> memref<1x2x1x3xf16, strided<[6, 3, 3, 1], offset: ?>>
        %15 = stream.binding.subspan %arg5[%c0] : !stream.binding -> memref<1x1x2xf32, strided<[2, 2, 1], offset: ?>>
        amdgpu.lds_barrier
        %16 = arith.mulf %12, %cst_11 : vector<4xf16>
        %17 = arith.muli %thread_id_z, %c6144 overflow<nsw, nuw> : index
        %18 = arith.muli %thread_id_y, %c6144 overflow<nsw, nuw> : index
        %19 = arith.addi %17, %18 overflow<nsw, nuw> : index
        %20 = arith.muli %19, %c3 overflow<nsw, nuw> : index
        %21 = arith.muli %thread_id_x, %c72 overflow<nsw, nuw> : index
        %22 = arith.addi %20, %21 overflow<nsw, nuw> : index
        %23 = arith.divsi %22, %c27 : index
        %24 = arith.remsi %23, %c64 : index
        %25 = arith.cmpi slt, %24, %c2 : index
        %26 = vector.splat %25 : vector<8xi1>
        %27 = arith.muli %thread_id_x, %c8 overflow<nsw, nuw> : index
        %28 = arith.muli %thread_id_y, %c2048 overflow<nsw, nuw> : index
        %29 = arith.muli %thread_id_z, %c2048 overflow<nsw, nuw> : index
        %30 = arith.addi %29, %28 overflow<nsw, nuw> : index
        %31 = arith.addi %30, %27 overflow<nsw, nuw> : index
        %32 = arith.remsi %31, %c3 : index
        %33 = vector.maskedload %14[%c0, %24, %c0, %32], %26, %cst_2 : memref<1x2x1x3xf16, strided<[6, 3, 3, 1], offset: ?>>, vector<8xi1>, vector<8xf16> into vector<8xf16>
        vector.store %33, %alloc_12[%c0, %c0, %24, %32] : memref<1x1x64x7xf16, #gpu.address_space<workgroup>>, vector<8xf16>
        %34 = arith.muli %thread_id_y, %c32 overflow<nsw, nuw> : index
        %35 = arith.muli %thread_id_z, %c32 overflow<nsw, nuw> : index
        %36 = arith.divsi %thread_id_x, %c8 : index
        %37 = arith.addi %36, %35 overflow<nsw, nuw> : index
        %38 = arith.addi %37, %34 overflow<nsw, nuw> : index
        %39 = arith.remsi %38, %c64 : index
        %40 = arith.remsi %thread_id_x, %c8 : index
        %41 = arith.muli %40, %c8 overflow<nsw, nuw> : index
        %42 = arith.muli %workgroup_id_1, %c64 overflow<nsw, nuw> : index
        %43 = arith.addi %42, %41 overflow<nsw, nuw> : index
        %44 = vector.splat %43 : vector<8xindex>
        %45 = arith.addi %44, %cst_6 overflow<nsw, nuw> : vector<8xindex>
        %46 = arith.cmpi slt, %45, %cst_1 : vector<8xindex>
        %47 = arith.cmpi slt, %39, %c2 : index
        %48 = vector.splat %47 : vector<8xi1>
        %49 = arith.andi %46, %48 : vector<8xi1>
        %50 = vector.maskedload %13[%c0, %39, %c0, %43], %49, %cst_2 : memref<1x2x1x2xf16, strided<[4, 2, 2, 1], offset: ?>>, vector<8xi1>, vector<8xf16> into vector<8xf16>
        vector.store %50, %alloc[%c0, %39, %c0, %41] : memref<1x64x1x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
        %51 = arith.addi %38, %c32 overflow<nsw, nuw> : index
        %52 = arith.remsi %51, %c64 : index
        %53 = arith.cmpi slt, %52, %c2 : index
        %54 = vector.splat %53 : vector<8xi1>
        %55 = arith.andi %46, %54 : vector<8xi1>
        %56 = vector.maskedload %13[%c0, %52, %c0, %43], %55, %cst_2 : memref<1x2x1x2xf16, strided<[4, 2, 2, 1], offset: ?>>, vector<8xi1>, vector<8xf16> into vector<8xf16>
        vector.store %56, %alloc[%c0, %52, %c0, %41] : memref<1x64x1x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
        amdgpu.lds_barrier
        %57 = arith.muli %thread_id_y, %c64 overflow<nsw, nuw> : index
        %58 = arith.addi %4, %57 overflow<nsw, nuw> : index
        %59 = arith.muli %10, %c272 overflow<nsw> : index
        %60 = arith.addi %59, %58 overflow<nsw> : index
        %reinterpret_cast = memref.reinterpret_cast %alloc to offset: [%60], sizes: [%c1073741822], strides: [1] : memref<1x64x1x68xf16, #gpu.address_space<workgroup>> to memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>
        %61 = vector.load %reinterpret_cast[%c0] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
        %62 = vector.load %reinterpret_cast[%c68] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
        %63 = vector.load %reinterpret_cast[%c136] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
        %64 = vector.load %reinterpret_cast[%c204] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
        %65 = vector.extract %61[0] : f16 from vector<1xf16>
        %66 = vector.extract %62[0] : f16 from vector<1xf16>
        %67 = vector.extract %63[0] : f16 from vector<1xf16>
        %68 = vector.extract %64[0] : f16 from vector<1xf16>
        %69 = vector.from_elements %65, %66, %67, %68 : vector<4xf16>
        %70 = arith.addi %11, %c8 overflow<nsw, nuw> : index
        %71 = arith.muli %70, %c68 overflow<nsw> : index
        %72 = arith.addi %71, %58 overflow<nsw> : index
        %reinterpret_cast_13 = memref.reinterpret_cast %alloc to offset: [%72], sizes: [%c1073741822], strides: [1] : memref<1x64x1x68xf16, #gpu.address_space<workgroup>> to memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>
        %73 = vector.load %reinterpret_cast_13[%c0] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
        %74 = vector.load %reinterpret_cast_13[%c68] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
        %75 = vector.load %reinterpret_cast_13[%c136] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
        %76 = vector.load %reinterpret_cast_13[%c204] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
        %77 = vector.extract %73[0] : f16 from vector<1xf16>
        %78 = vector.extract %74[0] : f16 from vector<1xf16>
        %79 = vector.extract %75[0] : f16 from vector<1xf16>
        %80 = vector.extract %76[0] : f16 from vector<1xf16>
        %81 = vector.from_elements %77, %78, %79, %80 : vector<4xf16>
        %82 = arith.addi %11, %c16 overflow<nsw, nuw> : index
        %83 = arith.muli %82, %c68 overflow<nsw> : index
        %84 = arith.addi %83, %58 overflow<nsw> : index
        %reinterpret_cast_14 = memref.reinterpret_cast %alloc to offset: [%84], sizes: [%c1073741822], strides: [1] : memref<1x64x1x68xf16, #gpu.address_space<workgroup>> to memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>
        %85 = vector.load %reinterpret_cast_14[%c0] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
        %86 = vector.load %reinterpret_cast_14[%c68] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
        %87 = vector.load %reinterpret_cast_14[%c136] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
        %88 = vector.load %reinterpret_cast_14[%c204] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
        %89 = vector.extract %85[0] : f16 from vector<1xf16>
        %90 = vector.extract %86[0] : f16 from vector<1xf16>
        %91 = vector.extract %87[0] : f16 from vector<1xf16>
        %92 = vector.extract %88[0] : f16 from vector<1xf16>
        %93 = vector.from_elements %89, %90, %91, %92 : vector<4xf16>
        %94 = arith.addi %11, %c24 overflow<nsw, nuw> : index
        %95 = arith.muli %94, %c68 overflow<nsw> : index
        %96 = arith.addi %95, %58 overflow<nsw> : index
        %reinterpret_cast_15 = memref.reinterpret_cast %alloc to offset: [%96], sizes: [%c1073741822], strides: [1] : memref<1x64x1x68xf16, #gpu.address_space<workgroup>> to memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>
        %97 = vector.load %reinterpret_cast_15[%c0] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
        %98 = vector.load %reinterpret_cast_15[%c68] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
        %99 = vector.load %reinterpret_cast_15[%c136] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
        %100 = vector.load %reinterpret_cast_15[%c204] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
        %101 = vector.extract %97[0] : f16 from vector<1xf16>
        %102 = vector.extract %98[0] : f16 from vector<1xf16>
        %103 = vector.extract %99[0] : f16 from vector<1xf16>
        %104 = vector.extract %100[0] : f16 from vector<1xf16>
        %105 = vector.from_elements %101, %102, %103, %104 : vector<4xf16>
        %106 = arith.addi %11, %c32 overflow<nsw, nuw> : index
        %107 = arith.muli %106, %c68 overflow<nsw> : index
        %108 = arith.addi %107, %58 overflow<nsw> : index
        %reinterpret_cast_16 = memref.reinterpret_cast %alloc to offset: [%108], sizes: [%c1073741822], strides: [1] : memref<1x64x1x68xf16, #gpu.address_space<workgroup>> to memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>
        %109 = vector.load %reinterpret_cast_16[%c0] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
        %110 = vector.load %reinterpret_cast_16[%c68] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
        %111 = vector.load %reinterpret_cast_16[%c136] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
        %112 = vector.load %reinterpret_cast_16[%c204] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
        %113 = vector.extract %109[0] : f16 from vector<1xf16>
        %114 = vector.extract %110[0] : f16 from vector<1xf16>
        %115 = vector.extract %111[0] : f16 from vector<1xf16>
        %116 = vector.extract %112[0] : f16 from vector<1xf16>
        %117 = vector.from_elements %113, %114, %115, %116 : vector<4xf16>
        %118 = arith.addi %11, %c40 overflow<nsw, nuw> : index
        %119 = arith.muli %118, %c68 overflow<nsw> : index
        %120 = arith.addi %119, %58 overflow<nsw> : index
        %reinterpret_cast_17 = memref.reinterpret_cast %alloc to offset: [%120], sizes: [%c1073741822], strides: [1] : memref<1x64x1x68xf16, #gpu.address_space<workgroup>> to memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>
        %121 = vector.load %reinterpret_cast_17[%c0] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
        %122 = vector.load %reinterpret_cast_17[%c68] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
        %123 = vector.load %reinterpret_cast_17[%c136] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
        %124 = vector.load %reinterpret_cast_17[%c204] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
        %125 = vector.extract %121[0] : f16 from vector<1xf16>
        %126 = vector.extract %122[0] : f16 from vector<1xf16>
        %127 = vector.extract %123[0] : f16 from vector<1xf16>
        %128 = vector.extract %124[0] : f16 from vector<1xf16>
        %129 = vector.from_elements %125, %126, %127, %128 : vector<4xf16>
        %130 = arith.addi %11, %c48 overflow<nsw, nuw> : index
        %131 = arith.muli %130, %c68 overflow<nsw> : index
        %132 = arith.addi %131, %58 overflow<nsw> : index
        %reinterpret_cast_18 = memref.reinterpret_cast %alloc to offset: [%132], sizes: [%c1073741822], strides: [1] : memref<1x64x1x68xf16, #gpu.address_space<workgroup>> to memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>
        %133 = vector.load %reinterpret_cast_18[%c0] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
        %134 = vector.load %reinterpret_cast_18[%c68] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
        %135 = vector.load %reinterpret_cast_18[%c136] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
        %136 = vector.load %reinterpret_cast_18[%c204] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
        %137 = vector.extract %133[0] : f16 from vector<1xf16>
        %138 = vector.extract %134[0] : f16 from vector<1xf16>
        %139 = vector.extract %135[0] : f16 from vector<1xf16>
        %140 = vector.extract %136[0] : f16 from vector<1xf16>
        %141 = vector.from_elements %137, %138, %139, %140 : vector<4xf16>
        %142 = arith.addi %11, %c56 overflow<nsw, nuw> : index
        %143 = arith.muli %142, %c68 overflow<nsw> : index
        %144 = arith.addi %143, %58 overflow<nsw> : index
        %reinterpret_cast_19 = memref.reinterpret_cast %alloc to offset: [%144], sizes: [%c1073741822], strides: [1] : memref<1x64x1x68xf16, #gpu.address_space<workgroup>> to memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>
        %145 = vector.load %reinterpret_cast_19[%c0] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
        %146 = vector.load %reinterpret_cast_19[%c68] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
        %147 = vector.load %reinterpret_cast_19[%c136] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
        %148 = vector.load %reinterpret_cast_19[%c204] : memref<?xf16, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
        %149 = vector.extract %145[0] : f16 from vector<1xf16>
        %150 = vector.extract %146[0] : f16 from vector<1xf16>
        %151 = vector.extract %147[0] : f16 from vector<1xf16>
        %152 = vector.extract %148[0] : f16 from vector<1xf16>
        %153 = vector.from_elements %149, %150, %151, %152 : vector<4xf16>
        %154 = vector.load %alloc_12[%c0, %c0, %4, %11] : memref<1x1x64x7xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %155 = arith.addi %4, %c32 overflow<nsw, nuw> : index
        %156 = vector.load %alloc_12[%c0, %c0, %155, %11] : memref<1x1x64x7xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %157 = amdgpu.mfma %154 * %16 + %cst_10 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
        %158 = amdgpu.mfma %156 * %16 + %cst_10 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
        %159 = arith.addf %157, %cst_7 : vector<16xf32>
        %160 = arith.addf %158, %cst_7 : vector<16xf32>
        %161 = vector.splat %11 : vector<4xindex>
        %162 = arith.addi %161, %cst_5 overflow<nsw, nuw> : vector<4xindex>
        %163 = arith.cmpi slt, %162, %cst : vector<4xindex>
        %164 = arith.andi %163, %8 : vector<4xi1>
        vector.maskedstore %15[%c0, %6, %11], %164, %cst_0 : memref<1x1x2xf32, strided<[2, 2, 1], offset: ?>>, vector<4xi1>, vector<4xf32>
        %165 = vector.extract_strided_slice %159 {offsets = [0], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
        %166 = vector.extract_strided_slice %159 {offsets = [1], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
        %167 = arith.maximumf %165, %166 : vector<1xf32>
        %168 = vector.extract_strided_slice %159 {offsets = [2], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
        %169 = arith.maximumf %167, %168 : vector<1xf32>
        %170 = vector.extract_strided_slice %159 {offsets = [3], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
        %171 = arith.maximumf %169, %170 : vector<1xf32>
        %172 = vector.extract_strided_slice %159 {offsets = [4], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
        %173 = arith.maximumf %171, %172 : vector<1xf32>
        %174 = vector.extract_strided_slice %159 {offsets = [5], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
        %175 = arith.maximumf %173, %174 : vector<1xf32>
        %176 = vector.extract_strided_slice %159 {offsets = [6], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
        %177 = arith.maximumf %175, %176 : vector<1xf32>
        %178 = vector.extract_strided_slice %159 {offsets = [7], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
        %179 = arith.maximumf %177, %178 : vector<1xf32>
        %180 = vector.extract_strided_slice %159 {offsets = [8], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
        %181 = arith.maximumf %179, %180 : vector<1xf32>
        %182 = vector.extract_strided_slice %159 {offsets = [9], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
        %183 = arith.maximumf %181, %182 : vector<1xf32>
        %184 = vector.extract_strided_slice %159 {offsets = [10], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
        %185 = arith.maximumf %183, %184 : vector<1xf32>
        %186 = vector.extract_strided_slice %159 {offsets = [11], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
        %187 = arith.maximumf %185, %186 : vector<1xf32>
        %188 = vector.extract_strided_slice %159 {offsets = [12], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
        %189 = arith.maximumf %187, %188 : vector<1xf32>
        %190 = vector.extract_strided_slice %159 {offsets = [13], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
        %191 = arith.maximumf %189, %190 : vector<1xf32>
        %192 = vector.extract_strided_slice %159 {offsets = [14], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
        %193 = arith.maximumf %191, %192 : vector<1xf32>
        %194 = vector.extract_strided_slice %159 {offsets = [15], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
        %195 = arith.maximumf %193, %194 : vector<1xf32>
        %196 = vector.extract_strided_slice %160 {offsets = [0], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
        %197 = vector.extract_strided_slice %160 {offsets = [1], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
        %198 = arith.maximumf %196, %197 : vector<1xf32>
        %199 = vector.extract_strided_slice %160 {offsets = [2], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
        %200 = arith.maximumf %198, %199 : vector<1xf32>
        %201 = vector.extract_strided_slice %160 {offsets = [3], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
        %202 = arith.maximumf %200, %201 : vector<1xf32>
        %203 = vector.extract_strided_slice %160 {offsets = [4], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
        %204 = arith.maximumf %202, %203 : vector<1xf32>
        %205 = vector.extract_strided_slice %160 {offsets = [5], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
        %206 = arith.maximumf %204, %205 : vector<1xf32>
        %207 = vector.extract_strided_slice %160 {offsets = [6], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
        %208 = arith.maximumf %206, %207 : vector<1xf32>
        %209 = vector.extract_strided_slice %160 {offsets = [7], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
        %210 = arith.maximumf %208, %209 : vector<1xf32>
        %211 = vector.extract_strided_slice %160 {offsets = [8], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
        %212 = arith.maximumf %210, %211 : vector<1xf32>
        %213 = vector.extract_strided_slice %160 {offsets = [9], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
        %214 = arith.maximumf %212, %213 : vector<1xf32>
        %215 = vector.extract_strided_slice %160 {offsets = [10], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
        %216 = arith.maximumf %214, %215 : vector<1xf32>
        %217 = vector.extract_strided_slice %160 {offsets = [11], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
        %218 = arith.maximumf %216, %217 : vector<1xf32>
        %219 = vector.extract_strided_slice %160 {offsets = [12], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
        %220 = arith.maximumf %218, %219 : vector<1xf32>
        %221 = vector.extract_strided_slice %160 {offsets = [13], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
        %222 = arith.maximumf %220, %221 : vector<1xf32>
        %223 = vector.extract_strided_slice %160 {offsets = [14], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
        %224 = arith.maximumf %222, %223 : vector<1xf32>
        %225 = vector.extract_strided_slice %160 {offsets = [15], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
        %226 = arith.maximumf %224, %225 : vector<1xf32>
        %227 = arith.maximumf %195, %226 : vector<1xf32>
        %228 = vector.extract %227[0] : f32 from vector<1xf32>
        %shuffleResult, %valid = gpu.shuffle  xor %228, %c32_i32, %c64_i32 : f32
        %229 = vector.broadcast %shuffleResult : f32 to vector<1xf32>
        %230 = arith.maximumf %227, %229 : vector<1xf32>
        %231 = arith.maximumf %230, %cst_8 : vector<1xf32>
        %232 = arith.subf %cst_8, %231 : vector<1xf32>
        %233 = math.exp2 %232 : vector<1xf32>
        %234 = vector.extract %231[0] : f32 from vector<1xf32>
        %235 = vector.splat %234 : vector<16xf32>
        %236 = arith.subf %159, %235 : vector<16xf32>
        %237 = arith.subf %160, %235 : vector<16xf32>
        %238 = math.exp2 %236 : vector<16xf32>
        %239 = math.exp2 %237 : vector<16xf32>
        %240 = arith.mulf %233, %cst_9 : vector<1xf32>
        %241 = arith.addf %238, %239 : vector<16xf32>
        %242 = vector.extract_strided_slice %241 {offsets = [0], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
        %243 = vector.extract_strided_slice %241 {offsets = [1], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
        %244 = arith.addf %242, %243 : vector<1xf32>
        %245 = vector.extract_strided_slice %241 {offsets = [2], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
        %246 = arith.addf %244, %245 : vector<1xf32>
        %247 = vector.extract_strided_slice %241 {offsets = [3], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
        %248 = arith.addf %246, %247 : vector<1xf32>
        %249 = vector.extract_strided_slice %241 {offsets = [4], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
        %250 = arith.addf %248, %249 : vector<1xf32>
        %251 = vector.extract_strided_slice %241 {offsets = [5], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
        %252 = arith.addf %250, %251 : vector<1xf32>
        %253 = vector.extract_strided_slice %241 {offsets = [6], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
        %254 = arith.addf %252, %253 : vector<1xf32>
        %255 = vector.extract_strided_slice %241 {offsets = [7], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
        %256 = arith.addf %254, %255 : vector<1xf32>
        %257 = vector.extract_strided_slice %241 {offsets = [8], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
        %258 = arith.addf %256, %257 : vector<1xf32>
        %259 = vector.extract_strided_slice %241 {offsets = [9], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
        %260 = arith.addf %258, %259 : vector<1xf32>
        %261 = vector.extract_strided_slice %241 {offsets = [10], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
        %262 = arith.addf %260, %261 : vector<1xf32>
        %263 = vector.extract_strided_slice %241 {offsets = [11], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
        %264 = arith.addf %262, %263 : vector<1xf32>
        %265 = vector.extract_strided_slice %241 {offsets = [12], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
        %266 = arith.addf %264, %265 : vector<1xf32>
        %267 = vector.extract_strided_slice %241 {offsets = [13], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
        %268 = arith.addf %266, %267 : vector<1xf32>
        %269 = vector.extract_strided_slice %241 {offsets = [14], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
        %270 = arith.addf %268, %269 : vector<1xf32>
        %271 = vector.extract_strided_slice %241 {offsets = [15], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
        %272 = arith.addf %270, %271 : vector<1xf32>
        %273 = vector.extract %272[0] : f32 from vector<1xf32>
        %shuffleResult_20, %valid_21 = gpu.shuffle  xor %273, %c32_i32, %c64_i32 : f32
        %274 = vector.broadcast %shuffleResult_20 : f32 to vector<1xf32>
        %275 = arith.addf %272, %274 : vector<1xf32>
        %276 = arith.addf %240, %275 : vector<1xf32>
        %277 = arith.truncf %238 : vector<16xf32> to vector<16xf16>
        %278 = arith.truncf %239 : vector<16xf32> to vector<16xf16>
        %279 = vector.extract %233[0] : f32 from vector<1xf32>
        %280 = vector.splat %279 : vector<16xf32>
        %281 = arith.mulf %280, %cst_10 : vector<16xf32>
        %282 = vector.extract_strided_slice %277 {offsets = [0], sizes = [4], strides = [1]} : vector<16xf16> to vector<4xf16>
        %283 = vector.extract_strided_slice %277 {offsets = [4], sizes = [4], strides = [1]} : vector<16xf16> to vector<4xf16>
        %284 = vector.extract_strided_slice %277 {offsets = [8], sizes = [4], strides = [1]} : vector<16xf16> to vector<4xf16>
        %285 = vector.extract_strided_slice %277 {offsets = [12], sizes = [4], strides = [1]} : vector<16xf16> to vector<4xf16>
        %286 = vector.extract_strided_slice %278 {offsets = [0], sizes = [4], strides = [1]} : vector<16xf16> to vector<4xf16>
        %287 = vector.extract_strided_slice %278 {offsets = [4], sizes = [4], strides = [1]} : vector<16xf16> to vector<4xf16>
        %288 = vector.extract_strided_slice %278 {offsets = [8], sizes = [4], strides = [1]} : vector<16xf16> to vector<4xf16>
        %289 = vector.extract_strided_slice %278 {offsets = [12], sizes = [4], strides = [1]} : vector<16xf16> to vector<4xf16>
        %290 = amdgpu.mfma %69 * %282 + %281 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
        %291 = amdgpu.mfma %81 * %283 + %290 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
        %292 = amdgpu.mfma %93 * %284 + %291 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
        %293 = amdgpu.mfma %105 * %285 + %292 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
        %294 = amdgpu.mfma %117 * %286 + %293 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
        %295 = amdgpu.mfma %129 * %287 + %294 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
        %296 = amdgpu.mfma %141 * %288 + %295 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
        %297 = amdgpu.mfma %153 * %289 + %296 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
        %298 = arith.divf %cst_4, %276 : vector<1xf32>
        %299 = vector.extract %298[0] : f32 from vector<1xf32>
        %300 = vector.splat %299 : vector<16xf32>
        %301 = arith.mulf %297, %300 : vector<16xf32>
        %302 = vector.extract_strided_slice %301 {offsets = [0], sizes = [4], strides = [1]} : vector<16xf32> to vector<4xf32>
        %303 = stream.binding.subspan %arg4[%c0] : !stream.binding -> memref<1x1x1x2xf32, strided<[2, 2, 2, 1], offset: ?>>
        %304 = arith.addi %42, %57 overflow<nsw, nuw> : index
        %305 = arith.addi %304, %11 overflow<nsw, nuw> : index
        %306 = vector.splat %305 : vector<4xindex>
        %307 = arith.addi %306, %cst_5 overflow<nsw, nuw> : vector<4xindex>
        %308 = arith.cmpi slt, %307, %cst : vector<4xindex>
        %309 = arith.andi %8, %308 : vector<4xi1>
        vector.maskedstore %303[%c0, %6, %c0, %305], %309, %302 : memref<1x1x1x2xf32, strided<[2, 2, 2, 1], offset: ?>>, vector<4xi1>, vector<4xf32>
        return
      }
    }
  }
  func.func @isolated_benchmark(%arg0: tensor<1x1x1x3xf16>, %arg1: tensor<1x2x1x3xf16>, %arg2: tensor<1x2x1x2xf16>, %arg3: tensor<1x1xi8>, %arg4: tensor<1x1x1x2xf32>, %arg5: tensor<1x1x2xf32>) -> (tensor<1x1x1x2xf32>, tensor<1x1x2xf32>) {
    %0:2 = flow.dispatch @base_attention_custom_mask::@base_attention_custom_mask(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5) : (tensor<1x1x1x3xf16>, tensor<1x2x1x3xf16>, tensor<1x2x1x2xf16>, tensor<1x1xi8>, tensor<1x1x1x2xf32>, tensor<1x1x2xf32>) -> (%arg4, %arg5)
    return %0#0, %0#1 : tensor<1x1x1x2xf32>, tensor<1x1x2xf32>
  }
}
