#translation = #iree_codegen.translation_info<pipeline = None workgroup_size = [256, 1, 1] subgroup_size = 64>
module attributes {transform.with_named_sequence} {
  stream.executable private @extend_attention {
    stream.executable.export public @extend_attention workgroups(%arg0: index, %arg1: index) -> (index, index, index) {
      %c8 = arith.constant 8 : index
      %c2 = arith.constant 2 : index
      %c32 = arith.constant 32 : index
      stream.return %c8, %c2, %c32 : index, index, index
    }
    builtin.module {
      func.func @extend_attention(%arg0: !stream.binding, %arg1: !stream.binding, %arg2: !stream.binding, %arg3: !stream.binding, %arg4: !stream.binding, %arg5: !stream.binding, %arg6: !stream.binding, %arg7: !stream.binding, %arg8: !stream.binding, %arg9: !stream.binding, %arg10: !stream.binding, %arg11: index, %arg12: index) attributes {translation_info = #translation} {
        %cst = arith.constant dense<16> : vector<1xindex>
        %cst_0 = arith.constant dense<0.000000e+00> : vector<8xf16>
        %cst_1 = arith.constant dense<[0, 128, 256, 384]> : vector<4xindex>
        %cst_2 = arith.constant dense<2160> : vector<4xindex>
        %cst_3 = arith.constant dense<2144> : vector<4xindex>
        %cst_4 = arith.constant dense<2128> : vector<4xindex>
        %cst_5 = arith.constant dense<2112> : vector<4xindex>
        %cst_6 = arith.constant dense<2096> : vector<4xindex>
        %cst_7 = arith.constant dense<2080> : vector<4xindex>
        %cst_8 = arith.constant dense<2064> : vector<4xindex>
        %cst_9 = arith.constant dense<2048> : vector<4xindex>
        %cst_10 = arith.constant dense<112> : vector<4xindex>
        %cst_11 = arith.constant dense<96> : vector<4xindex>
        %cst_12 = arith.constant dense<80> : vector<4xindex>
        %cst_13 = arith.constant dense<64> : vector<4xindex>
        %cst_14 = arith.constant dense<48> : vector<4xindex>
        %cst_15 = arith.constant dense<32> : vector<4xindex>
        %cst_16 = arith.constant dense<128> : vector<4xindex>
        %cst_17 = arith.constant dense<16> : vector<4xindex>
        %cst_18 = arith.constant dense<0> : vector<4xi32>
        %c51 = arith.constant 51 : index
        %c50 = arith.constant 50 : index
        %c49 = arith.constant 49 : index
        %c35 = arith.constant 35 : index
        %c34 = arith.constant 34 : index
        %c33 = arith.constant 33 : index
        %c19 = arith.constant 19 : index
        %c18 = arith.constant 18 : index
        %c17 = arith.constant 17 : index
        %c3 = arith.constant 3 : index
        %cst_19 = arith.constant dense<1.000000e+00> : vector<1xf32>
        %cst_20 = arith.constant dense<[0, 128, 256, 384, 512, 640, 768, 896]> : vector<8xindex>
        %cst_21 = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7]> : vector<8xindex>
        %c8 = arith.constant 8 : index
        %c32_i32 = arith.constant 32 : i32
        %c64_i32 = arith.constant 64 : i32
        %c16_i32 = arith.constant 16 : i32
        %cst_22 = arith.constant dense<1.000000e+00> : vector<4xf16>
        %cst_23 = arith.constant dense<0.000000e+00> : vector<4xf16>
        %c512 = arith.constant 512 : index
        %c4096 = arith.constant 4096 : index
        %c128 = arith.constant 128 : index
        %c112 = arith.constant 112 : index
        %c96 = arith.constant 96 : index
        %c80 = arith.constant 80 : index
        %c48 = arith.constant 48 : index
        %c32 = arith.constant 32 : index
        %cst_24 = arith.constant dense<[0, 1, 2, 3]> : vector<4xindex>
        %c4 = arith.constant 4 : index
        %c64 = arith.constant 64 : index
        %c1 = arith.constant 1 : index
        %c16 = arith.constant 16 : index
        %c2 = arith.constant 2 : index
        %c0 = arith.constant 0 : index
        %cst_25 = arith.constant dense<43.2808495> : vector<4xf32>
        %cst_26 = arith.constant dense<-1.000000e+06> : vector<4xf32>
        %cst_27 = arith.constant dense<-1.000000e+06> : vector<1xf32>
        %cst_28 = arith.constant dense<0.000000e+00> : vector<1xf32>
        %cst_29 = arith.constant dense<0.000000e+00> : vector<4xf32>
        %workgroup_id_0 = stream.dispatch.workgroup.id[0] : index
        %workgroup_id_1 = stream.dispatch.workgroup.id[1] : index
        %workgroup_id_2 = stream.dispatch.workgroup.id[2] : index
        %thread_id_x = gpu.thread_id  x
        %thread_id_y = gpu.thread_id  y
        %thread_id_z = gpu.thread_id  z
        %0 = stream.binding.subspan %arg6[%c0] : !stream.binding -> memref<2xi32, strided<[1], offset: ?>>
        %1 = arith.remsi %workgroup_id_2, %c2 : index
        %2 = vector.load %0[%1] : memref<2xi32, strided<[1], offset: ?>>, vector<1xi32>
        %3 = arith.index_cast %2 : vector<1xi32> to vector<1xindex>
        %4 = vector.extract %3[0] : index from vector<1xindex>
        %5 = stream.binding.subspan %arg9[%c0] : !stream.binding -> memref<2xi32, strided<[1], offset: ?>>
        %6 = vector.load %5[%1] : memref<2xi32, strided<[1], offset: ?>>, vector<1xi32>
        %7 = arith.index_cast %6 : vector<1xi32> to vector<1xindex>
        %8 = vector.extract %7[0] : index from vector<1xindex>
        %9 = stream.binding.subspan %arg8[%c0] : !stream.binding -> memref<2xi32, strided<[1], offset: ?>>
        %10 = vector.load %9[%1] : memref<2xi32, strided<[1], offset: ?>>, vector<1xi32>
        %11 = arith.index_cast %10 : vector<1xi32> to vector<1xindex>
        %12 = vector.extract %11[0] : index from vector<1xindex>
        %13 = stream.binding.subspan %arg7[%c0] : !stream.binding -> memref<2xi32, strided<[1], offset: ?>>
        %14 = vector.load %13[%1] : memref<2xi32, strided<[1], offset: ?>>, vector<1xi32>
        %15 = arith.subi %14, %10 : vector<1xi32>
        %16 = arith.index_cast %15 : vector<1xi32> to vector<1xindex>
        %17 = vector.extract %16[0] : index from vector<1xindex>
        %alloc = memref.alloc() : memref<1x64x36xf16, #gpu.address_space<workgroup>>
        %alloc_30 = memref.alloc() : memref<1x32x132xf16, #gpu.address_space<workgroup>>
        %18 = stream.binding.subspan %arg0[%c0] : !stream.binding -> memref<?x16x128xf16, strided<[2048, 128, 1], offset: ?>>{%arg11}
        %19 = arith.divsi %thread_id_x, %c64 : index
        %20 = arith.muli %19, %c16 overflow<nsw, nuw> : index
        %21 = arith.muli %workgroup_id_0, %c64 overflow<nsw, nuw> : index
        %22 = arith.remsi %thread_id_x, %c16 : index
        %23 = arith.addi %22, %21 overflow<nsw, nuw> : index
        %24 = arith.addi %23, %20 overflow<nsw, nuw> : index
        %25 = arith.cmpi slt, %24, %12 : index
        %26 = vector.splat %25 : vector<4xi1>
        %27 = arith.addi %24, %8 overflow<nsw, nuw> : index
        %28 = arith.divsi %workgroup_id_2, %c2 : index
        %29 = arith.remsi %28, %c16 : index
        %30 = arith.remsi %thread_id_x, %c64 : index
        %31 = arith.divsi %30, %c16 : index
        %32 = arith.muli %31, %c4 overflow<nsw, nuw> : index
        %33 = vector.maskedload %18[%27, %29, %32], %26, %cst_23 : memref<?x16x128xf16, strided<[2048, 128, 1], offset: ?>>, vector<4xi1>, vector<4xf16> into vector<4xf16>
        %34 = arith.addi %32, %c16 overflow<nsw, nuw> : index
        %35 = vector.maskedload %18[%27, %29, %34], %26, %cst_23 : memref<?x16x128xf16, strided<[2048, 128, 1], offset: ?>>, vector<4xi1>, vector<4xf16> into vector<4xf16>
        %36 = arith.addi %32, %c32 overflow<nsw, nuw> : index
        %37 = vector.maskedload %18[%27, %29, %36], %26, %cst_23 : memref<?x16x128xf16, strided<[2048, 128, 1], offset: ?>>, vector<4xi1>, vector<4xf16> into vector<4xf16>
        %38 = arith.addi %32, %c48 overflow<nsw, nuw> : index
        %39 = vector.maskedload %18[%27, %29, %38], %26, %cst_23 : memref<?x16x128xf16, strided<[2048, 128, 1], offset: ?>>, vector<4xi1>, vector<4xf16> into vector<4xf16>
        %40 = arith.addi %32, %c64 overflow<nsw, nuw> : index
        %41 = vector.maskedload %18[%27, %29, %40], %26, %cst_23 : memref<?x16x128xf16, strided<[2048, 128, 1], offset: ?>>, vector<4xi1>, vector<4xf16> into vector<4xf16>
        %42 = arith.addi %32, %c80 overflow<nsw, nuw> : index
        %43 = vector.maskedload %18[%27, %29, %42], %26, %cst_23 : memref<?x16x128xf16, strided<[2048, 128, 1], offset: ?>>, vector<4xi1>, vector<4xf16> into vector<4xf16>
        %44 = arith.addi %32, %c96 overflow<nsw, nuw> : index
        %45 = vector.maskedload %18[%27, %29, %44], %26, %cst_23 : memref<?x16x128xf16, strided<[2048, 128, 1], offset: ?>>, vector<4xi1>, vector<4xf16> into vector<4xf16>
        %46 = arith.addi %32, %c112 overflow<nsw, nuw> : index
        %47 = vector.maskedload %18[%27, %29, %46], %26, %cst_23 : memref<?x16x128xf16, strided<[2048, 128, 1], offset: ?>>, vector<4xi1>, vector<4xf16> into vector<4xf16>
        %48 = arith.subi %17, %c1 : index
        %49 = arith.divui %48, %c32 : index
        %50 = arith.addi %49, %c1 : index
        %51 = arith.cmpi eq, %17, %c0 : index
        %52 = arith.select %51, %c0, %50 : index
        %53 = stream.binding.subspan %arg4[%c0] : !stream.binding -> memref<1316x1x128xf16, strided<[128, 128, 1], offset: ?>>
        %54 = stream.binding.subspan %arg3[%c0] : !stream.binding -> memref<1316x1x128xf16, strided<[128, 128, 1], offset: ?>>
        %55 = stream.binding.subspan %arg5[%c0] : !stream.binding -> memref<2x864xi32, strided<[864, 1], offset: ?>>
        %56 = vector.splat %17 : vector<4xindex>
        %57 = arith.muli %22, %c128 overflow<nsw, nuw> : index
        %58 = arith.divsi %29, %c16 : index
        %59 = arith.muli %58, %c128 overflow<nsw, nuw> : index
        %60 = vector.splat %32 : vector<4xindex>
        %61 = arith.addi %22, %c16 overflow<nsw, nuw> : index
        %62 = arith.muli %thread_id_y, %c64 overflow<nsw, nuw> : index
        %63 = arith.muli %workgroup_id_1, %c64 overflow<nsw, nuw> : index
        %64 = arith.muli %31, %c512 overflow<nsw, nuw> : index
        %65 = vector.splat %63 : vector<4xindex>
        %66 = vector.splat %62 : vector<4xindex>
        %67 = arith.addi %22, %62 overflow<nsw, nuw> : index
        %68 = arith.addi %67, %c16 overflow<nsw, nuw> : index
        %69 = arith.addi %67, %c32 overflow<nsw, nuw> : index
        %70 = arith.addi %67, %c48 overflow<nsw, nuw> : index
        %71:6 = scf.for %arg13 = %c0 to %52 step %c1 iter_args(%arg14 = %cst_27, %arg15 = %cst_28, %arg16 = %cst_29, %arg17 = %cst_29, %arg18 = %cst_29, %arg19 = %cst_29) -> (vector<1xf32>, vector<1xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>) {
          amdgpu.lds_barrier
          %166 = arith.muli %arg13, %c32 overflow<nsw, nuw> : index
          %167 = arith.addi %22, %166 overflow<nsw, nuw> : index
          %168 = vector.splat %167 : vector<4xindex>
          %169 = arith.addi %168, %cst_24 overflow<nsw, nuw> : vector<4xindex>
          %170 = arith.cmpi slt, %169, %56 : vector<4xindex>
          %171 = vector.maskedload %55[%4, %167], %170, %cst_18 : memref<2x864xi32, strided<[864, 1], offset: ?>>, vector<4xi1>, vector<4xi32> into vector<4xi32>
          %172 = arith.addi %169, %cst_17 overflow<nsw, nuw> : vector<4xindex>
          %173 = arith.cmpi slt, %172, %56 : vector<4xindex>
          %174 = arith.addi %167, %c16 overflow<nsw, nuw> : index
          %175 = vector.maskedload %55[%4, %174], %173, %cst_18 : memref<2x864xi32, strided<[864, 1], offset: ?>>, vector<4xi1>, vector<4xi32> into vector<4xi32>
          %176 = arith.index_cast %171 : vector<4xi32> to vector<4xindex>
          %177 = arith.cmpi slt, %167, %17 : index
          %178 = vector.splat %177 : vector<4xi1>
          %179 = arith.muli %176, %cst_16 overflow<nsw, nuw> : vector<4xindex>
          %180 = arith.muli %arg13, %c4096 overflow<nsw, nuw> : index
          %181 = arith.addi %180, %59 overflow<nsw, nuw> : index
          %182 = arith.addi %181, %57 overflow<nsw, nuw> : index
          %183 = vector.splat %182 : vector<4xindex>
          %184 = arith.addi %183, %179 overflow<nsw, nuw> : vector<4xindex>
          %185 = arith.addi %184, %60 overflow<nsw, nuw> : vector<4xindex>
          %186 = arith.addi %185, %cst_24 overflow<nsw, nuw> : vector<4xindex>
          %187 = vector.gather %54[%c0, %c0, %c0] [%186], %178, %cst_23 : memref<1316x1x128xf16, strided<[128, 128, 1], offset: ?>>, vector<4xindex>, vector<4xi1>, vector<4xf16> into vector<4xf16>
          %188 = arith.addi %186, %cst_17 overflow<nsw, nuw> : vector<4xindex>
          %189 = vector.gather %54[%c0, %c0, %c0] [%188], %178, %cst_23 : memref<1316x1x128xf16, strided<[128, 128, 1], offset: ?>>, vector<4xindex>, vector<4xi1>, vector<4xf16> into vector<4xf16>
          %190 = arith.addi %186, %cst_15 overflow<nsw, nuw> : vector<4xindex>
          %191 = vector.gather %54[%c0, %c0, %c0] [%190], %178, %cst_23 : memref<1316x1x128xf16, strided<[128, 128, 1], offset: ?>>, vector<4xindex>, vector<4xi1>, vector<4xf16> into vector<4xf16>
          %192 = arith.addi %186, %cst_14 overflow<nsw, nuw> : vector<4xindex>
          %193 = vector.gather %54[%c0, %c0, %c0] [%192], %178, %cst_23 : memref<1316x1x128xf16, strided<[128, 128, 1], offset: ?>>, vector<4xindex>, vector<4xi1>, vector<4xf16> into vector<4xf16>
          %194 = arith.addi %186, %cst_13 overflow<nsw, nuw> : vector<4xindex>
          %195 = vector.gather %54[%c0, %c0, %c0] [%194], %178, %cst_23 : memref<1316x1x128xf16, strided<[128, 128, 1], offset: ?>>, vector<4xindex>, vector<4xi1>, vector<4xf16> into vector<4xf16>
          %196 = arith.addi %186, %cst_12 overflow<nsw, nuw> : vector<4xindex>
          %197 = vector.gather %54[%c0, %c0, %c0] [%196], %178, %cst_23 : memref<1316x1x128xf16, strided<[128, 128, 1], offset: ?>>, vector<4xindex>, vector<4xi1>, vector<4xf16> into vector<4xf16>
          %198 = arith.addi %186, %cst_11 overflow<nsw, nuw> : vector<4xindex>
          %199 = vector.gather %54[%c0, %c0, %c0] [%198], %178, %cst_23 : memref<1316x1x128xf16, strided<[128, 128, 1], offset: ?>>, vector<4xindex>, vector<4xi1>, vector<4xf16> into vector<4xf16>
          %200 = arith.addi %186, %cst_10 overflow<nsw, nuw> : vector<4xindex>
          %201 = vector.gather %54[%c0, %c0, %c0] [%200], %178, %cst_23 : memref<1316x1x128xf16, strided<[128, 128, 1], offset: ?>>, vector<4xindex>, vector<4xi1>, vector<4xf16> into vector<4xf16>
          %202 = arith.index_cast %175 : vector<4xi32> to vector<4xindex>
          %203 = arith.cmpi slt, %174, %17 : index
          %204 = vector.splat %203 : vector<4xi1>
          %205 = arith.muli %202, %cst_16 overflow<nsw, nuw> : vector<4xindex>
          %206 = arith.addi %183, %205 overflow<nsw, nuw> : vector<4xindex>
          %207 = arith.addi %206, %60 overflow<nsw, nuw> : vector<4xindex>
          %208 = arith.addi %207, %cst_24 overflow<nsw, nuw> : vector<4xindex>
          %209 = arith.addi %208, %cst_9 overflow<nsw, nuw> : vector<4xindex>
          %210 = vector.gather %54[%c0, %c0, %c0] [%209], %204, %cst_23 : memref<1316x1x128xf16, strided<[128, 128, 1], offset: ?>>, vector<4xindex>, vector<4xi1>, vector<4xf16> into vector<4xf16>
          %211 = arith.addi %208, %cst_8 overflow<nsw, nuw> : vector<4xindex>
          %212 = vector.gather %54[%c0, %c0, %c0] [%211], %204, %cst_23 : memref<1316x1x128xf16, strided<[128, 128, 1], offset: ?>>, vector<4xindex>, vector<4xi1>, vector<4xf16> into vector<4xf16>
          %213 = arith.addi %208, %cst_7 overflow<nsw, nuw> : vector<4xindex>
          %214 = vector.gather %54[%c0, %c0, %c0] [%213], %204, %cst_23 : memref<1316x1x128xf16, strided<[128, 128, 1], offset: ?>>, vector<4xindex>, vector<4xi1>, vector<4xf16> into vector<4xf16>
          %215 = arith.addi %208, %cst_6 overflow<nsw, nuw> : vector<4xindex>
          %216 = vector.gather %54[%c0, %c0, %c0] [%215], %204, %cst_23 : memref<1316x1x128xf16, strided<[128, 128, 1], offset: ?>>, vector<4xindex>, vector<4xi1>, vector<4xf16> into vector<4xf16>
          %217 = arith.addi %208, %cst_5 overflow<nsw, nuw> : vector<4xindex>
          %218 = vector.gather %54[%c0, %c0, %c0] [%217], %204, %cst_23 : memref<1316x1x128xf16, strided<[128, 128, 1], offset: ?>>, vector<4xindex>, vector<4xi1>, vector<4xf16> into vector<4xf16>
          %219 = arith.addi %208, %cst_4 overflow<nsw, nuw> : vector<4xindex>
          %220 = vector.gather %54[%c0, %c0, %c0] [%219], %204, %cst_23 : memref<1316x1x128xf16, strided<[128, 128, 1], offset: ?>>, vector<4xindex>, vector<4xi1>, vector<4xf16> into vector<4xf16>
          %221 = arith.addi %208, %cst_3 overflow<nsw, nuw> : vector<4xindex>
          %222 = vector.gather %54[%c0, %c0, %c0] [%221], %204, %cst_23 : memref<1316x1x128xf16, strided<[128, 128, 1], offset: ?>>, vector<4xindex>, vector<4xi1>, vector<4xf16> into vector<4xf16>
          %223 = arith.addi %208, %cst_2 overflow<nsw, nuw> : vector<4xindex>
          %224 = vector.gather %54[%c0, %c0, %c0] [%223], %204, %cst_23 : memref<1316x1x128xf16, strided<[128, 128, 1], offset: ?>>, vector<4xindex>, vector<4xi1>, vector<4xf16> into vector<4xf16>
          vector.store %187, %alloc_30[%c0, %22, %32] : memref<1x32x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          vector.store %189, %alloc_30[%c0, %22, %34] : memref<1x32x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          vector.store %191, %alloc_30[%c0, %22, %36] : memref<1x32x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          vector.store %193, %alloc_30[%c0, %22, %38] : memref<1x32x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          vector.store %195, %alloc_30[%c0, %22, %40] : memref<1x32x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          vector.store %197, %alloc_30[%c0, %22, %42] : memref<1x32x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          vector.store %199, %alloc_30[%c0, %22, %44] : memref<1x32x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          vector.store %201, %alloc_30[%c0, %22, %46] : memref<1x32x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          vector.store %210, %alloc_30[%c0, %61, %32] : memref<1x32x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          vector.store %212, %alloc_30[%c0, %61, %34] : memref<1x32x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          vector.store %214, %alloc_30[%c0, %61, %36] : memref<1x32x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          vector.store %216, %alloc_30[%c0, %61, %38] : memref<1x32x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          vector.store %218, %alloc_30[%c0, %61, %40] : memref<1x32x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          vector.store %220, %alloc_30[%c0, %61, %42] : memref<1x32x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          vector.store %222, %alloc_30[%c0, %61, %44] : memref<1x32x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          vector.store %224, %alloc_30[%c0, %61, %46] : memref<1x32x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %225 = arith.addi %166, %32 overflow<nsw, nuw> : index
          %226 = vector.splat %225 : vector<4xindex>
          %227 = arith.addi %226, %cst_24 overflow<nsw, nuw> : vector<4xindex>
          %228 = arith.cmpi slt, %227, %56 : vector<4xindex>
          %229 = arith.addi %22, %180 overflow<nsw, nuw> : index
          %230 = arith.addi %229, %64 overflow<nsw, nuw> : index
          %231 = arith.addi %230, %59 overflow<nsw, nuw> : index
          %232 = vector.splat %231 : vector<4xindex>
          %233 = arith.addi %232, %179 overflow<nsw, nuw> : vector<4xindex>
          %234 = arith.addi %233, %cst_1 overflow<nsw, nuw> : vector<4xindex>
          %235 = arith.addi %234, %65 overflow<nsw, nuw> : vector<4xindex>
          %236 = arith.addi %235, %66 overflow<nsw, nuw> : vector<4xindex>
          %237 = vector.gather %53[%c0, %c0, %c0] [%236], %228, %cst_23 : memref<1316x1x128xf16, strided<[128, 128, 1], offset: ?>>, vector<4xindex>, vector<4xi1>, vector<4xf16> into vector<4xf16>
          %238 = arith.addi %227, %cst_17 overflow<nsw, nuw> : vector<4xindex>
          %239 = arith.cmpi slt, %238, %56 : vector<4xindex>
          %240 = arith.addi %232, %205 overflow<nsw, nuw> : vector<4xindex>
          %241 = arith.addi %240, %cst_1 overflow<nsw, nuw> : vector<4xindex>
          %242 = arith.addi %241, %65 overflow<nsw, nuw> : vector<4xindex>
          %243 = arith.addi %242, %66 overflow<nsw, nuw> : vector<4xindex>
          %244 = arith.addi %243, %cst_9 overflow<nsw, nuw> : vector<4xindex>
          %245 = vector.gather %53[%c0, %c0, %c0] [%244], %239, %cst_23 : memref<1316x1x128xf16, strided<[128, 128, 1], offset: ?>>, vector<4xindex>, vector<4xi1>, vector<4xf16> into vector<4xf16>
          %246 = arith.addi %236, %cst_17 overflow<nsw, nuw> : vector<4xindex>
          %247 = vector.gather %53[%c0, %c0, %c0] [%246], %228, %cst_23 : memref<1316x1x128xf16, strided<[128, 128, 1], offset: ?>>, vector<4xindex>, vector<4xi1>, vector<4xf16> into vector<4xf16>
          %248 = arith.addi %243, %cst_8 overflow<nsw, nuw> : vector<4xindex>
          %249 = vector.gather %53[%c0, %c0, %c0] [%248], %239, %cst_23 : memref<1316x1x128xf16, strided<[128, 128, 1], offset: ?>>, vector<4xindex>, vector<4xi1>, vector<4xf16> into vector<4xf16>
          %250 = arith.addi %236, %cst_15 overflow<nsw, nuw> : vector<4xindex>
          %251 = vector.gather %53[%c0, %c0, %c0] [%250], %228, %cst_23 : memref<1316x1x128xf16, strided<[128, 128, 1], offset: ?>>, vector<4xindex>, vector<4xi1>, vector<4xf16> into vector<4xf16>
          %252 = arith.addi %243, %cst_7 overflow<nsw, nuw> : vector<4xindex>
          %253 = vector.gather %53[%c0, %c0, %c0] [%252], %239, %cst_23 : memref<1316x1x128xf16, strided<[128, 128, 1], offset: ?>>, vector<4xindex>, vector<4xi1>, vector<4xf16> into vector<4xf16>
          %254 = arith.addi %236, %cst_14 overflow<nsw, nuw> : vector<4xindex>
          %255 = vector.gather %53[%c0, %c0, %c0] [%254], %228, %cst_23 : memref<1316x1x128xf16, strided<[128, 128, 1], offset: ?>>, vector<4xindex>, vector<4xi1>, vector<4xf16> into vector<4xf16>
          %256 = arith.addi %243, %cst_6 overflow<nsw, nuw> : vector<4xindex>
          %257 = vector.gather %53[%c0, %c0, %c0] [%256], %239, %cst_23 : memref<1316x1x128xf16, strided<[128, 128, 1], offset: ?>>, vector<4xindex>, vector<4xi1>, vector<4xf16> into vector<4xf16>
          vector.store %237, %alloc[%c0, %67, %32] : memref<1x64x36xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          vector.store %245, %alloc[%c0, %67, %34] : memref<1x64x36xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          vector.store %247, %alloc[%c0, %68, %32] : memref<1x64x36xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          vector.store %249, %alloc[%c0, %68, %34] : memref<1x64x36xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          vector.store %251, %alloc[%c0, %69, %32] : memref<1x64x36xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          vector.store %253, %alloc[%c0, %69, %34] : memref<1x64x36xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          vector.store %255, %alloc[%c0, %70, %32] : memref<1x64x36xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          vector.store %257, %alloc[%c0, %70, %34] : memref<1x64x36xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          amdgpu.lds_barrier
          %258 = vector.load %alloc[%c0, %67, %32] : memref<1x64x36xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %259 = vector.load %alloc[%c0, %67, %34] : memref<1x64x36xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %260 = vector.load %alloc[%c0, %68, %32] : memref<1x64x36xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %261 = vector.load %alloc[%c0, %68, %34] : memref<1x64x36xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %262 = vector.load %alloc[%c0, %69, %32] : memref<1x64x36xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %263 = vector.load %alloc[%c0, %69, %34] : memref<1x64x36xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %264 = vector.load %alloc[%c0, %70, %32] : memref<1x64x36xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %265 = vector.load %alloc[%c0, %70, %34] : memref<1x64x36xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %266 = vector.load %alloc_30[%c0, %22, %32] : memref<1x32x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %267 = vector.load %alloc_30[%c0, %22, %34] : memref<1x32x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %268 = vector.load %alloc_30[%c0, %22, %36] : memref<1x32x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %269 = vector.load %alloc_30[%c0, %22, %38] : memref<1x32x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %270 = vector.load %alloc_30[%c0, %22, %40] : memref<1x32x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %271 = vector.load %alloc_30[%c0, %22, %42] : memref<1x32x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %272 = vector.load %alloc_30[%c0, %22, %44] : memref<1x32x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %273 = vector.load %alloc_30[%c0, %22, %46] : memref<1x32x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %274 = vector.load %alloc_30[%c0, %61, %32] : memref<1x32x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %275 = vector.load %alloc_30[%c0, %61, %34] : memref<1x32x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %276 = vector.load %alloc_30[%c0, %61, %36] : memref<1x32x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %277 = vector.load %alloc_30[%c0, %61, %38] : memref<1x32x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %278 = vector.load %alloc_30[%c0, %61, %40] : memref<1x32x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %279 = vector.load %alloc_30[%c0, %61, %42] : memref<1x32x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %280 = vector.load %alloc_30[%c0, %61, %44] : memref<1x32x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %281 = vector.load %alloc_30[%c0, %61, %46] : memref<1x32x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %282 = arith.cmpi slt, %176, %56 : vector<4xindex>
          %283 = arith.cmpi slt, %202, %56 : vector<4xindex>
          %284 = arith.select %282, %cst_22, %cst_23 : vector<4xi1>, vector<4xf16>
          %285 = arith.select %283, %cst_22, %cst_23 : vector<4xi1>, vector<4xf16>
          %286 = arith.mulf %266, %284 : vector<4xf16>
          %287 = arith.mulf %267, %284 : vector<4xf16>
          %288 = arith.mulf %268, %284 : vector<4xf16>
          %289 = arith.mulf %269, %284 : vector<4xf16>
          %290 = arith.mulf %270, %284 : vector<4xf16>
          %291 = arith.mulf %271, %284 : vector<4xf16>
          %292 = arith.mulf %272, %284 : vector<4xf16>
          %293 = arith.mulf %273, %284 : vector<4xf16>
          %294 = arith.mulf %274, %285 : vector<4xf16>
          %295 = arith.mulf %275, %285 : vector<4xf16>
          %296 = arith.mulf %276, %285 : vector<4xf16>
          %297 = arith.mulf %277, %285 : vector<4xf16>
          %298 = arith.mulf %278, %285 : vector<4xf16>
          %299 = arith.mulf %279, %285 : vector<4xf16>
          %300 = arith.mulf %280, %285 : vector<4xf16>
          %301 = arith.mulf %281, %285 : vector<4xf16>
          %302 = amdgpu.mfma %286 * %33 + %cst_29 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %303 = amdgpu.mfma %287 * %35 + %302 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %304 = amdgpu.mfma %288 * %37 + %303 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %305 = amdgpu.mfma %289 * %39 + %304 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %306 = amdgpu.mfma %290 * %41 + %305 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %307 = amdgpu.mfma %291 * %43 + %306 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %308 = amdgpu.mfma %292 * %45 + %307 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %309 = amdgpu.mfma %293 * %47 + %308 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %310 = amdgpu.mfma %294 * %33 + %cst_29 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %311 = amdgpu.mfma %295 * %35 + %310 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %312 = amdgpu.mfma %296 * %37 + %311 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %313 = amdgpu.mfma %297 * %39 + %312 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %314 = amdgpu.mfma %298 * %41 + %313 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %315 = amdgpu.mfma %299 * %43 + %314 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %316 = amdgpu.mfma %300 * %45 + %315 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %317 = amdgpu.mfma %301 * %47 + %316 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %318 = arith.divf %309, %cst_25 : vector<4xf32>
          %319 = arith.divf %317, %cst_25 : vector<4xf32>
          %320 = math.tanh %318 : vector<4xf32>
          %321 = math.tanh %319 : vector<4xf32>
          %322 = arith.mulf %320, %cst_25 : vector<4xf32>
          %323 = arith.mulf %321, %cst_25 : vector<4xf32>
          %324 = arith.select %282, %cst_29, %cst_26 : vector<4xi1>, vector<4xf32>
          %325 = arith.select %283, %cst_29, %cst_26 : vector<4xi1>, vector<4xf32>
          %326 = arith.addf %322, %324 : vector<4xf32>
          %327 = arith.addf %323, %325 : vector<4xf32>
          %328 = vector.extract_strided_slice %326 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
          %329 = vector.extract_strided_slice %326 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
          %330 = arith.maximumf %328, %329 : vector<1xf32>
          %331 = vector.extract_strided_slice %326 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
          %332 = arith.maximumf %330, %331 : vector<1xf32>
          %333 = vector.extract_strided_slice %326 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
          %334 = arith.maximumf %332, %333 : vector<1xf32>
          %335 = vector.extract_strided_slice %327 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
          %336 = vector.extract_strided_slice %327 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
          %337 = arith.maximumf %335, %336 : vector<1xf32>
          %338 = vector.extract_strided_slice %327 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
          %339 = arith.maximumf %337, %338 : vector<1xf32>
          %340 = vector.extract_strided_slice %327 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
          %341 = arith.maximumf %339, %340 : vector<1xf32>
          %342 = arith.maximumf %334, %341 : vector<1xf32>
          %343 = vector.extract %342[0] : f32 from vector<1xf32>
          %shuffleResult, %valid = gpu.shuffle  xor %343, %c16_i32, %c64_i32 : f32
          %344 = vector.broadcast %shuffleResult : f32 to vector<1xf32>
          %345 = arith.maximumf %342, %344 : vector<1xf32>
          %346 = vector.extract %345[0] : f32 from vector<1xf32>
          %shuffleResult_33, %valid_34 = gpu.shuffle  xor %346, %c32_i32, %c64_i32 : f32
          %347 = vector.broadcast %shuffleResult_33 : f32 to vector<1xf32>
          %348 = arith.maximumf %345, %347 : vector<1xf32>
          %349 = arith.maximumf %arg14, %348 : vector<1xf32>
          %350 = arith.subf %arg14, %349 : vector<1xf32>
          %351 = math.exp2 %350 : vector<1xf32>
          %352 = vector.extract %349[0] : f32 from vector<1xf32>
          %353 = vector.splat %352 : vector<4xf32>
          %354 = arith.subf %326, %353 : vector<4xf32>
          %355 = arith.subf %327, %353 : vector<4xf32>
          %356 = math.exp2 %354 : vector<4xf32>
          %357 = math.exp2 %355 : vector<4xf32>
          %358 = arith.mulf %arg15, %351 : vector<1xf32>
          %359 = arith.addf %356, %357 : vector<4xf32>
          %360 = vector.extract_strided_slice %359 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
          %361 = vector.extract_strided_slice %359 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
          %362 = arith.addf %360, %361 : vector<1xf32>
          %363 = vector.extract_strided_slice %359 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
          %364 = arith.addf %362, %363 : vector<1xf32>
          %365 = vector.extract_strided_slice %359 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
          %366 = arith.addf %364, %365 : vector<1xf32>
          %367 = vector.extract %366[0] : f32 from vector<1xf32>
          %shuffleResult_35, %valid_36 = gpu.shuffle  xor %367, %c16_i32, %c64_i32 : f32
          %368 = vector.broadcast %shuffleResult_35 : f32 to vector<1xf32>
          %369 = arith.addf %366, %368 : vector<1xf32>
          %370 = vector.extract %369[0] : f32 from vector<1xf32>
          %shuffleResult_37, %valid_38 = gpu.shuffle  xor %370, %c32_i32, %c64_i32 : f32
          %371 = vector.broadcast %shuffleResult_37 : f32 to vector<1xf32>
          %372 = arith.addf %369, %371 : vector<1xf32>
          %373 = arith.addf %358, %372 : vector<1xf32>
          %374 = arith.truncf %356 : vector<4xf32> to vector<4xf16>
          %375 = arith.truncf %357 : vector<4xf32> to vector<4xf16>
          %376 = arith.mulf %258, %284 : vector<4xf16>
          %377 = arith.mulf %259, %285 : vector<4xf16>
          %378 = arith.mulf %260, %284 : vector<4xf16>
          %379 = arith.mulf %261, %285 : vector<4xf16>
          %380 = arith.mulf %262, %284 : vector<4xf16>
          %381 = arith.mulf %263, %285 : vector<4xf16>
          %382 = arith.mulf %264, %284 : vector<4xf16>
          %383 = arith.mulf %265, %285 : vector<4xf16>
          %384 = vector.extract %351[0] : f32 from vector<1xf32>
          %385 = vector.splat %384 : vector<4xf32>
          %386 = arith.mulf %arg16, %385 : vector<4xf32>
          %387 = arith.mulf %arg17, %385 : vector<4xf32>
          %388 = arith.mulf %arg18, %385 : vector<4xf32>
          %389 = arith.mulf %arg19, %385 : vector<4xf32>
          %390 = amdgpu.mfma %376 * %374 + %386 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %391 = amdgpu.mfma %377 * %375 + %390 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %392 = amdgpu.mfma %378 * %374 + %387 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %393 = amdgpu.mfma %379 * %375 + %392 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %394 = amdgpu.mfma %380 * %374 + %388 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %395 = amdgpu.mfma %381 * %375 + %394 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %396 = amdgpu.mfma %382 * %374 + %389 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %397 = amdgpu.mfma %383 * %375 + %396 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          scf.yield %349, %373, %391, %393, %395, %397 : vector<1xf32>, vector<1xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>
        }
        %alloc_31 = memref.alloc() : memref<1x64x36xf16, #gpu.address_space<workgroup>>
        %alloc_32 = memref.alloc() : memref<1x32x132xf16, #gpu.address_space<workgroup>>
        %72 = vector.maskedload %18[%27, %29, %32], %26, %cst_23 : memref<?x16x128xf16, strided<[2048, 128, 1], offset: ?>>, vector<4xi1>, vector<4xf16> into vector<4xf16>
        %73 = vector.maskedload %18[%27, %29, %34], %26, %cst_23 : memref<?x16x128xf16, strided<[2048, 128, 1], offset: ?>>, vector<4xi1>, vector<4xf16> into vector<4xf16>
        %74 = vector.maskedload %18[%27, %29, %36], %26, %cst_23 : memref<?x16x128xf16, strided<[2048, 128, 1], offset: ?>>, vector<4xi1>, vector<4xf16> into vector<4xf16>
        %75 = vector.maskedload %18[%27, %29, %38], %26, %cst_23 : memref<?x16x128xf16, strided<[2048, 128, 1], offset: ?>>, vector<4xi1>, vector<4xf16> into vector<4xf16>
        %76 = vector.maskedload %18[%27, %29, %40], %26, %cst_23 : memref<?x16x128xf16, strided<[2048, 128, 1], offset: ?>>, vector<4xi1>, vector<4xf16> into vector<4xf16>
        %77 = vector.maskedload %18[%27, %29, %42], %26, %cst_23 : memref<?x16x128xf16, strided<[2048, 128, 1], offset: ?>>, vector<4xi1>, vector<4xf16> into vector<4xf16>
        %78 = vector.maskedload %18[%27, %29, %44], %26, %cst_23 : memref<?x16x128xf16, strided<[2048, 128, 1], offset: ?>>, vector<4xi1>, vector<4xf16> into vector<4xf16>
        %79 = vector.maskedload %18[%27, %29, %46], %26, %cst_23 : memref<?x16x128xf16, strided<[2048, 128, 1], offset: ?>>, vector<4xi1>, vector<4xf16> into vector<4xf16>
        %80 = arith.subi %12, %c1 : index
        %81 = arith.divui %80, %c32 : index
        %82 = arith.addi %81, %c1 : index
        %83 = arith.cmpi eq, %12, %c0 : index
        %84 = arith.select %83, %c0, %82 : index
        %85 = stream.binding.subspan %arg2[%c0] : !stream.binding -> memref<?x1x128xf16, strided<[128, 128, 1], offset: ?>>{%arg12}
        %86 = stream.binding.subspan %arg1[%c0] : !stream.binding -> memref<?x1x128xf16, strided<[128, 128, 1], offset: ?>>{%arg12}
        %87 = arith.muli %thread_id_y, %c16 overflow<nsw, nuw> : index
        %88 = arith.muli %thread_id_z, %c16 overflow<nsw, nuw> : index
        %89 = arith.divsi %thread_id_x, %c16 : index
        %90 = arith.addi %89, %88 overflow<nsw, nuw> : index
        %91 = arith.addi %90, %87 overflow<nsw, nuw> : index
        %92 = arith.remsi %91, %c32 : index
        %93 = arith.divsi %29, %c16 : index
        %94 = arith.muli %22, %c8 overflow<nsw, nuw> : index
        %95 = arith.addi %91, %c16 overflow<nsw, nuw> : index
        %96 = arith.remsi %95, %c32 : index
        %97 = arith.remsi %thread_id_x, %c4 : index
        %98 = arith.muli %97, %c8 overflow<nsw, nuw> : index
        %99 = vector.splat %12 : vector<8xindex>
        %100 = arith.muli %workgroup_id_1, %c64 overflow<nsw, nuw> : index
        %101 = arith.divsi %thread_id_x, %c4 : index
        %102 = arith.remsi %101, %c64 : index
        %103 = arith.addi %102, %100 overflow<nsw, nuw> : index
        %104 = arith.muli %thread_id_y, %c64 overflow<nsw, nuw> : index
        %105 = arith.addi %22, %104 overflow<nsw, nuw> : index
        %106 = arith.addi %105, %c16 overflow<nsw, nuw> : index
        %107 = arith.addi %105, %c32 overflow<nsw, nuw> : index
        %108 = arith.addi %105, %c48 overflow<nsw, nuw> : index
        %109 = arith.addi %22, %c16 overflow<nsw, nuw> : index
        %110 = vector.step : vector<4xindex>
        %111 = vector.splat %12 : vector<4xindex>
        %112 = vector.step : vector<1xindex>
        %113 = arith.muli %112, %cst : vector<1xindex>
        %114 = vector.splat %24 : vector<1xindex>
        %115 = arith.addi %113, %114 : vector<1xindex>
        %116 = arith.index_cast %115 : vector<1xindex> to vector<1xi32>
        %117 = vector.extract %116[0] : i32 from vector<1xi32>
        %118 = vector.splat %117 : vector<4xi32>
        %119:6 = scf.for %arg13 = %c0 to %84 step %c1 iter_args(%arg14 = %71#0, %arg15 = %71#1, %arg16 = %71#2, %arg17 = %71#3, %arg18 = %71#4, %arg19 = %71#5) -> (vector<1xf32>, vector<1xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>) {
          amdgpu.lds_barrier
          %166 = arith.muli %arg13, %c32 overflow<nsw, nuw> : index
          %167 = arith.addi %92, %166 overflow<nsw, nuw> : index
          %168 = arith.cmpi slt, %167, %12 : index
          %169 = vector.splat %168 : vector<8xi1>
          %170 = arith.addi %167, %8 overflow<nsw, nuw> : index
          %171 = vector.maskedload %86[%170, %93, %94], %169, %cst_0 : memref<?x1x128xf16, strided<[128, 128, 1], offset: ?>>, vector<8xi1>, vector<8xf16> into vector<8xf16>
          amdgpu.lds_barrier
          vector.store %171, %alloc_32[%c0, %92, %94] : memref<1x32x132xf16, #gpu.address_space<workgroup>>, vector<8xf16>
          %172 = arith.addi %96, %166 overflow<nsw, nuw> : index
          %173 = arith.cmpi slt, %172, %12 : index
          %174 = vector.splat %173 : vector<8xi1>
          %175 = arith.addi %172, %8 overflow<nsw, nuw> : index
          %176 = vector.maskedload %86[%175, %93, %94], %174, %cst_0 : memref<?x1x128xf16, strided<[128, 128, 1], offset: ?>>, vector<8xi1>, vector<8xf16> into vector<8xf16>
          vector.store %176, %alloc_32[%c0, %96, %94] : memref<1x32x132xf16, #gpu.address_space<workgroup>>, vector<8xf16>
          %177 = arith.addi %166, %98 overflow<nsw, nuw> : index
          %178 = vector.splat %177 : vector<8xindex>
          %179 = arith.addi %178, %cst_21 overflow<nsw, nuw> : vector<8xindex>
          %180 = arith.cmpi slt, %179, %99 : vector<8xindex>
          %181 = arith.addi %177, %8 overflow<nsw, nuw> : index
          %182 = vector.gather %85[%181, %93, %103] [%cst_20], %180, %cst_0 : memref<?x1x128xf16, strided<[128, 128, 1], offset: ?>>, vector<8xindex>, vector<8xi1>, vector<8xf16> into vector<8xf16>
          vector.store %182, %alloc_31[%c0, %102, %98] : memref<1x64x36xf16, #gpu.address_space<workgroup>>, vector<8xf16>
          amdgpu.lds_barrier
          %183 = vector.load %alloc_31[%c0, %105, %32] : memref<1x64x36xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %184 = vector.load %alloc_31[%c0, %105, %34] : memref<1x64x36xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %185 = vector.load %alloc_31[%c0, %106, %32] : memref<1x64x36xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %186 = vector.load %alloc_31[%c0, %106, %34] : memref<1x64x36xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %187 = vector.load %alloc_31[%c0, %107, %32] : memref<1x64x36xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %188 = vector.load %alloc_31[%c0, %107, %34] : memref<1x64x36xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %189 = vector.load %alloc_31[%c0, %108, %32] : memref<1x64x36xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %190 = vector.load %alloc_31[%c0, %108, %34] : memref<1x64x36xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %191 = vector.load %alloc_32[%c0, %22, %32] : memref<1x32x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %192 = vector.load %alloc_32[%c0, %22, %34] : memref<1x32x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %193 = vector.load %alloc_32[%c0, %22, %36] : memref<1x32x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %194 = vector.load %alloc_32[%c0, %22, %38] : memref<1x32x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %195 = vector.load %alloc_32[%c0, %22, %40] : memref<1x32x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %196 = vector.load %alloc_32[%c0, %22, %42] : memref<1x32x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %197 = vector.load %alloc_32[%c0, %22, %44] : memref<1x32x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %198 = vector.load %alloc_32[%c0, %22, %46] : memref<1x32x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %199 = vector.load %alloc_32[%c0, %109, %32] : memref<1x32x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %200 = vector.load %alloc_32[%c0, %109, %34] : memref<1x32x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %201 = vector.load %alloc_32[%c0, %109, %36] : memref<1x32x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %202 = vector.load %alloc_32[%c0, %109, %38] : memref<1x32x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %203 = vector.load %alloc_32[%c0, %109, %40] : memref<1x32x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %204 = vector.load %alloc_32[%c0, %109, %42] : memref<1x32x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %205 = vector.load %alloc_32[%c0, %109, %44] : memref<1x32x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %206 = vector.load %alloc_32[%c0, %109, %46] : memref<1x32x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %207 = amdgpu.mfma %191 * %72 + %cst_29 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %208 = amdgpu.mfma %192 * %73 + %207 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %209 = amdgpu.mfma %193 * %74 + %208 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %210 = amdgpu.mfma %194 * %75 + %209 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %211 = amdgpu.mfma %195 * %76 + %210 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %212 = amdgpu.mfma %196 * %77 + %211 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %213 = amdgpu.mfma %197 * %78 + %212 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %214 = amdgpu.mfma %198 * %79 + %213 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %215 = amdgpu.mfma %199 * %72 + %cst_29 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %216 = amdgpu.mfma %200 * %73 + %215 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %217 = amdgpu.mfma %201 * %74 + %216 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %218 = amdgpu.mfma %202 * %75 + %217 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %219 = amdgpu.mfma %203 * %76 + %218 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %220 = amdgpu.mfma %204 * %77 + %219 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %221 = amdgpu.mfma %205 * %78 + %220 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %222 = amdgpu.mfma %206 * %79 + %221 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %223 = arith.divf %214, %cst_25 : vector<4xf32>
          %224 = arith.divf %222, %cst_25 : vector<4xf32>
          %225 = math.tanh %223 : vector<4xf32>
          %226 = math.tanh %224 : vector<4xf32>
          %227 = arith.mulf %225, %cst_25 : vector<4xf32>
          %228 = arith.mulf %226, %cst_25 : vector<4xf32>
          %229 = arith.addi %166, %32 overflow<nsw, nuw> : index
          %230 = vector.splat %229 : vector<4xindex>
          %231 = arith.addi %110, %230 : vector<4xindex>
          %232 = arith.index_cast %231 : vector<4xindex> to vector<4xi32>
          %233 = arith.addi %229, %c16 overflow<nsw, nuw> : index
          %234 = vector.splat %233 : vector<4xindex>
          %235 = arith.addi %110, %234 : vector<4xindex>
          %236 = arith.index_cast %235 : vector<4xindex> to vector<4xi32>
          %237 = arith.cmpi slt, %231, %111 : vector<4xindex>
          %238 = arith.cmpi sge, %118, %232 : vector<4xi32>
          %239 = arith.cmpi sge, %118, %236 : vector<4xi32>
          %240 = arith.andi %238, %237 : vector<4xi1>
          %241 = arith.andi %239, %237 : vector<4xi1>
          %242 = arith.select %240, %cst_29, %cst_26 : vector<4xi1>, vector<4xf32>
          %243 = arith.select %241, %cst_29, %cst_26 : vector<4xi1>, vector<4xf32>
          %244 = arith.addf %227, %242 : vector<4xf32>
          %245 = arith.addf %228, %243 : vector<4xf32>
          %246 = vector.extract_strided_slice %244 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
          %247 = vector.extract_strided_slice %244 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
          %248 = arith.maximumf %246, %247 : vector<1xf32>
          %249 = vector.extract_strided_slice %244 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
          %250 = arith.maximumf %248, %249 : vector<1xf32>
          %251 = vector.extract_strided_slice %244 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
          %252 = arith.maximumf %250, %251 : vector<1xf32>
          %253 = vector.extract_strided_slice %245 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
          %254 = vector.extract_strided_slice %245 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
          %255 = arith.maximumf %253, %254 : vector<1xf32>
          %256 = vector.extract_strided_slice %245 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
          %257 = arith.maximumf %255, %256 : vector<1xf32>
          %258 = vector.extract_strided_slice %245 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
          %259 = arith.maximumf %257, %258 : vector<1xf32>
          %260 = arith.maximumf %252, %259 : vector<1xf32>
          %261 = vector.extract %260[0] : f32 from vector<1xf32>
          %shuffleResult, %valid = gpu.shuffle  xor %261, %c16_i32, %c64_i32 : f32
          %262 = vector.broadcast %shuffleResult : f32 to vector<1xf32>
          %263 = arith.maximumf %260, %262 : vector<1xf32>
          %264 = vector.extract %263[0] : f32 from vector<1xf32>
          %shuffleResult_33, %valid_34 = gpu.shuffle  xor %264, %c32_i32, %c64_i32 : f32
          %265 = vector.broadcast %shuffleResult_33 : f32 to vector<1xf32>
          %266 = arith.maximumf %263, %265 : vector<1xf32>
          %267 = arith.maximumf %arg14, %266 : vector<1xf32>
          %268 = arith.subf %arg14, %267 : vector<1xf32>
          %269 = math.exp2 %268 : vector<1xf32>
          %270 = vector.extract %267[0] : f32 from vector<1xf32>
          %271 = vector.splat %270 : vector<4xf32>
          %272 = arith.subf %244, %271 : vector<4xf32>
          %273 = arith.subf %245, %271 : vector<4xf32>
          %274 = math.exp2 %272 : vector<4xf32>
          %275 = math.exp2 %273 : vector<4xf32>
          %276 = arith.mulf %arg15, %269 : vector<1xf32>
          %277 = arith.addf %274, %275 : vector<4xf32>
          %278 = vector.extract_strided_slice %277 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
          %279 = vector.extract_strided_slice %277 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
          %280 = arith.addf %278, %279 : vector<1xf32>
          %281 = vector.extract_strided_slice %277 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
          %282 = arith.addf %280, %281 : vector<1xf32>
          %283 = vector.extract_strided_slice %277 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
          %284 = arith.addf %282, %283 : vector<1xf32>
          %285 = vector.extract %284[0] : f32 from vector<1xf32>
          %shuffleResult_35, %valid_36 = gpu.shuffle  xor %285, %c16_i32, %c64_i32 : f32
          %286 = vector.broadcast %shuffleResult_35 : f32 to vector<1xf32>
          %287 = arith.addf %284, %286 : vector<1xf32>
          %288 = vector.extract %287[0] : f32 from vector<1xf32>
          %shuffleResult_37, %valid_38 = gpu.shuffle  xor %288, %c32_i32, %c64_i32 : f32
          %289 = vector.broadcast %shuffleResult_37 : f32 to vector<1xf32>
          %290 = arith.addf %287, %289 : vector<1xf32>
          %291 = arith.addf %276, %290 : vector<1xf32>
          %292 = arith.truncf %274 : vector<4xf32> to vector<4xf16>
          %293 = arith.truncf %275 : vector<4xf32> to vector<4xf16>
          %294 = vector.extract %269[0] : f32 from vector<1xf32>
          %295 = vector.splat %294 : vector<4xf32>
          %296 = arith.mulf %arg16, %295 : vector<4xf32>
          %297 = arith.mulf %arg17, %295 : vector<4xf32>
          %298 = arith.mulf %arg18, %295 : vector<4xf32>
          %299 = arith.mulf %arg19, %295 : vector<4xf32>
          %300 = amdgpu.mfma %183 * %292 + %296 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %301 = amdgpu.mfma %184 * %293 + %300 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %302 = amdgpu.mfma %185 * %292 + %297 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %303 = amdgpu.mfma %186 * %293 + %302 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %304 = amdgpu.mfma %187 * %292 + %298 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %305 = amdgpu.mfma %188 * %293 + %304 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %306 = amdgpu.mfma %189 * %292 + %299 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %307 = amdgpu.mfma %190 * %293 + %306 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          scf.yield %267, %291, %301, %303, %305, %307 : vector<1xf32>, vector<1xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>
        }
        %120 = arith.divf %cst_19, %119#1 : vector<1xf32>
        %121 = vector.extract %120[0] : f32 from vector<1xf32>
        %122 = vector.splat %121 : vector<4xf32>
        %123 = arith.mulf %119#2, %122 : vector<4xf32>
        %124 = arith.mulf %119#3, %122 : vector<4xf32>
        %125 = arith.mulf %119#4, %122 : vector<4xf32>
        %126 = arith.mulf %119#5, %122 : vector<4xf32>
        %127 = vector.extract_strided_slice %123 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %128 = stream.binding.subspan %arg10[%c0] : !stream.binding -> memref<?x16x128xf32, strided<[2048, 128, 1], offset: ?>>{%arg11}
        %129 = vector.splat %24 : vector<1xindex>
        %130 = vector.splat %12 : vector<1xindex>
        %131 = arith.cmpi slt, %129, %130 : vector<1xindex>
        %132 = arith.muli %thread_id_y, %c64 overflow<nsw, nuw> : index
        %133 = arith.muli %workgroup_id_1, %c64 overflow<nsw, nuw> : index
        %134 = arith.addi %133, %132 overflow<nsw, nuw> : index
        %135 = arith.addi %134, %32 overflow<nsw, nuw> : index
        vector.maskedstore %128[%27, %29, %135], %131, %127 : memref<?x16x128xf32, strided<[2048, 128, 1], offset: ?>>, vector<1xi1>, vector<1xf32>
        %136 = vector.extract_strided_slice %123 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %137 = arith.addi %135, %c1 overflow<nsw, nuw> : index
        vector.maskedstore %128[%27, %29, %137], %131, %136 : memref<?x16x128xf32, strided<[2048, 128, 1], offset: ?>>, vector<1xi1>, vector<1xf32>
        %138 = vector.extract_strided_slice %123 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %139 = arith.addi %135, %c2 overflow<nsw, nuw> : index
        vector.maskedstore %128[%27, %29, %139], %131, %138 : memref<?x16x128xf32, strided<[2048, 128, 1], offset: ?>>, vector<1xi1>, vector<1xf32>
        %140 = vector.extract_strided_slice %123 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %141 = arith.addi %135, %c3 overflow<nsw, nuw> : index
        vector.maskedstore %128[%27, %29, %141], %131, %140 : memref<?x16x128xf32, strided<[2048, 128, 1], offset: ?>>, vector<1xi1>, vector<1xf32>
        %142 = vector.extract_strided_slice %124 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %143 = arith.addi %135, %c16 overflow<nsw, nuw> : index
        vector.maskedstore %128[%27, %29, %143], %131, %142 : memref<?x16x128xf32, strided<[2048, 128, 1], offset: ?>>, vector<1xi1>, vector<1xf32>
        %144 = vector.extract_strided_slice %124 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %145 = arith.addi %135, %c17 overflow<nsw, nuw> : index
        vector.maskedstore %128[%27, %29, %145], %131, %144 : memref<?x16x128xf32, strided<[2048, 128, 1], offset: ?>>, vector<1xi1>, vector<1xf32>
        %146 = vector.extract_strided_slice %124 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %147 = arith.addi %135, %c18 overflow<nsw, nuw> : index
        vector.maskedstore %128[%27, %29, %147], %131, %146 : memref<?x16x128xf32, strided<[2048, 128, 1], offset: ?>>, vector<1xi1>, vector<1xf32>
        %148 = vector.extract_strided_slice %124 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %149 = arith.addi %135, %c19 overflow<nsw, nuw> : index
        vector.maskedstore %128[%27, %29, %149], %131, %148 : memref<?x16x128xf32, strided<[2048, 128, 1], offset: ?>>, vector<1xi1>, vector<1xf32>
        %150 = vector.extract_strided_slice %125 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %151 = arith.addi %135, %c32 overflow<nsw, nuw> : index
        vector.maskedstore %128[%27, %29, %151], %131, %150 : memref<?x16x128xf32, strided<[2048, 128, 1], offset: ?>>, vector<1xi1>, vector<1xf32>
        %152 = vector.extract_strided_slice %125 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %153 = arith.addi %135, %c33 overflow<nsw, nuw> : index
        vector.maskedstore %128[%27, %29, %153], %131, %152 : memref<?x16x128xf32, strided<[2048, 128, 1], offset: ?>>, vector<1xi1>, vector<1xf32>
        %154 = vector.extract_strided_slice %125 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %155 = arith.addi %135, %c34 overflow<nsw, nuw> : index
        vector.maskedstore %128[%27, %29, %155], %131, %154 : memref<?x16x128xf32, strided<[2048, 128, 1], offset: ?>>, vector<1xi1>, vector<1xf32>
        %156 = vector.extract_strided_slice %125 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %157 = arith.addi %135, %c35 overflow<nsw, nuw> : index
        vector.maskedstore %128[%27, %29, %157], %131, %156 : memref<?x16x128xf32, strided<[2048, 128, 1], offset: ?>>, vector<1xi1>, vector<1xf32>
        %158 = vector.extract_strided_slice %126 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %159 = arith.addi %135, %c48 overflow<nsw, nuw> : index
        vector.maskedstore %128[%27, %29, %159], %131, %158 : memref<?x16x128xf32, strided<[2048, 128, 1], offset: ?>>, vector<1xi1>, vector<1xf32>
        %160 = vector.extract_strided_slice %126 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %161 = arith.addi %135, %c49 overflow<nsw, nuw> : index
        vector.maskedstore %128[%27, %29, %161], %131, %160 : memref<?x16x128xf32, strided<[2048, 128, 1], offset: ?>>, vector<1xi1>, vector<1xf32>
        %162 = vector.extract_strided_slice %126 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %163 = arith.addi %135, %c50 overflow<nsw, nuw> : index
        vector.maskedstore %128[%27, %29, %163], %131, %162 : memref<?x16x128xf32, strided<[2048, 128, 1], offset: ?>>, vector<1xi1>, vector<1xf32>
        %164 = vector.extract_strided_slice %126 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %165 = arith.addi %135, %c51 overflow<nsw, nuw> : index
        vector.maskedstore %128[%27, %29, %165], %131, %164 : memref<?x16x128xf32, strided<[2048, 128, 1], offset: ?>>, vector<1xi1>, vector<1xf32>
        return
      }
    }
  }
  func.func @isolated_benchmark(%arg0: tensor<?x16x128xf16>, %arg1: tensor<?x1x128xf16>, %arg2: tensor<?x1x128xf16>, %arg3: tensor<1316x1x128xf16>, %arg4: tensor<1316x1x128xf16>, %arg5: tensor<2x864xi32>, %arg6: tensor<2xi32>, %arg7: tensor<2xi32>, %arg8: tensor<2xi32>, %arg9: tensor<2xi32>, %arg10: tensor<?x16x128xf32>, %arg11: index, %arg12: index) -> tensor<?x16x128xf32> {
    %0 = flow.dispatch @extend_attention::@extend_attention[%arg11, %arg12](%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12) : (tensor<?x16x128xf16>{%arg11}, tensor<?x1x128xf16>{%arg12}, tensor<?x1x128xf16>{%arg12}, tensor<1316x1x128xf16>, tensor<1316x1x128xf16>, tensor<2x864xi32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>, tensor<?x16x128xf32>{%arg11}, index, index) -> %arg10{%arg11}
    return %0 : tensor<?x16x128xf32>
  }
}
