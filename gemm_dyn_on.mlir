#map = affine_map<()[s0] -> (s0 ceildiv 64)>
#map1 = affine_map<()[s0] -> (s0 ceildiv 32)>
#map2 = affine_map<()[s0, s1, s2] -> (s1 * 32 + s2 * 64 + s0 floordiv 4 - ((s1 * 32 + s0 floordiv 4) floordiv 64) * 64)>
#map3 = affine_map<()[s0, s1] -> ((s1 * 32 + s0 floordiv 4) mod 64)>
#map4 = affine_map<()[s0] -> (s0 * 8 - (s0 floordiv 4) * 32)>
#map5 = affine_map<()[s0, s1] -> (s0 + s1 * 32 - (s0 floordiv 16) * 16)>
#map6 = affine_map<()[s0] -> (((s0 mod 64) floordiv 16) * 4)>
#map7 = affine_map<()[s0] -> (((s0 mod 64) floordiv 16) * 4 + 16)>
#map8 = affine_map<()[s0, s1] -> (s0 + s1 * 32 - (s0 floordiv 16) * 16 + 16)>
#map9 = affine_map<()[s0] -> (s0 mod 16 + (s0 floordiv 64) * 32)>
#map10 = affine_map<()[s0] -> (s0 mod 16 + (s0 floordiv 64) * 32 + 16)>
#map11 = affine_map<()[s0, s1] -> (s0 * 32 + s1 * 8 - (s1 floordiv 4) * 32)>
#map12 = affine_map<()[s0, s1] -> (s0 * 64 + (s1 floordiv 64) * 32 + ((s1 mod 64) floordiv 16) * 4)>
#map13 = affine_map<()[s0, s1, s2] -> (s0 + s1 * 64 + s2 * 32 - (s0 floordiv 16) * 16)>
#map14 = affine_map<()[s0, s1] -> (s0 * 64 + (s1 floordiv 64) * 32 + ((s1 mod 64) floordiv 16) * 4 + 1)>
#map15 = affine_map<()[s0, s1] -> (s0 * 64 + (s1 floordiv 64) * 32 + ((s1 mod 64) floordiv 16) * 4 + 2)>
#map16 = affine_map<()[s0, s1] -> (s0 * 64 + (s1 floordiv 64) * 32 + ((s1 mod 64) floordiv 16) * 4 + 3)>
#map17 = affine_map<()[s0, s1, s2] -> (s0 + s1 * 64 + s2 * 32 - (s0 floordiv 16) * 16 + 16)>
#map18 = affine_map<()[s0, s1] -> (s0 * 64 + (s1 floordiv 64) * 32 + ((s1 mod 64) floordiv 16) * 4 + 16)>
#map19 = affine_map<()[s0, s1] -> (s0 * 64 + (s1 floordiv 64) * 32 + ((s1 mod 64) floordiv 16) * 4 + 17)>
#map20 = affine_map<()[s0, s1] -> (s0 * 64 + (s1 floordiv 64) * 32 + ((s1 mod 64) floordiv 16) * 4 + 18)>
#map21 = affine_map<()[s0, s1] -> (s0 * 64 + (s1 floordiv 64) * 32 + ((s1 mod 64) floordiv 16) * 4 + 19)>
#translation = #iree_codegen.translation_info<pipeline = None workgroup_size = [128, 2, 1] subgroup_size = 64>
module attributes {transform.with_named_sequence} {
  stream.executable private @gemm {
    stream.executable.export public @gemm workgroups(%arg0: index, %arg1: index, %arg2: index) -> (index, index, index) {
      %c1 = arith.constant 1 : index
      %0 = affine.apply #map()[%arg0]
      %1 = affine.apply #map()[%arg1]
      stream.return %0, %1, %c1 : index, index, index
    }
    builtin.module {
      func.func @gemm(%arg0: !stream.binding, %arg1: !stream.binding, %arg2: !stream.binding, %arg3: index, %arg4: index, %arg5: index) attributes {translation_info = #translation} {
        %cst = arith.constant dense<0.000000e+00> : vector<8xf16>
        %cst_0 = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7]> : vector<8xindex>
        %c1 = arith.constant 1 : index
        %c0 = arith.constant 0 : index
        %cst_1 = arith.constant dense<0.000000e+00> : vector<4xf32>
        %workgroup_id_0 = stream.dispatch.workgroup.id[0] : index
        %workgroup_id_1 = stream.dispatch.workgroup.id[1] : index
        %thread_id_x = gpu.thread_id  x
        %thread_id_y = gpu.thread_id  y
        %alloc = memref.alloc() : memref<64x36xf16, #gpu.address_space<workgroup>>
        %alloc_2 = memref.alloc() : memref<64x36xf16, #gpu.address_space<workgroup>>
        %0 = affine.apply #map1()[%arg5]
        %1 = stream.binding.subspan %arg1[%c0] : !stream.binding -> memref<?x?xf16, strided<[?, 1], offset: ?>>{%arg4, %arg5}
        %2 = stream.binding.subspan %arg0[%c0] : !stream.binding -> memref<?x?xf16, strided<[?, 1], offset: ?>>{%arg3, %arg5}
        %3 = affine.apply #map2()[%thread_id_x, %thread_id_y, %workgroup_id_0]
        %4 = vector.splat %arg5 : vector<8xindex>
        %5 = arith.cmpi slt, %3, %arg3 : index
        %6 = vector.splat %5 : vector<8xi1>
        %7 = affine.apply #map3()[%thread_id_x, %thread_id_y]
        %8 = affine.apply #map4()[%thread_id_x]
        %9 = affine.apply #map2()[%thread_id_x, %thread_id_y, %workgroup_id_1]
        %10 = arith.cmpi slt, %9, %arg4 : index
        %11 = vector.splat %10 : vector<8xi1>
        %12 = affine.apply #map5()[%thread_id_x, %thread_id_y]
        %13 = affine.apply #map6()[%thread_id_x]
        %14 = affine.apply #map7()[%thread_id_x]
        %15 = affine.apply #map8()[%thread_id_x, %thread_id_y]
        %16 = affine.apply #map9()[%thread_id_x]
        %17 = affine.apply #map10()[%thread_id_x]
        %18:4 = scf.for %arg6 = %c0 to %0 step %c1 iter_args(%arg7 = %cst_1, %arg8 = %cst_1, %arg9 = %cst_1, %arg10 = %cst_1) -> (vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>) {
          amdgpu.lds_barrier
          %88 = affine.apply #map11()[%arg6, %thread_id_x]
          %89 = vector.splat %88 : vector<8xindex>
          %90 = arith.addi %89, %cst_0 overflow<nsw, nuw> : vector<8xindex>
          %91 = arith.cmpi slt, %90, %4 : vector<8xindex>
          %92 = arith.andi %91, %6 : vector<8xi1>
          %93 = vector.maskedload %2[%3, %88], %92, %cst : memref<?x?xf16, strided<[?, 1], offset: ?>>, vector<8xi1>, vector<8xf16> into vector<8xf16>
          vector.store %93, %alloc_2[%7, %8] : memref<64x36xf16, #gpu.address_space<workgroup>>, vector<8xf16>
          %94 = arith.andi %91, %11 : vector<8xi1>
          %95 = vector.maskedload %1[%9, %88], %94, %cst : memref<?x?xf16, strided<[?, 1], offset: ?>>, vector<8xi1>, vector<8xf16> into vector<8xf16>
          vector.store %95, %alloc[%7, %8] : memref<64x36xf16, #gpu.address_space<workgroup>>, vector<8xf16>
          amdgpu.lds_barrier
          %96 = vector.load %alloc[%12, %13] : memref<64x36xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %97 = vector.load %alloc[%12, %14] : memref<64x36xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %98 = vector.load %alloc[%15, %13] : memref<64x36xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %99 = vector.load %alloc[%15, %14] : memref<64x36xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %100 = vector.load %alloc_2[%16, %13] : memref<64x36xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %101 = vector.load %alloc_2[%16, %14] : memref<64x36xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %102 = vector.load %alloc_2[%17, %13] : memref<64x36xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %103 = vector.load %alloc_2[%17, %14] : memref<64x36xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %104 = amdgpu.mfma %100 * %96 + %arg7 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %105 = amdgpu.mfma %101 * %97 + %104 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %106 = amdgpu.mfma %100 * %98 + %arg8 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %107 = amdgpu.mfma %101 * %99 + %106 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %108 = amdgpu.mfma %102 * %96 + %arg9 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %109 = amdgpu.mfma %103 * %97 + %108 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %110 = amdgpu.mfma %102 * %98 + %arg10 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %111 = amdgpu.mfma %103 * %99 + %110 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          scf.yield %105, %107, %109, %111 : vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>
        }
        %19 = vector.extract_strided_slice %18#0 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %20 = stream.binding.subspan %arg2[%c0] : !stream.binding -> memref<?x?xf32, strided<[?, 1], offset: ?>>{%arg3, %arg4}
        %21 = affine.apply #map12()[%workgroup_id_0, %thread_id_x]
        %22 = affine.apply #map13()[%thread_id_x, %workgroup_id_1, %thread_id_y]
        %23 = arith.cmpi slt, %22, %arg4 : index
        %24 = arith.cmpi slt, %21, %arg3 : index
        %25 = arith.andi %23, %24 : i1
        %26 = vector.splat %25 : vector<1xi1>
        vector.maskedstore %20[%21, %22], %26, %19 : memref<?x?xf32, strided<[?, 1], offset: ?>>, vector<1xi1>, vector<1xf32>
        %27 = vector.extract_strided_slice %18#0 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %28 = affine.apply #map14()[%workgroup_id_0, %thread_id_x]
        %29 = arith.cmpi slt, %28, %arg3 : index
        %30 = arith.andi %23, %29 : i1
        %31 = vector.splat %30 : vector<1xi1>
        vector.maskedstore %20[%28, %22], %31, %27 : memref<?x?xf32, strided<[?, 1], offset: ?>>, vector<1xi1>, vector<1xf32>
        %32 = vector.extract_strided_slice %18#0 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %33 = affine.apply #map15()[%workgroup_id_0, %thread_id_x]
        %34 = arith.cmpi slt, %33, %arg3 : index
        %35 = arith.andi %23, %34 : i1
        %36 = vector.splat %35 : vector<1xi1>
        vector.maskedstore %20[%33, %22], %36, %32 : memref<?x?xf32, strided<[?, 1], offset: ?>>, vector<1xi1>, vector<1xf32>
        %37 = vector.extract_strided_slice %18#0 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %38 = affine.apply #map16()[%workgroup_id_0, %thread_id_x]
        %39 = arith.cmpi slt, %38, %arg3 : index
        %40 = arith.andi %23, %39 : i1
        %41 = vector.splat %40 : vector<1xi1>
        vector.maskedstore %20[%38, %22], %41, %37 : memref<?x?xf32, strided<[?, 1], offset: ?>>, vector<1xi1>, vector<1xf32>
        %42 = vector.extract_strided_slice %18#1 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %43 = affine.apply #map17()[%thread_id_x, %workgroup_id_1, %thread_id_y]
        %44 = arith.cmpi slt, %43, %arg4 : index
        %45 = arith.andi %44, %24 : i1
        %46 = vector.splat %45 : vector<1xi1>
        vector.maskedstore %20[%21, %43], %46, %42 : memref<?x?xf32, strided<[?, 1], offset: ?>>, vector<1xi1>, vector<1xf32>
        %47 = vector.extract_strided_slice %18#1 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %48 = arith.andi %44, %29 : i1
        %49 = vector.splat %48 : vector<1xi1>
        vector.maskedstore %20[%28, %43], %49, %47 : memref<?x?xf32, strided<[?, 1], offset: ?>>, vector<1xi1>, vector<1xf32>
        %50 = vector.extract_strided_slice %18#1 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %51 = arith.andi %44, %34 : i1
        %52 = vector.splat %51 : vector<1xi1>
        vector.maskedstore %20[%33, %43], %52, %50 : memref<?x?xf32, strided<[?, 1], offset: ?>>, vector<1xi1>, vector<1xf32>
        %53 = vector.extract_strided_slice %18#1 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %54 = arith.andi %44, %39 : i1
        %55 = vector.splat %54 : vector<1xi1>
        vector.maskedstore %20[%38, %43], %55, %53 : memref<?x?xf32, strided<[?, 1], offset: ?>>, vector<1xi1>, vector<1xf32>
        %56 = vector.extract_strided_slice %18#2 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %57 = affine.apply #map18()[%workgroup_id_0, %thread_id_x]
        %58 = arith.cmpi slt, %57, %arg3 : index
        %59 = arith.andi %23, %58 : i1
        %60 = vector.splat %59 : vector<1xi1>
        vector.maskedstore %20[%57, %22], %60, %56 : memref<?x?xf32, strided<[?, 1], offset: ?>>, vector<1xi1>, vector<1xf32>
        %61 = vector.extract_strided_slice %18#2 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %62 = affine.apply #map19()[%workgroup_id_0, %thread_id_x]
        %63 = arith.cmpi slt, %62, %arg3 : index
        %64 = arith.andi %23, %63 : i1
        %65 = vector.splat %64 : vector<1xi1>
        vector.maskedstore %20[%62, %22], %65, %61 : memref<?x?xf32, strided<[?, 1], offset: ?>>, vector<1xi1>, vector<1xf32>
        %66 = vector.extract_strided_slice %18#2 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %67 = affine.apply #map20()[%workgroup_id_0, %thread_id_x]
        %68 = arith.cmpi slt, %67, %arg3 : index
        %69 = arith.andi %23, %68 : i1
        %70 = vector.splat %69 : vector<1xi1>
        vector.maskedstore %20[%67, %22], %70, %66 : memref<?x?xf32, strided<[?, 1], offset: ?>>, vector<1xi1>, vector<1xf32>
        %71 = vector.extract_strided_slice %18#2 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %72 = affine.apply #map21()[%workgroup_id_0, %thread_id_x]
        %73 = arith.cmpi slt, %72, %arg3 : index
        %74 = arith.andi %23, %73 : i1
        %75 = vector.splat %74 : vector<1xi1>
        vector.maskedstore %20[%72, %22], %75, %71 : memref<?x?xf32, strided<[?, 1], offset: ?>>, vector<1xi1>, vector<1xf32>
        %76 = vector.extract_strided_slice %18#3 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %77 = arith.andi %44, %58 : i1
        %78 = vector.splat %77 : vector<1xi1>
        vector.maskedstore %20[%57, %43], %78, %76 : memref<?x?xf32, strided<[?, 1], offset: ?>>, vector<1xi1>, vector<1xf32>
        %79 = vector.extract_strided_slice %18#3 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %80 = arith.andi %44, %63 : i1
        %81 = vector.splat %80 : vector<1xi1>
        vector.maskedstore %20[%62, %43], %81, %79 : memref<?x?xf32, strided<[?, 1], offset: ?>>, vector<1xi1>, vector<1xf32>
        %82 = vector.extract_strided_slice %18#3 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %83 = arith.andi %44, %68 : i1
        %84 = vector.splat %83 : vector<1xi1>
        vector.maskedstore %20[%67, %43], %84, %82 : memref<?x?xf32, strided<[?, 1], offset: ?>>, vector<1xi1>, vector<1xf32>
        %85 = vector.extract_strided_slice %18#3 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %86 = arith.andi %44, %73 : i1
        %87 = vector.splat %86 : vector<1xi1>
        vector.maskedstore %20[%72, %43], %87, %85 : memref<?x?xf32, strided<[?, 1], offset: ?>>, vector<1xi1>, vector<1xf32>
        return
      }
    }
  }
  func.func @isolated_benchmark(%arg0: tensor<?x?xf16>, %arg1: tensor<?x?xf16>, %arg2: tensor<?x?xf32>, %arg3: index, %arg4: index, %arg5: index) -> tensor<?x?xf32> {
    %0 = flow.dispatch @gemm::@gemm[%arg3, %arg4, %arg5](%arg0, %arg1, %arg2, %arg3, %arg4, %arg5) : (tensor<?x?xf16>{%arg3, %arg5}, tensor<?x?xf16>{%arg4, %arg5}, tensor<?x?xf32>{%arg3, %arg4}, index, index, index) -> %arg2{%arg3, %arg4}
    return %0 : tensor<?x?xf32>
  }
}
