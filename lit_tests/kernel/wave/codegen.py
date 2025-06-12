# RUN: python %s | FileCheck %s

import pytest
from typing import Callable
import iree.turbine.kernel as tk
import iree.turbine.kernel.lang as tkl
import iree.turbine.kernel.wave as tkw
from iree.turbine.kernel.lang.global_symbols import *
from iree.turbine.kernel.wave.compile import WaveCompileOptions, wave_compile
from iree.turbine.kernel.wave.utils.general_utils import (
    run_test,
)
from iree.turbine.kernel.wave.utils.compile_utils import (
    set_default_compile_config,
)
import sympy

M = tkl.sym.M
N = tkl.sym.N
K = tkl.sym.K
B = tkl.sym.B
BLOCK_M = tkl.sym.BLOCK_M
BLOCK_N = tkl.sym.BLOCK_N
BLOCK_K = tkl.sym.BLOCK_K
BLOCK_B = tkl.sym.BLOCK_B
LOAD_ELEMS_PER_THREAD = tkl.sym.LOAD_ELEMS_PER_THREAD
STORE_ELEMS_PER_THREAD = tkl.sym.STORE_ELEMS_PER_THREAD
ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
ADDRESS_SPACE_0 = tkl.sym.ADDRESS_SPACE_0


def get_wave_compile_options(
    canonicalize: bool = False, dynamic_symbols=[], additional_symbols={}
):
    bindings = {
        M: 16,
        N: 16,
        K: 16,
        BLOCK_M: 16,
        BLOCK_N: 16,
        BLOCK_K: 16,
        ADDRESS_SPACE: tkl.AddressSpace.SHARED_MEMORY.value,
    }
    bindings.update(additional_symbols)

    # Remove dynamic symbols from the bindings.
    for sym in dynamic_symbols:
        if sym in bindings:
            del bindings[sym]

    return WaveCompileOptions(
        subs=bindings,
        canonicalize=canonicalize,
        dynamic_symbols=dynamic_symbols,
        compile_to_mlir=True,
    )


@run_test
def test_read():
    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=64, waves_per_block=(1, 1, 1), vector_shapes={M: 16, N: 16}
        )
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N)]

    @tkw.wave(constraints)
    def read(a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16]):
        tkw.read(a)

    read = wave_compile(get_wave_compile_options(), read)
    print(read.asm)

    # CHECK-LABEL:    test_read
    # CHECK-DAG:        #[[MAP0:.*]] = affine_map<()[s0, s1, s2, s3, s4, s5] -> (s0 mod s1 + ((s2 * s3) floordiv s4) * s5)>
    # CHECK:          func.func @read
    # CHECK-SAME:       (%[[ARG0:[a-zA-Z0-9_]+]]: !stream.binding)
    # CHECK:            %[[WORKGROUP_ID_0:.+]] = stream.dispatch.workgroup.id[0] : index
    # CHECK:            %[[WORKGROUP_ID_1:.+]] = stream.dispatch.workgroup.id[1] : index
    # CHECK:            %[[WORKGROUP_ID_2:.+]] = stream.dispatch.workgroup.id[2] : index
    # CHECK-DAG:        %[[THREAD_ID_X:.+]] = gpu.thread_id  x
    # CHECK-DAG:        %[[THREAD_ID_Y:.+]] = gpu.thread_id  y
    # CHECK-DAG:        %[[THREAD_ID_Z:.+]] = gpu.thread_id  z
    # CHECK-DAG:        %[[C0:.+]] = arith.constant 0 : index
    # CHECK:            %[[D0:.+]] = stream.binding.subspan %[[ARG0]][%[[C0]]] : !stream.binding -> memref<16x16xf16,
    # CHECK-SAME:         strided<[16, 1], offset: ?>>
    # CHECK:            %[[I0:.*]] = affine.apply #[[MAP0]]()
    # CHECK:            %{{.*}} = vector.load %[[D0]][%[[I0]], %{{.*}}] : memref<16x16xf16, strided<[16, 1], offset: ?>>,
    # CHECK-SAME:         vector<16xf16>


@run_test
def test_read_mapped():
    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=64, waves_per_block=(1, 1, 1), vector_shapes={M: 16, N: 16}
        )
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N)]

    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)
    mapping = tkw.IndexMapping(
        num_iterators=2, inputs={N: i, M: j}, outputs={N: i, M: j}
    )

    @tkw.wave(constraints)
    def read_mapped(a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16]):
        tkw.read(a, mapping=mapping, elements_per_thread=16)

    read_mapped = wave_compile(get_wave_compile_options(), read_mapped)
    print(read_mapped.asm)

    # CHECK-LABEL:    test_read_mapped
    # CHECK-DAG:        #{{.*}} = affine_map<()[s0, s1, s2, s3, s4, s5, s6] -> (((s0 * s1) floordiv s2) * s3 + (s4 mod s5) * s6)>
    # CHECK:          func.func @read_mapped
    # CHECK-COUNT-16:   vector.maskedload
    # CHECK-COUNT-16:   vector.extract
    # CHECK:            vector.from_elements


@run_test
def test_read_mapped_buffer():
    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=64, waves_per_block=(1, 1, 1), vector_shapes={M: 16, N: 16}
        )
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N)]

    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)
    mapping = tkw.IndexMapping(
        num_iterators=2, inputs={N: i, M: j}, outputs={N: i, M: j}
    )

    @tkw.wave(constraints)
    def read_mapped_buffer(a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16]):
        tkw.read(a, mapping=mapping, elements_per_thread=16)

    options = WaveCompileOptions(
        subs={
            M: 16,
            N: 16,
            K: 16,
            BLOCK_M: 16,
            BLOCK_N: 16,
            BLOCK_K: 16,
            ADDRESS_SPACE: tkl.AddressSpace.SHARED_MEMORY.value,
        },
        use_buffer_load_ops=True,
        use_buffer_store_ops=True,
        compile_to_mlir=True,
    )
    read_mapped_buffer = wave_compile(options, read_mapped_buffer)
    print(read_mapped_buffer.asm)

    # CHECK-LABEL:    func.func @read_mapped_buffer
    # CHECK-COUNT-1:    amdgpu.fat_raw_buffer_cast [[VAL0:%[0-9a-zA-Z_]+]] cacheSwizzleStride([[VAL3:%[0-9a-zA-Z_]+]])


@run_test
def test_read_write():
    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=64, waves_per_block=(1, 1, 1), vector_shapes={M: 16, N: 16}
        )
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N)]

    @tkw.wave(constraints)
    def read_write(
        a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
    ):
        res = tkw.read(a)
        tkw.write(res, b)

    read_write = wave_compile(get_wave_compile_options(canonicalize=True), read_write)
    print(read_write.asm)

    # CHECK-LABEL:    test_read_write
    # CHECK-DAG:        #[[MAP0:.*]] = affine_map<()[s0] -> (s0 - (s0 floordiv 64) * 48)>
    # CHECK:          func.func @read_write
    # CHECK-SAME:       (%[[ARG0:.*]]: !stream.binding, %[[ARG1:.*]]: !stream.binding)
    # CHECK-DAG:        %[[C0:.*]] = arith.constant 0 : index
    # CHECK:            %[[thread_id_x:.*]] = gpu.thread_id  x
    # CHECK:            %[[S0:.*]] = stream.binding.subspan %[[ARG0]][%[[C0]]] : !stream.binding -> memref<16x16xf16, strided<[16, 1], offset: ?>>
    # CHECK:            %[[I0:.*]] = affine.apply #[[MAP0]]()[%[[thread_id_x]]]
    # CHECK:            %[[V:.*]] = vector.load %[[S0]][%[[I0]], %[[C0]]] : memref<16x16xf16, strided<[16, 1], offset: ?>>, vector<16xf16>
    # CHECK:            %[[S1:.*]] = stream.binding.subspan %[[ARG1]][%[[C0]]] : !stream.binding -> memref<16x16xf16, strided<[16, 1], offset: ?>>
    # CHECK:            vector.store %[[V]], %[[S1]][%[[I0]], %[[C0]]] : memref<16x16xf16, strided<[16, 1], offset: ?>>, vector<16xf16>
    # CHECK:            return


@run_test
def test_read_write_diagonal():
    # This test, tests for functionality of tkw.self_index, by
    # generating code that generate a triangular matrix if M > N.
    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=64, waves_per_block=(1, 1, 1), vector_shapes={M: 16, N: 16}
        )
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N)]

    @tkw.wave(constraints)
    def read_write_diagonal(
        c: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
    ):
        ZEROF = tkl.Register[M, N, tkl.f16](0.0)
        ONEF = tkl.Register[M, N, tkl.f16](1.0)
        m_index = tkw.self_index(M, tkl.i64)
        m_index = tkw.broadcast(m_index, target_shape=[M, N])
        n_index = tkw.self_index(N, tkl.i64)
        res = tkw.select(m_index >= n_index, ZEROF, ONEF)
        tkw.write(res, c)

    read_write_diagonal = wave_compile(
        get_wave_compile_options(canonicalize=True), read_write_diagonal
    )
    print(read_write_diagonal.asm)

    # CHECK-LABEL:    test_read_write_diagonal
    # CHECK-DAG:        #[[map:.*]] = affine_map<()[s0] -> (s0 - (s0 floordiv 64) * 48)>
    # CHECK:          func.func @read_write_diagonal
    # CHECK-SAME:       (%[[ARG0:.*]]: !stream.binding)
    # CHECK-DAG:        %[[C0:.*]] = arith.constant 0 : index
    # CHECK-DAG:        %[[CST:.*]] = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]> : vector<16xi64>
    # CHECK-DAG:        %[[CST_0:.*]] = arith.constant dense<1.000000e+00> : vector<16xf16>
    # CHECK-DAG:        %[[CST_1:.*]] = arith.constant dense<0.000000e+00> : vector<16xf16>
    # CHECK:            %[[THREAD_ID_X:.*]] = gpu.thread_id  x
    # CHECK:            %[[D0:.*]] = affine.apply #[[map]]()[%[[THREAD_ID_X]]]
    # CHECK:            %[[D1:.*]] = arith.index_cast %[[D0]] : index to i64
    # CHECK:            %[[D4:.*]] = vector.splat %[[D1]] : vector<16xi64>
    # CHECK:            %[[D5:.*]] = arith.cmpi sge, %[[D4]], %[[CST]] : vector<16xi64>
    # CHECK:            %[[D6:.*]] = arith.select %[[D5]], %[[CST_1]], %[[CST_0]] : vector<16xi1>, vector<16xf16>
    # CHECK:            %[[D7:.*]] = stream.binding.subspan %[[ARG0]][%[[C0]]] : !stream.binding -> memref<16x16xf16, strided<[16, 1], offset: ?>>
    # CHECK:            vector.store %[[D6]], %[[D7]][%[[D0]], %[[C0]]] : memref<16x16xf16, strided<[16, 1], offset: ?>>, vector<16xf16>


@run_test
def test_read_write_masked():
    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=64, waves_per_block=(1, 1, 1), vector_shapes={M: 4, N: 4}
        )
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N)]

    @tkw.wave(constraints)
    def read_write_masked(
        a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
    ):
        res = tkw.read(a)
        tkw.write(res, b)

    options = WaveCompileOptions(
        subs={
            M: 1,
            N: 3,
            BLOCK_M: 4,
            BLOCK_N: 4,
            ADDRESS_SPACE: tkl.AddressSpace.SHARED_MEMORY.value,
        },
        canonicalize=True,
        compile_to_mlir=True,
    )
    read_write_masked = wave_compile(options, read_write_masked)
    print(read_write_masked.asm)

    # CHECK-LABEL:    test_read_write_masked
    # CHECK-DAG:        #[[map:.*]] = affine_map<()[s0] -> (s0 - (s0 floordiv 64) * 60)>
    # CHECK:          func.func @read_write_masked
    # CHECK-SAME:       (%[[ARG0:.*]]: !stream.binding, %[[ARG1:.*]]: !stream.binding)
    # CHECK-DAG:        %[[CST:.*]] = arith.constant dense<0.000000e+00> : vector<4xf16>
    # CHECK-DAG:        %[[CST_0:.*]] = arith.constant dense<[true, true, true, false]> : vector<4xi1>
    # CHECK-DAG:        %[[C1:.*]] = arith.constant 1 : index
    # CHECK-DAG:        %[[C0:.*]] = arith.constant 0 : index
    # CHECK:            %[[THREAD_ID_X:.*]] = gpu.thread_id  x
    # CHECK:            %[[D0:.*]] = stream.binding.subspan %[[ARG0]][%[[C0]]] : !stream.binding -> memref<1x3xf16, strided<[3, 1], offset: ?>>
    # CHECK:            %[[D1:.*]] = affine.apply #[[map]]()[%[[THREAD_ID_X]]]
    # CHECK:            %[[D2:.*]] = arith.cmpi slt, %[[D1]], %[[C1]] : index
    # CHECK:            %[[D3:.*]] = vector.splat %[[D2]] : vector<4xi1>
    # CHECK:            %[[D4:.*]] = arith.andi %[[D3]], %[[CST_0]] : vector<4xi1>
    # CHECK:            %[[D5:.*]] = vector.maskedload %[[D0]][%[[D1]], %[[C0]]], %[[D4]], %[[CST]] : memref<1x3xf16, strided<[3, 1], offset: ?>>, vector<4xi1>, vector<4xf16> into vector<4xf16>
    # CHECK:            %[[D6:.*]] = stream.binding.subspan %[[ARG1]][%[[C0]]] : !stream.binding -> memref<1x3xf16, strided<[3, 1], offset: ?>>
    # CHECK:            vector.maskedstore %[[D6]][%[[D1]], %[[C0]]], %[[D4]], %[[D5]] : memref<1x3xf16, strided<[3, 1], offset: ?>>, vector<4xi1>, vector<4xf16>


@run_test
def test_read_write_masked_shared():
    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=64, waves_per_block=(1, 1, 1), vector_shapes={M: 4, N: 4}
        )
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N)]

    @tkw.wave(constraints)
    def read_write_masked_shared(
        a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[M, N, ADDRESS_SPACE_0, tkl.f16],
    ):
        res = tkw.read(a)
        tkw.write(res, b)

    options = WaveCompileOptions(
        subs={
            M: 1,
            N: 3,
            BLOCK_M: 4,
            BLOCK_N: 4,
            ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
            ADDRESS_SPACE_0: GLOBAL_ADDRESS_SPACE,
        },
        canonicalize=True,
        compile_to_mlir=True,
    )
    read_write_masked_shared = wave_compile(options, read_write_masked_shared)
    print(read_write_masked_shared.asm)

    # CHECK-LABEL:    func.func @read_write_masked_shared
    # Check shared mem load stores are non masked
    # CHECK:            %{{.*}} = vector.maskedload {{.*}} : memref<1x3xf16, strided<[3, 1], offset: ?>>, vector<4xi1>, vector<4xf16> into vector<4xf16>
    # CHECK:            vector.store {{.*}} : memref<4x8xf16, #gpu.address_space<workgroup>>, vector<4xf16>
    # CHECK:            %{{.*}} = vector.load {{.*}} : memref<4x8xf16, #gpu.address_space<workgroup>>, vector<4xf16>
    # CHECK:            vector.maskedstore {{.*}} : memref<1x3xf16, strided<[3, 1], offset: ?>>, vector<4xi1>, vector<4xf16>


@run_test
def test_read_write_mapping():
    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=64, waves_per_block=(1, 1, 1), vector_shapes={M: 16, N: 16}
        )
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N)]

    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)
    mapping = tkw.IndexMapping(
        num_iterators=2, inputs={M: i, N: j}, outputs={M: i, N: j}
    )

    @tkw.wave(constraints)
    def read_write_mapping(
        a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[N, M, ADDRESS_SPACE, tkl.f16],
    ):
        res = tkw.read(a)
        tkw.write(res, b, mapping=mapping)

    read_write_mapping = wave_compile(
        get_wave_compile_options(canonicalize=True), read_write_mapping
    )
    print(read_write_mapping.asm)

    # CHECK-LABEL:    test_read_write_mapping
    # CHECK:          func.func @read_write_mapping
    # We are transposing, so contiguous load becomes non-contiguous store
    # CHECK:            vector.load
    # CHECK-NOT:        vector.load
    # CHECK-COUNT-16:   memref.store


@run_test
def test_read_write_dynamic_mapping():
    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=64, waves_per_block=(1, 1, 1), vector_shapes={M: 16, N: 16}
        )
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N)]

    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)
    k = tkw.IndexMapping.dynamic_val(0)
    mapping = tkw.IndexMapping(
        num_iterators=2,
        inputs={M: i, N: k},
        outputs={M: i, N: j},
        dynamic_val_mappings={M: i, N: j},
    )

    @tkw.wave(constraints)
    def read_write_dynamic_mapping(
        a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
        off: tkl.Memory[M, N, ADDRESS_SPACE, tkl.i32],
        b: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
    ):
        offset = tkw.read(off)
        res = tkw.read(
            a,
            mapping=mapping,
            mapping_dynamic_vals=(offset,),
        )
        tkw.write(res, b)

    read_write_dynamic_mapping = wave_compile(
        get_wave_compile_options(canonicalize=True), read_write_dynamic_mapping
    )
    print(read_write_dynamic_mapping.asm)

    # CHECK-LABEL:    test_read_write_dynamic_mapping
    # CHECK-DAG:        #[[map0:.*]] = affine_map<()[s0] -> (s0 - (s0 floordiv 64) * 48)>
    # CHECK-DAG:        #[[map1:.*]] = affine_map<()[s0] -> (s0 * 16 - (s0 floordiv 64) * 768)>
    # CHECK:          func.func @read_write_dynamic_mapping
    # CHECK-SAME:       (%[[ARG0:.*]]: !stream.binding, %[[ARG1:.*]]: !stream.binding, %[[ARG2:.*]]: !stream.binding)
    # CHECK-DAG:        %[[C0:.*]] = arith.constant 0 : index
    # CHECK:            %[[THREAD_ID_X:.*]] = gpu.thread_id  x
    # CHECK:            %[[D0:.*]] = stream.binding.subspan %[[ARG1]][%[[C0]]] : !stream.binding -> memref<16x16xi32, strided<[16, 1], offset: ?>>
    # CHECK:            %[[D1:.*]] = affine.apply #[[map0]]()[%[[THREAD_ID_X]]]
    # CHECK:            %[[D2:.*]] = vector.load %[[D0]][%[[D1]], %[[C0]]] : memref<16x16xi32, strided<[16, 1], offset: ?>>, vector<16xi32>
    # CHECK:            %[[D3:.*]] = stream.binding.subspan %[[ARG0]][%[[C0]]] : !stream.binding -> memref<16x16xf16, strided<[16, 1], offset: ?>>
    # CHECK:            %[[D4:.*]] = arith.index_cast %[[D2]] : vector<16xi32> to vector<16xindex>
    # CHECK:            %[[D5:.*]] = affine.apply #[[map1]]()[%[[THREAD_ID_X]]]
    # CHECK:            %[[D6:.*]] = vector.splat %[[D5]] : vector<16xindex>
    # CHECK:            %[[D7:.*]] = arith.addi %[[D6]], %[[D4]] overflow<nsw, nuw> : vector<16xindex>
    # CHECK-COUNT-16:   memref.load
    # CHECK-NOT:        vector.extract
    # CHECK:            %[[RES:.*]] = vector.from_elements
    # CHECK:            %[[D58:.*]] = stream.binding.subspan %[[ARG2]][%[[C0]]] : !stream.binding -> memref<16x16xf16, strided<[16, 1], offset: ?>>
    # CHECK:            vector.store %[[RES]], %[[D58]][%[[D1]], %[[C0]]] : memref<16x16xf16, strided<[16, 1], offset: ?>>, vector<16xf16>


@run_test
def test_read_write_dynamic_mapping_broadcast():
    ONE = tkl.sym.ONE
    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(1, 1, 1),
            vector_shapes={M: 16, N: 16, ONE: 1},
        )
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N)]

    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)
    k = tkw.IndexMapping.dynamic_val(0)
    mapping = tkw.IndexMapping(
        num_iterators=2,
        inputs={M: i, N: k + j % 16},
        outputs={M: i, N: j},
        dynamic_val_mappings={M: i, ONE: j // 16},
    )

    @tkw.wave(constraints)
    def read_write_dynamic_mapping_broadcast(
        a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
        off: tkl.Memory[M, ONE, ADDRESS_SPACE, tkl.i32],
        b: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
    ):
        offset = tkw.read(off)
        res = tkw.read(
            a,
            mapping=mapping,
            mapping_dynamic_vals=(offset,),
        )
        tkw.write(res, b)

    read_write_dynamic_mapping_broadcast = wave_compile(
        get_wave_compile_options(canonicalize=True, additional_symbols={ONE: 1}),
        read_write_dynamic_mapping_broadcast,
    )
    print(read_write_dynamic_mapping_broadcast.asm)

    # CHECK-LABEL:    func.func @read_write_dynamic_mapping_broadcast
    # CHECK:            %[[OFF:.*]] = memref.load %{{.*}}[%[[M:.*]], %{{.*}}] : memref<16x1xi32, strided<[1, 1], offset: ?>>
    # CHECK:            %[[IDX:.*]] = arith.index_cast %[[OFF]] : i32 to index
    # CHECK:            %[[RES:.*]] = vector.load %{{.*}}[%[[M]], %[[IDX]]] : memref<16x16xf16, strided<[16, 1], offset: ?>>, vector<16xf16>
    # CHECK:            vector.store %[[RES]], %{{.*}}[%[[M]], %{{.*}}] : memref<16x16xf16, strided<[16, 1], offset: ?>>, vector<16xf16>


@run_test
def test_read_write_dynamic_mapping_chain():
    SIZE1 = tkl.sym.SIZE1
    SIZE2 = tkl.sym.SIZE2
    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(1, 1, 1),
            vector_shapes={M: 16, N: 4, SIZE1: 1, SIZE2: 1},
        )
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N)]

    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)
    k = tkw.IndexMapping.dynamic_val(0)
    mapping1 = tkw.IndexMapping(
        num_iterators=2,
        inputs={M: i, SIZE2: k},
        outputs={M: i, SIZE2: j},
        dynamic_val_mappings={M: i, SIZE1: j // 2},
    )
    mapping2 = tkw.IndexMapping(
        num_iterators=2,
        inputs={M: i, N: k + j % 4},
        outputs={M: i, N: j},
        dynamic_val_mappings={M: i, SIZE2: j // 4},
    )

    @tkw.wave(constraints)
    def read_write_dynamic_mapping_chain(
        a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
        off1: tkl.Memory[M, SIZE1, ADDRESS_SPACE, tkl.i32],
        off2: tkl.Memory[M, SIZE2, ADDRESS_SPACE, tkl.i32],
        b: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
    ):
        offset1 = tkw.read(off1)
        offset2 = tkw.read(
            off2,
            mapping=mapping1,
            mapping_dynamic_vals=(offset1,),
        )
        res = tkw.read(
            a,
            mapping=mapping2,
            mapping_dynamic_vals=(offset2,),
        )
        tkw.write(res, b)

    read_write_dynamic_mapping_chain = wave_compile(
        get_wave_compile_options(
            canonicalize=True, additional_symbols={BLOCK_N: 4, SIZE1: 2, SIZE2: 4}
        ),
        read_write_dynamic_mapping_chain,
    )
    print(read_write_dynamic_mapping_chain.asm)

    # CHECK-LABEL:    test_read_write_dynamic_mapping_chain
    # CHECK-DAG:        #[[map0:.*]] = affine_map<()[s0] -> (s0 - (s0 floordiv 64) * 48)>
    # CHECK-DAG:        #[[map1:.*]] = affine_map<()[s0] -> (s0 floordiv 2)>
    # CHECK-DAG:        #[[map2:.*]] = affine_map<()[s0] -> (s0 * 4)>
    # CHECK:          func.func @read_write_dynamic_mapping_chain
    # CHECK-SAME:     (%[[ARG0:.*]]: !stream.binding, %[[ARG1:.*]]: !stream.binding, %[[ARG2:.*]]: !stream.binding, %[[ARG3:.*]]: !stream.binding)
    # CHECK-DAG:      %[[C0:.*]] = arith.constant 0 : index
    # CHECK:          %[[WORKGROUP_ID_1:.*]] = stream.dispatch.workgroup.id[1] : index
    # CHECK:          %[[THREAD_ID_X:.*]] = gpu.thread_id  x
    # CHECK:          %[[D0:.*]] = stream.binding.subspan %[[ARG1]][%[[C0]]] : !stream.binding -> memref<16x2xi32, strided<[2, 1], offset: ?>>
    # CHECK:          %[[D1:.*]] = affine.apply #[[map0]]()[%[[THREAD_ID_X]]]
    # CHECK:          %[[D2:.*]] = affine.apply #[[map1]]()[%[[WORKGROUP_ID_1]]]
    # CHECK:          %[[D3:.*]] = memref.load %[[D0]][%[[D1]], %[[D2]]] : memref<16x2xi32, strided<[2, 1], offset: ?>>
    # CHECK:          %[[D4:.*]] = stream.binding.subspan %[[ARG2]][%[[C0]]] : !stream.binding -> memref<16x4xi32, strided<[4, 1], offset: ?>>
    # CHECK:          %[[D5:.*]] = arith.index_cast %[[D3]] : i32 to index
    # CHECK:          %[[D6:.*]] = memref.load %[[D4]][%[[D1]], %[[D5]]] : memref<16x4xi32, strided<[4, 1], offset: ?>>
    # CHECK:          %[[D8:.*]] = stream.binding.subspan %[[ARG0]][%[[C0]]] : !stream.binding -> memref<16x16xf16, strided<[16, 1], offset: ?>>
    # CHECK:          %[[D9:.*]] = arith.index_cast %[[D6]] : i32 to index
    # CHECK:          %[[D11:.*]] = vector.load %[[D8]][%[[D1]], %[[D9]]] : memref<16x16xf16, strided<[16, 1], offset: ?>>, vector<4xf16>
    # CHECK:          %[[D12:.*]] = stream.binding.subspan %[[ARG3]][%[[C0]]] : !stream.binding -> memref<16x16xf16, strided<[16, 1], offset: ?>>
    # CHECK:          %[[D13:.*]] = affine.apply #[[map2]]()[%[[WORKGROUP_ID_1]]]
    # CHECK:          vector.store %[[D11]], %[[D12]][%[[D1]], %[[D13]]] : memref<16x16xf16, strided<[16, 1], offset: ?>>, vector<4xf16>


@run_test
def test_read_write_dynamic_symbol():
    S = tkl.sym.S
    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(1, 1, 1),
            vector_shapes={M: 1, N: 1, S: 1},
        )
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N)]

    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)
    mapping = tkw.IndexMapping(
        num_iterators=2,
        inputs={S: S, N: j},
        outputs={S: i, N: j},
        dynamic_val_mappings={S: i, N: j},
    )

    @tkw.wave(constraints)
    def test_dyn_symbol(
        a: tkl.Memory[S, N, ADDRESS_SPACE, tkl.f16],
        off: tkl.Memory[M, N, ADDRESS_SPACE, tkl.i32],
        b: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
    ):
        offset = tkw.read(off)
        tkw.set_symbol(S, offset)
        res = tkw.read(
            a,
            mapping=mapping,
        )
        tkw.write(res, b)

    test_dyn_symbol = wave_compile(
        get_wave_compile_options(
            canonicalize=True,
            dynamic_symbols=[S],
            additional_symbols={BLOCK_M: 1, BLOCK_N: 1},
        ),
        test_dyn_symbol,
    )
    print(test_dyn_symbol.asm)

    # CHECK-LABEL:    func.func @test_dyn_symbol
    #  CHECK-SAME:      (%[[ARG0:.*]]: !stream.binding, %[[ARG1:.*]]: !stream.binding, %[[ARG2:.*]]: !stream.binding, %[[ARG3:.*]]: index)
    #   CHECK-DAG:      %[[C0:.*]] = arith.constant 0 : index
    #       CHECK:      %[[A2:.*]] = stream.binding.subspan %[[ARG1]][%[[C0]]] : !stream.binding -> memref<16x16xi32, strided<[16, 1], offset: ?>>
    #       CHECK:      %[[O1:.*]] = memref.load %[[A2]][%[[M:.*]], %[[N:.*]]] : memref<16x16xi32, strided<[16, 1], offset: ?>>
    #       CHECK:      %[[O2:.*]] = arith.index_cast %[[O1]] : i32 to index
    #       CHECK:      %[[A1:.*]] = stream.binding.subspan %[[ARG0]][%[[C0]]] : !stream.binding -> memref<?x16xf16, strided<[16, 1], offset: ?>>{%arg3}
    #       CHECK:      %[[RES:.*]] = vector.load %[[A1]][%[[O2]], %[[N]]] : memref<?x16xf16, strided<[16, 1], offset: ?>>, vector<1xf16>
    #       CHECK:      %[[A3:.*]] = stream.binding.subspan %[[ARG2]][%[[C0]]] : !stream.binding -> memref<16x16xf16, strided<[16, 1], offset: ?>>
    #       CHECK:      vector.store %[[RES]], %[[A3]][%[[M]], %[[N]]] : memref<16x16xf16, strided<[16, 1], offset: ?>>, vector<1xf16>


@run_test
def test_read_write_dynamic_symbol_expr():
    S = tkl.sym.S
    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(1, 1, 1),
            vector_shapes={M: 1, N: 1, S: 1},
        )
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N)]

    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)
    mapping = tkw.IndexMapping(
        num_iterators=2,
        inputs={S: S, N: j},
        outputs={S: i, N: j},
        dynamic_val_mappings={S: i, N: j},
    )

    @tkw.wave(constraints)
    def test_dyn_expr(
        a: tkl.Memory[S, N, ADDRESS_SPACE, tkl.f16],
        off: tkl.Memory[M, N, ADDRESS_SPACE, tkl.i32],
        b: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
    ):
        offset = tkw.read(off)
        offset = tkw.apply_expr(offset, lambda a: M - a - 1)
        tkw.set_symbol(S, offset)
        res = tkw.read(
            a,
            mapping=mapping,
        )
        tkw.write(res, b)

    test_dyn_expr = wave_compile(
        get_wave_compile_options(
            canonicalize=True,
            dynamic_symbols=[S],
            additional_symbols={BLOCK_M: 1, BLOCK_N: 1},
        ),
        test_dyn_expr,
    )
    print(test_dyn_expr.asm)

    # CHECK-LABEL:    func.func @test_dyn_expr
    #  CHECK-SAME:      (%[[ARG0:.*]]: !stream.binding, %[[ARG1:.*]]: !stream.binding, %[[ARG2:.*]]: !stream.binding, %[[ARG3:.*]]: index)
    #   CHECK-DAG:      %[[CST:.*]] = arith.constant 15 : index
    #   CHECK-DAG:      %[[C0:.*]] = arith.constant 0 : index
    #       CHECK:      %[[A2:.*]] = stream.binding.subspan %[[ARG1]][%[[C0]]] : !stream.binding -> memref<16x16xi32, strided<[16, 1], offset: ?>>
    #       CHECK:      %[[O1:.*]] = memref.load %[[A2]][%[[M:.*]], %[[N:.*]]] : memref<16x16xi32, strided<[16, 1], offset: ?>>
    #       CHECK:      %[[O2:.*]] = arith.index_cast %[[O1]] : i32 to index
    #       CHECK:      %[[O3:.*]] = arith.subi %[[CST]], %[[O2]] : index
    #       CHECK:      %[[A1:.*]] = stream.binding.subspan %[[ARG0]][%[[C0]]] : !stream.binding -> memref<?x16xf16, strided<[16, 1], offset: ?>>{%arg3}
    #       CHECK:      %[[RES:.*]] = vector.load %[[A1]][%[[O3]], %[[N]]] : memref<?x16xf16, strided<[16, 1], offset: ?>>, vector<1xf16>
    #       CHECK:      %[[A3:.*]] = stream.binding.subspan %[[ARG2]][%[[C0]]] : !stream.binding -> memref<16x16xf16, strided<[16, 1], offset: ?>>
    #       CHECK:      vector.store %[[RES]], %[[A3]][%[[M]], %[[N]]] : memref<16x16xf16, strided<[16, 1], offset: ?>>, vector<1xf16>


@run_test
def test_read_write_conditional():
    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(1, 1, 1),
            vector_shapes={M: 1, N: 1},
        )
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N)]

    @tkw.wave(constraints)
    def test_conditional(
        a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
        mask: tkl.Memory[M, N, ADDRESS_SPACE, tkl.i32],
        b: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
    ):
        res = tkw.read(a, elements_per_thread=1)
        cond = tkw.read(mask, elements_per_thread=1)

        cond = tkw.apply_expr(cond, lambda a: a > 0)

        @tkw.conditional(cond)
        def then():
            tkw.write(res, b, elements_per_thread=1)

    test_conditional = wave_compile(
        get_wave_compile_options(canonicalize=True), test_conditional
    )
    print(test_conditional.asm)

    # CHECK-LABEL:    func.func @test_conditional
    #  CHECK-SAME:      (%[[ARG0:.*]]: !stream.binding, %[[ARG1:.*]]: !stream.binding, %[[ARG2:.*]]: !stream.binding)
    #   CHECK-DAG:      %[[C0:.*]] = arith.constant 0 : index
    #       CHECK:      %[[A1:.*]] = stream.binding.subspan %[[ARG0]][%[[C0]]] : !stream.binding -> memref<16x16xf16, strided<[16, 1], offset: ?>>
    #       CHECK:      %[[RES:.*]] = vector.load %[[A1]][%[[M:.*]], %[[N:.*]]] : memref<16x16xf16, strided<[16, 1], offset: ?>>, vector<1xf16>
    #       CHECK:      %[[A2:.*]] = stream.binding.subspan %[[ARG1]][%[[C0]]] : !stream.binding -> memref<16x16xi32, strided<[16, 1], offset: ?>>
    #       CHECK:      %[[O1:.*]] = memref.load %[[A2]][%[[M]], %[[N]]] : memref<16x16xi32, strided<[16, 1], offset: ?>>
    #       CHECK:      %[[O2:.*]] = arith.index_cast %[[O1]] : i32 to index
    #       CHECK:      %[[O3:.*]] = arith.cmpi sgt, %[[O2]], %[[C0]] : index
    #       CHECK:      scf.if %[[O3]] {
    #       CHECK:        %[[A3:.*]] = stream.binding.subspan %[[ARG2]][%[[C0]]] : !stream.binding -> memref<16x16xf16, strided<[16, 1], offset: ?>>
    #       CHECK:        vector.store %[[RES]], %[[A3]][%[[M]], %[[N]]] : memref<16x16xf16, strided<[16, 1], offset: ?>>, vector<1xf16>
    #       CHECK:      }


@run_test
def test_dynamic_copy():
    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=64, waves_per_block=(1, 1, 1), vector_shapes={M: 16, N: 16}
        )
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N)]

    @tkw.wave(constraints)
    def dynamic_copy(a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16]):
        b = tkw.read(a, elements_per_thread=16)
        tkw.write(b, a, elements_per_thread=16)

    dynamic_copy = wave_compile(
        get_wave_compile_options(canonicalize=True, dynamic_symbols=[M, N]),
        dynamic_copy,
    )
    print(dynamic_copy.asm)

    # CHECK-LABEL:    test_dynamic_copy
    # CHECK-DAG:        #[[map1:.*]] = affine_map<()[s0, s1] -> (s0 + s1 * 16 - (s0 floordiv 64) * 48)>
    # CHECK-DAG:        #[[map2:.*]] = affine_map<()[s0] -> (s0 * 16)>
    # CHECK:          func.func @dynamic_copy
    # CHECH-SAME:       (%[[ARG0:.*]]: !stream.binding, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index)
    # CHECK-DAG:        %[[CST:.*]] = arith.constant dense<0.000000e+00> : vector<16xf16>
    # CHECK-DAG:        %[[CST_0:.*]] = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]> : vector<16xindex>
    # CHECK-DAG:        %[[C0:.*]] = arith.constant 0 : index
    # CHECK:            %[[WORKGROUP_ID_0:.*]] = stream.dispatch.workgroup.id[0] : index
    # CHECK:            %[[WORKGROUP_ID_1:.*]] = stream.dispatch.workgroup.id[1] : index
    # CHECK:            %[[THREAD_ID_X:.*]] = gpu.thread_id  x
    # CHECK:            %[[D0:.*]] = stream.binding.subspan %[[ARG0]][%[[C0]]] : !stream.binding -> memref<?x?xf16, strided<[?, 1], offset: ?>>{%[[ARG1]], %[[ARG2]]}
    # CHECK:            %[[D1:.*]] = affine.apply #[[map1]]()[%[[THREAD_ID_X]], %[[WORKGROUP_ID_0]]]
    # CHECK:            %[[D2:.*]] = affine.apply #[[map2]]()[%[[WORKGROUP_ID_1]]]
    # CHECK:            %[[D3:.*]] = vector.splat %[[D2]] : vector<16xindex>
    # CHECK:            %[[D4:.*]] = arith.addi %[[D3]], %[[CST_0]] overflow<nsw, nuw> : vector<16xindex>
    # CHECK:            %[[D5:.*]] = vector.splat %[[ARG2]] : vector<16xindex>
    # CHECK:            %[[D6:.*]] = arith.cmpi slt, %[[D4]], %[[D5]] : vector<16xindex>
    # CHECK:            %[[D7:.*]] = arith.cmpi slt, %[[D1]], %[[ARG1]] : index
    # CHECK:            %[[D8:.*]] = vector.splat %[[D7]] : vector<16xi1>
    # CHECK:            %[[D9:.*]] = arith.andi %[[D6]], %[[D8]] : vector<16xi1>
    # CHECK:            %[[D10:.*]] = vector.maskedload %[[D0]][%[[D1]], %[[D2]]], %[[D9]], %[[CST]] : memref<?x?xf16, strided<[?, 1], offset: ?>>, vector<16xi1>, vector<16xf16> into vector<16xf16>
    # CHECK:            vector.maskedstore %[[D0]][%[[D1]], %[[D2]]], %[[D9]], %[[D10]] : memref<?x?xf16, strided<[?, 1], offset: ?>>, vector<16xi1>, vector<16xf16>


@run_test
def test_add_float():
    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=64, waves_per_block=(1, 1, 1), vector_shapes={M: 16, N: 16}
        )
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N)]

    @tkw.wave(constraints)
    def add(a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16]):
        a_reg = tkw.read(a)
        res = a_reg + a_reg

    add = wave_compile(get_wave_compile_options(), add)
    print(add.asm)

    # CHECK-LABEL: func @add
    # CHECK: %[[SLICE:.+]] = vector.load
    # CHECK: arith.addf %[[SLICE]], %[[SLICE]] : vector<16xf16>


@run_test
def test_add_integer():
    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=64, waves_per_block=(1, 1, 1), vector_shapes={M: 16, N: 16}
        )
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N)]

    @tkw.wave(constraints)
    def test(a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.i32]):
        a_reg = tkw.read(a)
        res = a_reg + a_reg

    test = wave_compile(get_wave_compile_options(), test)
    print(test.asm)
    # CHECK-LABEL: func @test
    # CHECK: %[[SLICE:.+]] = vector.load
    # CHECK: arith.addi %[[SLICE]], %[[SLICE]] : vector<16xi32>


@run_test
def test_unary_lowerings():
    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=64, waves_per_block=(1, 1, 1), vector_shapes={M: 16, N: 16}
        )
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 2)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]

    @tkw.wave(constraints)
    def test(
        a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[M, N, ADDRESS_SPACE, tkl.i32],
    ):
        a_reg = tkw.read(a, elements_per_thread=4)
        b_reg = tkw.read(b, elements_per_thread=4)
        res = -a_reg
        res = tkw.exp2(res)
        res = tkw.reciprocal(res)
        res = tkw.abs(res)
        res_b = tkw.abs(b_reg)
        res = tkw.tanh(res)
        res = tkw.tanh_approx(res)
        res = tkw.softsign(res, logit_cap=30.0, apply_scaling=True, head_dim=128)
        res = tkw.roundeven(res)
        res = tkw.sin(res)
        res = tkw.cos(res)
        res = tkw.exp(res)
        tkw.write(res, a, elements_per_thread=4)
        tkw.write(res_b, b, elements_per_thread=4)

    test = wave_compile(get_wave_compile_options(), test)
    print(test.asm)
    # CHECK-LABEL: test_unary_lowerings
    # Testing Negate
    # CHECK: %[[NEG:.+]] = arith.negf

    # Testing exp2
    # CHECK: %[[EXP2:.+]] = math.exp2 %[[NEG]]

    # Testing reciprocal
    # CHECK: %[[ONES:.+]] = arith.constant dense<1.000000e+00> : vector<4xf16>
    # CHECK: %[[RECIPROCAL:.+]] = arith.divf %[[ONES]], %[[EXP2]] : vector<4xf16>

    # Testing abs
    # CHECK: %[[ABSF:.+]] = math.absf %[[RECIPROCAL]]
    # CHECK: %[[ABSI:.+]] = math.absi

    # Tests tanh
    # CHECK: %[[TANH:.+]] = math.tanh %[[ABSF]]

    # Tests tanh approx
    # CHECK: %[[ABS:.+]] = math.absf %[[TANH]]
    # CHECK: %[[FACTOR:.+]] = arith.constant dense<-2.884770e+00> : vector<4xf16>
    # CHECK: %[[T:.+]] = arith.mulf %[[ABS]], %[[FACTOR]]
    # CHECK: %[[EXP2:.+]] = math.exp2 %[[T]]
    # CHECK: %[[ONE:.+]] = arith.constant dense<1.000000e+00> : vector<4xf16>
    # CHECK: %[[D:.+]] = arith.addf %[[EXP2]], %[[ONE]]
    # CHECK: %[[RECIP:.+]] = arith.divf %[[ONE]], %[[D]] : vector<4xf16>
    # CHECK: %[[NEG_ONE:.+]] = arith.constant dense<-1.000000e+00> : vector<4xf16>
    # CHECK: %[[NEG_RECIP:.+]] = arith.mulf %[[RECIP]], %[[NEG_ONE:.+]] : vector<4xf16>
    # CHECK: %[[TEMP:.+]] = arith.mulf %[[EXP2]], %[[NEG_RECIP]] : vector<4xf16>
    # CHECK: %[[R:.+]] = arith.addf %[[TEMP]], %[[RECIP]] : vector<4xf16>
    # CHECK: %[[TANH_APPROX:.+]] = math.copysign %[[R]], %[[TANH]] : vector<4xf16>

    # Tests softsign
    # CHECK: %[[ONE:.+]] = arith.constant dense<1.000000e+00> : vector<4xf16>
    # CHECK: %[[CAP:.+]] = arith.constant dense<3.771970e-01> : vector<4xf16>
    # CHECK: %[[SCALED:.+]] = arith.mulf %[[TANH_APPROX]], %[[CAP]] : vector<4xf16>
    # CHECK: %[[ABS2:.+]] = math.absf %[[SCALED]] : vector<4xf16>
    # CHECK: %[[ADD:.+]] = arith.addf %[[ONE]], %[[ABS2]] : vector<4xf16>
    # CHECK: %[[RECIP_DENOM:.+]] = arith.divf %[[ONE]], %[[ADD]] : vector<4xf16>
    # CHECK: %[[SOFTSIGN:.+]] = arith.mulf %[[TANH_APPROX]], %[[RECIP_DENOM]] : vector<4xf16>

    # Tests roundeven
    # CHECK: %[[ROUNDEVEN:.+]] = math.roundeven %[[SOFTSIGN]]

    # Tests sin
    # CHECK: %[[SIN:.+]] = math.sin %[[ROUNDEVEN]]
    # Tests cos
    # CHECK: %[[COS:.+]] = math.cos %[[SIN]]

    # Tests exp
    # CHECK: %[[EXP:.+]] = math.exp %[[COS]]


# Important to check lowering of scheduling/barrier ops.
@run_test
def test_scheduling_ops():
    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=64, waves_per_block=(1, 1, 1), vector_shapes={M: 16, N: 16}
        )
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N)]

    @tkw.wave(constraints)
    def schedule_ops(a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16]):
        tkw.shared_memory_barrier()
        tkw.set_wave_prio(1)
        tkw.read(a)
        tkw.set_wave_prio(0)
        tkw.workgroup_barrier()

    schedule_ops = wave_compile(get_wave_compile_options(), schedule_ops)
    print(schedule_ops.asm)

    # CHECK-LABEL:    func.func @schedule_ops
    # CHECK:            amdgpu.lds_barrier
    # CHECK:            rocdl.s.setprio 1
    # CHECK:            vector.load
    # CHECK:            rocdl.s.setprio 0
    # CHECK:            rocdl.s.barrier


@run_test
def test_reduce_sum():
    M = tkl.sym.M
    N = tkl.sym.N
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    ELEMS_PER_THREAD = tkl.sym.ELEMS_PER_THREAD
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(1, 1, 1),
            vector_shapes={M: 1, N: BLOCK_N},
        )
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 1)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 0)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N)]

    @tkw.wave(constraints)
    def test(
        a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[M, ADDRESS_SPACE, tkl.f16],
    ):
        lhs = tkw.read(a, elements_per_thread=ELEMS_PER_THREAD)
        rhs = tkw.read(b, elements_per_thread=ELEMS_PER_THREAD)
        res = lhs * rhs
        res = tkw.sum(res, dim=N)
        tkw.write(res, c, elements_per_thread=1)

    shape = (256, 128)
    options = WaveCompileOptions(
        subs={
            M: shape[0],
            N: shape[1],
            BLOCK_M: 1,
            BLOCK_N: 128,
            ELEMS_PER_THREAD: 2,
            ADDRESS_SPACE: tkl.AddressSpace.GLOBAL_MEMORY.value,
        },
        canonicalize=True,
        compile_to_mlir=True,
    )
    options = set_default_compile_config(options)
    test = wave_compile(options, test)
    print(test.asm)

    # CHECK-LABEL: test_reduce_sum
    # CHECK-DAG: %[[C1:.+]] = arith.constant 1 : i32
    # CHECK-DAG: %[[C2:.+]] = arith.constant 2 : i32
    # CHECK-DAG: %[[C4:.+]] = arith.constant 4 : i32
    # CHECK-DAG: %[[C8:.+]] = arith.constant 8 : i32
    # CHECK-DAG: %[[C16:.+]] = arith.constant 16 : i32
    # CHECK-DAG: %[[C32:.+]] = arith.constant 32 : i32
    # Elementwise
    # CHECK: arith.mulf {{.*}} : vector<2xf16>
    # Local Reduction
    # CHECK: arith.addf {{.*}} : f16
    # Global Reduction
    # CHECK: gpu.shuffle  xor %{{.+}}, %[[C1]], %{{.+}} : vector<1xf16>
    # CHECK: arith.addf {{.*}} : vector<1xf16>
    # CHECK: gpu.shuffle  xor %{{.+}}, %[[C2]], %{{.+}} : vector<1xf16>
    # CHECK: arith.addf {{.*}} : vector<1xf16>
    # CHECK: gpu.shuffle  xor %{{.+}}, %[[C4]], %{{.+}} : vector<1xf16>
    # CHECK: arith.addf {{.*}} : vector<1xf16>
    # CHECK: gpu.shuffle  xor %{{.+}}, %[[C8]], %{{.+}} : vector<1xf16>
    # CHECK: arith.addf {{.*}} : vector<1xf16>
    # CHECK: gpu.shuffle  xor %{{.+}}, %[[C16]], %{{.+}} : vector<1xf16>
    # CHECK: arith.addf {{.*}} : vector<1xf16>
    # CHECK: gpu.shuffle  xor %{{.+}}, %[[C32]], %{{.+}} : vector<1xf16>
    # CHECK: arith.addf {{.*}} : vector<1xf16>


# Tests for multiple local reduction, and we to emit and iteratively slice and reduce over multiple variables correctly.
@run_test
def test_mutliple_local_reduce_sum():
    M = tkl.sym.M
    N = tkl.sym.N
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    ELEMS_PER_THREAD = tkl.sym.ELEMS_PER_THREAD
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(1, 1, 1),
            vector_shapes={M: 1, N: BLOCK_N},
        )
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 1)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 0)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N)]

    @tkw.wave(constraints)
    def test(
        a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[M, ADDRESS_SPACE, tkl.f16],
    ):
        lhs = tkw.read(a, elements_per_thread=ELEMS_PER_THREAD)
        rhs = tkw.read(b, elements_per_thread=ELEMS_PER_THREAD)
        res = tkw.sum([lhs, rhs], dim=N)
        tkw.write(res, c, elements_per_thread=1)

    shape = (256, 128)
    options = WaveCompileOptions(
        subs={
            M: shape[0],
            N: shape[1],
            BLOCK_M: 1,
            BLOCK_N: 128,
            ELEMS_PER_THREAD: 2,
            ADDRESS_SPACE: tkl.AddressSpace.GLOBAL_MEMORY.value,
        },
        canonicalize=True,
        compile_to_mlir=True,
    )
    options = set_default_compile_config(options)
    test = wave_compile(options, test)
    print(test.asm)

    # CHECK-LABEL: test_mutliple_local_reduce_sum
    # CHECK: %[[LHS:.+]] = vector.load {{.*}} : memref<256x128xf16
    # CHECK: %[[RHS:.+]] = vector.load {{.*}} : memref<256x128xf16
    # Reduce all sources locally.
    # CHECK: %[[SRC_REDUC:.+]] = arith.addf %[[LHS]], %[[RHS]] : vector<2xf16>
    # Do Local Reductions.
    # CHECK: %[[LOCAL_REDUC0:.+]] = vector.extract %[[SRC_REDUC]][0] : f16 from vector<2xf16>
    # CHECK: %[[LOCAL_REDUC1:.+]] = vector.extract %[[SRC_REDUC]][1] : f16 from vector<2xf16>
    # CHECK: %[[REDUC_0:.+]] = arith.addf %[[LOCAL_REDUC0]], %[[LOCAL_REDUC1]] : f16
    # Expanded Global Max Reduction
    # CHECK-COUNT-6: gpu.shuffle  xor


# This test is to ensure that the propagation of indexing_dims between reduction and operations
# outside the reduction is working properly.
@run_test
def test_reduction_and_elemwise():
    M = tkl.sym.M
    N = tkl.sym.N
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    ELEMS_PER_THREAD = tkl.sym.ELEMS_PER_THREAD
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(1, 1, 1),
            vector_shapes={M: 1, N: BLOCK_N},
        )
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 1)]
    constraints += [tkw.WorkgroupConstraint(N, N, 0)]
    constraints += [tkw.TilingConstraint(N, BLOCK_N)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N)]

    @tkw.wave(constraints)
    def test(
        a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[M, ADDRESS_SPACE, tkl.f16],
    ):
        init_max = tkl.Register[M, tkl.f16](-1e6)

        @tkw.iterate(N, init_args=[init_max])
        def repeat(
            partial_max: tkl.Register[M, tkl.f16],
        ) -> tkl.Register[M, tkl.f16]:
            lhs = tkw.read(a, elements_per_thread=ELEMS_PER_THREAD)
            partial_max = tkw.max(lhs, partial_max, dim=N)
            return partial_max

        result = repeat + repeat
        tkw.write(result, c, elements_per_thread=1)

    shape = (256, 512)
    options = WaveCompileOptions(
        subs={
            M: shape[0],
            N: shape[1],
            BLOCK_M: 2,
            BLOCK_N: 128,
            ELEMS_PER_THREAD: 2,
            ADDRESS_SPACE: tkl.AddressSpace.GLOBAL_MEMORY.value,
        },
        canonicalize=True,
        compile_to_mlir=True,
    )
    options = set_default_compile_config(options)
    test = wave_compile(options, test)
    print(test.asm)

    # CHECK-LABEL: test_reduction_and_elemwise
    # CHECK-DAG: %[[C0_IDX:.+]] = arith.constant 0 : index
    # CHECK-DAG: %[[C4_IDX:.+]] = arith.constant 4 : index
    # CHECK-DAG: %[[C1_IDX:.+]] = arith.constant 1 : index
    # CHECK-DAG: %[[INIT:.+]] = arith.constant dense<0xFC00> : vector<1xf16>

    # Tile Reduction Loop
    # CHECK: %[[TILED:.+]]:2 = scf.for %[[ITER:.+]] = %[[C0_IDX]] to %[[C4_IDX]] step %[[C1_IDX]]
    # CHECK-SAME: iter_args(%[[ACC0:.+]] = %[[INIT]], %[[ACC1:.+]] = %[[INIT]]) -> (vector<1xf16>, vector<1xf16>) {
    # 1st Expanded Local Reduction
    # CHECK: arith.maximumf {{.*}} : f16
    # 1st Expanded Global Reduction
    # CHECK-COUNT-6: gpu.shuffle  xor
    # 1st Expanded Accumulator Reduction
    # CHECK: %[[ACC_REDUCE_0:.+]] = arith.maximumf %[[ACC0]], %{{.*}}

    # 2nd Expanded Local Reduction
    # CHECK: arith.maximumf {{.*}} : f16
    # 2nd Expanded Global Reduction
    # CHECK-COUNT-6: gpu.shuffle  xor
    # 2nd Expanded Accumulator Reduction
    # CHECK: %[[ACC_REDUCE_1:.+]] = arith.maximumf %[[ACC1]], %{{.*}}

    # CHECK: scf.yield %[[ACC_REDUCE_0]], %[[ACC_REDUCE_1]] : vector<1xf16>, vector<1xf16>
    # CHECK: %[[POST_TILE_ELEMWISE_0:.+]] =  arith.addf %[[TILED]]#0, %[[TILED]]#0 : vector<1xf16>
    # CHECK: %[[POST_TILE_ELEMWISE_1:.+]] =  arith.addf %[[TILED]]#1, %[[TILED]]#1 : vector<1xf16>
    # CHECK: vector.store %[[POST_TILE_ELEMWISE_0:.+]], %{{.*}}
    # CHECK: vector.store %[[POST_TILE_ELEMWISE_1:.+]], %{{.*}}


@run_test
def test_tiled_reduce_max():
    M = tkl.sym.M
    N = tkl.sym.N
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    ELEMS_PER_THREAD = tkl.sym.ELEMS_PER_THREAD
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(1, 1, 1),
            vector_shapes={M: 1, N: BLOCK_N},
        )
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 1)]
    constraints += [tkw.WorkgroupConstraint(N, N, 0)]
    constraints += [tkw.TilingConstraint(N, BLOCK_N)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N)]

    @tkw.wave(constraints)
    def test(
        a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[M, ADDRESS_SPACE, tkl.f16],
    ):
        init_max = tkl.Register[M, tkl.f16](-1e6)

        @tkw.iterate(N, init_args=[init_max])
        def repeat(
            partial_max: tkl.Register[M, tkl.f16],
        ) -> tkl.Register[M, tkl.f16]:
            lhs = tkw.read(a, elements_per_thread=ELEMS_PER_THREAD)
            rhs = tkw.read(b, elements_per_thread=ELEMS_PER_THREAD)
            res = lhs * rhs
            partial_max = tkw.max(res, partial_max, dim=N)
            return partial_max

        tkw.write(repeat, c, elements_per_thread=1)

    shape = (256, 512)
    options = WaveCompileOptions(
        subs={
            M: shape[0],
            N: shape[1],
            BLOCK_M: 1,
            BLOCK_N: 128,
            ELEMS_PER_THREAD: 2,
            ADDRESS_SPACE: tkl.AddressSpace.GLOBAL_MEMORY.value,
        },
        canonicalize=True,
        compile_to_mlir=True,
    )
    options = set_default_compile_config(options)
    test = wave_compile(options, test)
    print(test.asm)

    # CHECK-LABEL: test_tiled_reduce_max
    # CHECK-DAG: %[[C1:.+]] = arith.constant 1 : i32
    # CHECK-DAG: %[[C2:.+]] = arith.constant 2 : i32
    # CHECK-DAG: %[[C4:.+]] = arith.constant 4 : i32
    # CHECK-DAG: %[[C8:.+]] = arith.constant 8 : i32
    # CHECK-DAG: %[[C16:.+]] = arith.constant 16 : i32
    # CHECK-DAG: %[[C32:.+]] = arith.constant 32 : i32
    # CHECK-DAG: %[[C0_IDX:.+]] = arith.constant 0 : index
    # CHECK-DAG: %[[C4_IDX:.+]] = arith.constant 4 : index
    # CHECK-DAG: %[[C1_IDX:.+]] = arith.constant 1 : index
    # CHECK-DAG: %[[INIT:.+]] = arith.constant dense<0xFC00> : vector<1xf16>
    # Tile Reduction Loop
    # CHECK: scf.for %[[ITER:.+]] = %[[C0_IDX]] to %[[C4_IDX]] step %[[C1_IDX]]
    # CHECK-SAME: iter_args(%[[ACC:.+]] = %[[INIT]]) -> (vector<1xf16>) {
    # Elementwise
    # CHECK: arith.mulf {{.*}} : vector<2xf16>
    # Local Reduction
    # CHECK: arith.maximumf {{.*}} : f16
    # Global Reduction
    # CHECK: gpu.shuffle  xor %{{.+}}, %[[C1]], %{{.+}} : vector<1xf16>
    # CHECK: arith.maximumf {{.*}} : vector<1xf16>
    # CHECK: gpu.shuffle  xor %{{.+}}, %[[C2]], %{{.+}} : vector<1xf16>
    # CHECK: arith.maximumf {{.*}} : vector<1xf16>
    # CHECK: gpu.shuffle  xor %{{.+}}, %[[C4]], %{{.+}} : vector<1xf16>
    # CHECK: arith.maximumf {{.*}} : vector<1xf16>
    # CHECK: gpu.shuffle  xor %{{.+}}, %[[C8]], %{{.+}} : vector<1xf16>
    # CHECK: arith.maximumf {{.*}} : vector<1xf16>
    # CHECK: gpu.shuffle  xor %{{.+}}, %[[C16]], %{{.+}} : vector<1xf16>
    # CHECK: arith.maximumf {{.*}} : vector<1xf16>
    # CHECK: gpu.shuffle  xor %{{.+}}, %[[C32]], %{{.+}} : vector<1xf16>
    # CHECK: %[[GLOBAL_REDUCE:.+]] = arith.maximumf {{.*}} : vector<1xf16>
    # Accumulator Reduction
    # CHECK: %[[ACC_REDUCE:.+]] = arith.maximumf %[[ACC]], %[[GLOBAL_REDUCE]]
    # CHECK: scf.yield %[[ACC_REDUCE]] : vector<1xf16>


@run_test
def test_tiled_reduce_min():
    M = tkl.sym.M
    N = tkl.sym.N
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(1, 1, 1),
            vector_shapes={M: 1, N: BLOCK_N},
        )
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 1)]
    constraints += [tkw.TilingConstraint(N, BLOCK_N)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M)]

    @tkw.wave(constraints)
    def tiled_reduce_min(
        a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[M, ADDRESS_SPACE, tkl.f16],
    ):
        init_min = tkl.Register[M, tkl.f16](1e6)

        @tkw.iterate(N, init_args=[init_min])
        def repeat(
            partial_min: tkl.Register[M, tkl.f16],
        ) -> tkl.Register[M, tkl.f16]:
            lhs = tkw.read(a)
            rhs = tkw.read(b)
            res = lhs * rhs
            partial_min = tkw.min(res, partial_min, dim=N)
            return partial_min

        tkw.write(repeat, c)

    shape = (256, 512)
    options = WaveCompileOptions(
        subs={
            M: shape[0],
            N: shape[1],
            BLOCK_M: 1,
            BLOCK_N: 128,
            ADDRESS_SPACE: tkl.AddressSpace.GLOBAL_MEMORY.value,
        },
        canonicalize=True,
        compile_to_mlir=True,
    )
    options = set_default_compile_config(options)
    test = wave_compile(options, tiled_reduce_min)
    print(test.asm)

    # CHECK-LABEL: func @tiled_reduce_min
    # CHECK-DAG: %[[C1:.+]] = arith.constant 1 : i32
    # CHECK-DAG: %[[C2:.+]] = arith.constant 2 : i32
    # CHECK-DAG: %[[C4:.+]] = arith.constant 4 : i32
    # CHECK-DAG: %[[C8:.+]] = arith.constant 8 : i32
    # CHECK-DAG: %[[C16:.+]] = arith.constant 16 : i32
    # CHECK-DAG: %[[C32:.+]] = arith.constant 32 : i32
    # CHECK-DAG: %[[C0_IDX:.+]] = arith.constant 0 : index
    # CHECK-DAG: %[[C4_IDX:.+]] = arith.constant 4 : index
    # CHECK-DAG: %[[C1_IDX:.+]] = arith.constant 1 : index
    # CHECK-DAG: %[[INIT:.+]] = arith.constant dense<0x7C00> : vector<1xf16>
    # Tile Reduction Loop
    # CHECK: scf.for %[[ITER:.+]] = %[[C0_IDX]] to %[[C4_IDX]] step %[[C1_IDX]]
    # CHECK-SAME: iter_args(%[[ACC:.+]] = %[[INIT]]) -> (vector<1xf16>) {
    # Elementwise
    # CHECK: arith.mulf {{.*}} : vector<2xf16>
    # Local Reduction
    # CHECK: arith.minimumf {{.*}} : f16
    # Global Reduction
    # CHECK: gpu.shuffle  xor %{{.+}}, %[[C1]], %{{.+}} : vector<1xf16>
    # CHECK: arith.minimumf {{.*}} : vector<1xf16>
    # CHECK: gpu.shuffle  xor %{{.+}}, %[[C2]], %{{.+}} : vector<1xf16>
    # CHECK: arith.minimumf {{.*}} : vector<1xf16>
    # CHECK: gpu.shuffle  xor %{{.+}}, %[[C4]], %{{.+}} : vector<1xf16>
    # CHECK: arith.minimumf {{.*}} : vector<1xf16>
    # CHECK: gpu.shuffle  xor %{{.+}}, %[[C8]], %{{.+}} : vector<1xf16>
    # CHECK: arith.minimumf {{.*}} : vector<1xf16>
    # CHECK: gpu.shuffle  xor %{{.+}}, %[[C16]], %{{.+}} : vector<1xf16>
    # CHECK: arith.minimumf {{.*}} : vector<1xf16>
    # CHECK: gpu.shuffle  xor %{{.+}}, %[[C32]], %{{.+}} : vector<1xf16>
    # CHECK: %[[GLOBAL_REDUCE:.+]] = arith.minimumf {{.*}} : vector<1xf16>
    # Accumulator Reduction
    # CHECK: %[[ACC_REDUCE:.+]] = arith.minimumf %[[ACC]], %[[GLOBAL_REDUCE]]
    # CHECK: scf.yield %[[ACC_REDUCE]] : vector<1xf16>


@run_test
def test_tiled_reduce_min_unaligned():
    M = tkl.sym.M
    N = tkl.sym.N
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(1, 1, 1),
            vector_shapes={M: 1, N: BLOCK_N},
        )
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 1)]
    constraints += [tkw.TilingConstraint(N, BLOCK_N)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M)]

    @tkw.wave(constraints)
    def tiled_reduce_min_unaligned(
        a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[M, ADDRESS_SPACE, tkl.f16],
    ):
        init_min = tkl.Register[M, tkl.f16](1e6)

        @tkw.iterate(N, init_args=[init_min])
        def repeat(
            partial_min: tkl.Register[M, tkl.f16],
        ) -> tkl.Register[M, tkl.f16]:
            lhs = tkw.read(a)
            rhs = tkw.read(b)
            res = lhs * rhs
            partial_min = tkw.min(res, partial_min, dim=N)
            return partial_min

        tkw.write(repeat, c)

    shape = (256, 527)
    options = WaveCompileOptions(
        subs={
            M: shape[0],
            N: shape[1],
            BLOCK_M: 1,
            BLOCK_N: 128,
            ADDRESS_SPACE: tkl.AddressSpace.GLOBAL_MEMORY.value,
        },
        canonicalize=True,
        compile_to_mlir=True,
    )
    options = set_default_compile_config(options)
    test = wave_compile(options, tiled_reduce_min_unaligned)
    print(test.asm)

    # CHECK-LABEL: test_tiled_reduce_min_unaligned
    # CHECK-DAG:     #[[map1:.*]] = affine_map<()[s0, s1] -> (s0 * 128 + s1 * 2 - (s1 floordiv 64) * 128)>
    # CHECK:       func @tiled_reduce_min_unaligned
    # CHECK-DAG:     %[[cst_0:.*]] = arith.constant dense<527> : vector<2xindex>
    # CHECK-DAG:     %[[cst_1:.*]] = arith.constant dense<[0, 1]> : vector<2xindex>
    # CHECK-DAG:     %[[C0:.*]] = arith.constant 0 : index
    # CHECK-DAG:     %[[C1:.*]] = arith.constant 1 : index
    # CHECK-DAG:     %[[C5:.*]] = arith.constant 5 : index
    # CHECK:         %[[THREAD_ID_X:.+]] = gpu.thread_id  x
    # Tiled Reduction Loop
    # CHECK:         scf.for %[[ITER:.*]] = %[[C0]] to %[[C5]] step %[[C1]]
    # CHECK:           %[[D6:.*]] = affine.apply #[[map1]]()[%[[ITER]], %[[THREAD_ID_X]]]
    # CHECK:           %[[D7:.*]] = vector.splat %[[D6]] : vector<2xindex>
    # CHECK:           %[[D8:.*]] = arith.addi %[[D7]], %[[cst_1]] overflow<nsw, nuw> : vector<2xindex>
    # CHECK:           %[[D9:.*]] = arith.cmpi slt, %[[D8]], %[[cst_0]] : vector<2xindex>
    # CHECK-COUNT-2:   vector.maskedload %{{.*}}[%{{.*}}, %[[D6]]], %[[D9]]


# This test is to ensure that the we can handle multiple IV in reduction properly.
@run_test
def test_multiple_reduction_iv():
    M = tkl.sym.M
    N = tkl.sym.N
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    ELEMS_PER_THREAD = tkl.sym.ELEMS_PER_THREAD
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(1, 1, 1),
            vector_shapes={M: 1, N: BLOCK_N},
        )
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 1)]
    constraints += [tkw.WorkgroupConstraint(N, N, 0)]
    constraints += [tkw.TilingConstraint(N, BLOCK_N)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N)]

    @tkw.wave(constraints)
    def test(
        a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[M, ADDRESS_SPACE, tkl.f16],
        d: tkl.Memory[M, ADDRESS_SPACE, tkl.f16],
    ):
        init_max = tkl.Register[M, tkl.f16](-1e6)
        init_sum = tkl.Register[M, tkl.f16](0)

        @tkw.iterate(N, init_args=[init_max, init_sum])
        def repeat(
            partial_max: tkl.Register[M, tkl.f16],
            partial_sum: tkl.Register[M, tkl.f16],
        ) -> tkl.Register[M, tkl.f16]:
            lhs = tkw.read(a, elements_per_thread=ELEMS_PER_THREAD)
            partial_max = tkw.max(lhs, partial_max, dim=N)
            partial_sum = tkw.sum(lhs, partial_sum, dim=N)
            return partial_max, partial_sum

        res_max, res_sum = repeat
        tkw.write(res_max, c, elements_per_thread=1)
        tkw.write(res_sum, d, elements_per_thread=1)

    shape = (256, 512)
    options = WaveCompileOptions(
        subs={
            M: shape[0],
            N: shape[1],
            BLOCK_M: 2,
            BLOCK_N: 128,
            ELEMS_PER_THREAD: 2,
            ADDRESS_SPACE: tkl.AddressSpace.GLOBAL_MEMORY.value,
        },
        canonicalize=True,
        compile_to_mlir=True,
    )
    options = set_default_compile_config(options)
    test = wave_compile(options, test)
    print(test.asm)

    # CHECK-LABEL: test_multiple_reduction_iv
    # CHECK:       func @test
    # CHECK-DAG: %[[C0_IDX:.+]] = arith.constant 0 : index
    # CHECK-DAG: %[[C4_IDX:.+]] = arith.constant 4 : index
    # CHECK-DAG: %[[C1_IDX:.+]] = arith.constant 1 : index
    # CHECK-DAG: %[[INIT_MAX:.+]] = arith.constant dense<0xFC00> : vector<1xf16>
    # CHECK-DAG: %[[INIT_SUM:.+]] = arith.constant dense<0.000000e+00> : vector<1xf16>

    # Tile Reduction Loop
    # CHECK: %[[TILED:.+]]:4 = scf.for %[[ITER:.+]] = %[[C0_IDX]] to %[[C4_IDX]] step %[[C1_IDX]]
    # CHECK-SAME: iter_args(%[[ACC0:.+]] = %[[INIT_MAX]], %[[ACC1:.+]] = %[[INIT_MAX]], %[[ACC2:.+]] = %[[INIT_SUM]], %[[ACC3:.+]] = %[[INIT_SUM]])
    # CHECK-SAME: -> (vector<1xf16>, vector<1xf16>, vector<1xf16>, vector<1xf16>) {
    # 1st Expanded Local Max Reduction
    # CHECK: arith.maximumf {{.*}} : f16
    # 1st Expanded Global Max Reduction
    # CHECK-COUNT-6: gpu.shuffle  xor
    # 1st Expanded Accumulator Max Reduction
    # CHECK: %[[ACC_MAX_0:.+]] = arith.maximumf %[[ACC0]], %{{.*}}

    # 2nd Expanded Local Max Reduction
    # CHECK: arith.maximumf {{.*}} : f16
    # 2nd Expanded Global Max Reduction
    # CHECK-COUNT-6: gpu.shuffle  xor
    # 2nd Expanded Accumulator Max Reduction
    # CHECK: %[[ACC_MAX_1:.+]] = arith.maximumf %[[ACC1]], %{{.*}}

    # 1st Expanded Local Sum Reduction
    # CHECK: arith.addf {{.*}} : f16
    # 1st Expanded Global Sum Reduction
    # CHECK-COUNT-6: gpu.shuffle  xor
    # 1st Expanded Accumulator Sum Reduction
    # CHECK: %[[ACC_SUM_0:.+]] = arith.addf %[[ACC2]], %{{.*}}

    # 2nd Expanded Local Sum Reduction
    # CHECK: arith.addf {{.*}} : f16
    # 2nd Expanded Global Sum Reduction
    # CHECK-COUNT-6: gpu.shuffle  xor
    # 2nd Expanded Accumulator Sum Reduction
    # CHECK: %[[ACC_SUM_1:.+]] = arith.addf %[[ACC3]], %{{.*}}

    # CHECK: scf.yield %[[ACC_MAX_0]], %[[ACC_MAX_1]], %[[ACC_SUM_0]], %[[ACC_SUM_1]]


# This test is used to ensure:
# 1. ReduceOp has correct symbolic shape for thread shape analysis.
# 2. We can propagate the resolved indexing from broadcast.(in this case from sub to exp2.)
@run_test
def test_reduce_propagate_broadcast():
    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(1, 1, 1),
            vector_shapes={M: 1, N: BLOCK_N},
        )
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 1)]
    constraints += [tkw.WorkgroupConstraint(N, N, 0)]
    constraints += [tkw.TilingConstraint(N, BLOCK_N)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N)]

    @tkw.wave(constraints)
    def test(
        a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f32],
        c: tkl.Memory[M, ADDRESS_SPACE, tkl.f32],
    ):
        init_max = tkl.Register[M, tkl.f32](-1e6)
        init_sum = tkl.Register[M, tkl.f32](0)

        @tkw.iterate(N, init_args=[init_max, init_sum])
        def repeat(
            partial_max: tkl.Register[M, tkl.f32],
            partial_sum: tkl.Register[M, tkl.f32],
        ) -> tkl.Register[M, tkl.f32]:
            src = tkw.read(a, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            m_src = tkw.max(src, partial_max, dim=N)
            exp_d = tkw.exp2(src - m_src)
            sum_d = tkw.sum(exp_d, partial_sum, dim=N)
            return m_src, sum_d

        res_max, res_sum = repeat
        tkw.write(res_sum, c, elements_per_thread=1)

    shape = (256, 1024)
    options = WaveCompileOptions(
        subs={
            M: shape[0],
            N: shape[1],
            BLOCK_M: 1,
            BLOCK_N: 128,
            LOAD_ELEMS_PER_THREAD: 2,
            ADDRESS_SPACE: tkl.AddressSpace.GLOBAL_MEMORY.value,
        },
        canonicalize=True,
        compile_to_mlir=True,
    )
    options = set_default_compile_config(options)
    test = wave_compile(options, test)
    print(test.asm)

    # CHECK-LABEL: test_reduce_propagate_broadcast
    # CHECK:       func @test
    # CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
    # CHECK-DAG: %[[C8:.+]] = arith.constant 8 : index
    # CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
    # CHECK: %[[CST:.+]] = arith.constant dense<0.000000e+00> : vector<1xf32>
    # CHECK: scf.for %{{.*}} = %[[C0]] to %[[C8]] step %[[C1]]
    # CHECK-COUNT-7: arith.maximumf
    # CHECK: %[[ACC_MAX:.+]] = arith.maximumf
    # CHECK: %[[EXTRACT:.+]] = vector.extract %[[ACC_MAX]][0] : f32 from vector<1xf32>
    # CHECK: %[[BROADCAST:.+]] = vector.splat %[[EXTRACT]] : vector<2xf32>
    # CHECK: %[[SUBF:.+]] = arith.subf %{{.+}}, %[[BROADCAST]] : vector<2xf32>
    # CHECK: %[[EXP2:.+]] = math.exp2 %[[SUBF]] : vector<2xf32>
    # CHECK: %[[EXP2_SLICE_0:.+]] = vector.extract %[[EXP2]][0] : f32 from vector<2xf32>
    # CHECK: %[[EXP2_SLICE_1:.+]] = vector.extract %[[EXP2]][1] : f32 from vector<2xf32>
    # CHECK: arith.addf %[[EXP2_SLICE_0]], %[[EXP2_SLICE_1]]
    # CHECK-COUNT-6: gpu.shuffle xor


@run_test
def test_block_reduce_sum():
    round_to_divisible = lambda src, denom: sympy.ceiling(src / denom) * denom
    M = tkl.sym.M
    N = tkl.sym.N
    wave_size = 64
    num_waves = 4
    BLOCK_M = 1

    # Distribute N dim across num_waves, and pad to disivible by wave_size.
    ELEMS_PER_WAVE = round_to_divisible(sympy.ceiling(N / num_waves), wave_size)
    # Minimum number of elems per wave should be size of wave.
    ELEMS_PER_WAVE = sympy.Max(ELEMS_PER_WAVE, wave_size)
    BLOCK_N = ELEMS_PER_WAVE * num_waves
    ELEMS_PER_THREAD = ELEMS_PER_WAVE // wave_size
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=wave_size,
            waves_per_block=(num_waves, 1, 1),
            vector_shapes={M: 1, N: ELEMS_PER_WAVE},
        )
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 1)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 0)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M)]
    constraints += [tkw.WaveConstraint(N, ELEMS_PER_WAVE)]

    @tkw.wave(constraints)
    def test_block_reduce_sum(
        a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[M, ADDRESS_SPACE, tkl.f16],
    ):
        lhs = tkw.read(a, elements_per_thread=ELEMS_PER_THREAD)
        rhs = tkw.read(b, elements_per_thread=ELEMS_PER_THREAD)
        res = lhs * rhs
        res = tkw.sum(res, dim=N, block=True)
        tkw.write(res, c, elements_per_thread=1)

    shape = (4, 512)
    options = WaveCompileOptions(
        subs={
            M: shape[0],
            N: shape[1],
            ADDRESS_SPACE: tkl.AddressSpace.GLOBAL_MEMORY.value,
        },
        canonicalize=True,
        compile_to_mlir=True,
    )
    options = set_default_compile_config(options)
    test_block_reduce_sum = wave_compile(options, test_block_reduce_sum)
    print(test_block_reduce_sum.asm)

    # CHECK-LABEL: test_block_reduce_sum
    # CHECK-DAG: #[[map2:.+]] = affine_map<()[s0] -> ((s0 floordiv 64) mod 4)>
    # CHECK-DAG: %[[c0:.+]] = arith.constant 0 : index
    # CHECK: %[[tid:.+]] = gpu.thread_id  x
    # CHECK: %[[alloc:.+]] = memref.alloc() : memref<8xi8, #gpu.address_space<workgroup>>

    # Local Reduce
    # CHECK-COUNT-2: vector.extract
    # CHECK-NEXT: arith.addf

    # Global Reduce
    # CHECK-COUNT-6: gpu.shuffle  xor
    # CHECK-NEXT: %[[global_reduce:.+]] = arith.addf

    # Write partial wave result into shared memory to be accessible by other waves.
    # CHECK: %[[view:.+]] = memref.view %[[alloc]][%[[c0]]][] : memref<8xi8, #gpu.address_space<workgroup>> to memref<4xf16, #gpu.address_space<workgroup>>
    # CHECK: scf.if {{.*}} {
    # CHECK:    %[[wave_id:.+]] = affine.apply #[[map2]]()[%[[tid]]]
    # CHECK:    vector.store %[[global_reduce]], %[[view]][%[[wave_id]]] : memref<4xf16, #gpu.address_space<workgroup>>, vector<1xf16>
    # CHECK: }
    # CHECK-NEXT: amdgpu.lds_barrier

    # Get all partial wave results and locally reduce
    # CHECK: %[[wave_res:.+]] = vector.load %[[view]]
    # CHECK: vector.extract %[[wave_res]][0] : f16 from vector<4xf16>
    # CHECK: vector.extract %[[wave_res]][1] : f16 from vector<4xf16>
    # CHECK-NEXT: arith.addf
    # CHECK: vector.extract %[[wave_res]][2] : f16 from vector<4xf16>
    # CHECK-NEXT: arith.addf
    # CHECK: vector.extract %[[wave_res]][3] : f16 from vector<4xf16>
    # CHECK-NEXT: arith.addf


@run_test
def test_explicit_broadcast():
    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(1, 1, 1),
            vector_shapes={M: 1, N: BLOCK_N},
        )
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 1)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 0)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N)]

    @tkw.wave(constraints)
    def explicit_broadcast(
        a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[M, ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
    ):
        lhs = tkw.read(a, elements_per_thread=LOAD_ELEMS_PER_THREAD)
        rhs = tkw.read(b, elements_per_thread=1)
        broadcast_rhs = tkw.broadcast(rhs, (M, N))
        res = lhs + broadcast_rhs
        tkw.write(res, c, elements_per_thread=STORE_ELEMS_PER_THREAD)

    shape = (256, 128)
    options = WaveCompileOptions(
        subs={
            M: shape[0],
            N: shape[1],
            BLOCK_M: 2,
            BLOCK_N: 128,
            LOAD_ELEMS_PER_THREAD: 2,
            STORE_ELEMS_PER_THREAD: 2,
            ADDRESS_SPACE: tkl.AddressSpace.GLOBAL_MEMORY.value,
        },
        canonicalize=True,
        compile_to_mlir=True,
    )
    options = set_default_compile_config(options)
    explicit_broadcast = wave_compile(options, explicit_broadcast)
    print(explicit_broadcast.asm)

    # CHECK-LABEL: test_explicit_broadcast
    # CHECK-DAG:     #[[map0:.+]] = affine_map<()[s0] -> (s0 * 2)>
    # CHECK-DAG:     #[[map1:.+]] = affine_map<()[s0] -> (s0 * 2 + 1)>
    # CHECK:       func.func @explicit_broadcast
    # CHECK-SAME: (%[[ARG0:.+]]: !stream.binding, %[[ARG1:.+]]: !stream.binding, %{{.+}}: !stream.binding)
    # CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
    # CHECK: %[[workgroup_id_1:.*]] = stream.dispatch.workgroup.id[1] : index
    # CHECK: %[[thread_id_x:.*]] = gpu.thread_id  x

    # Slicing LHS
    # CHECK: %[[LHS:.+]] = stream.binding.subspan %[[ARG0]][%[[C0]]] : !stream.binding -> memref<256x128xf16
    # CHECK: %[[X_SLICE_0:.+]] = affine.apply #[[map0]]()[%[[workgroup_id_1]]]
    # CHECK: %[[LHS_0:.+]] = vector.load %[[LHS]][%[[X_SLICE_0]], %[[Y_SLICE:.+]]] : memref<256x128xf16, strided<[128, 1], offset: ?>>, vector<2xf16>
    # CHECK: %[[X_SLICE_1:.+]] = affine.apply #[[map1]]()[%[[workgroup_id_1]]]
    # CHECK: %[[LHS_1:.+]] = vector.load %[[LHS]][%[[X_SLICE_1]], %[[Y_SLICE]]] : memref<256x128xf16, strided<[128, 1], offset: ?>>, vector<2xf16>

    # Slicing RHS
    # CHECK: %[[RHS:.+]] = stream.binding.subspan %[[ARG1]][%[[C0]]] : !stream.binding -> memref<256xf16
    # CHECK: %[[RHS_0:.+]] = memref.load %[[RHS]][%[[X_SLICE_0]]] : memref<256xf16, strided<[1], offset: ?>>
    # CHECK: %[[RHS_1:.+]] = memref.load %[[RHS]][%[[X_SLICE_1]]] : memref<256xf16, strided<[1], offset: ?>>

    # 1st Broadcast RHS
    # CHECK: %[[BCAST_RHS_0:.+]] = vector.splat %[[RHS_0]] : vector<2xf16>

    # 2nd Broadcast RHS
    # CHECK: %[[BCAST_RHS_1:.+]] = vector.splat %[[RHS_1]] : vector<2xf16>

    # Broadcast-ADD RHS
    # CHECK: arith.addf %[[LHS_0]], %[[BCAST_RHS_0]] : vector<2xf16>
    # CHECK: arith.addf %[[LHS_1]], %[[BCAST_RHS_1]] : vector<2xf16>


@run_test
def test_broadcast_add():
    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(1, 1, 1),
            vector_shapes={M: 1, N: BLOCK_N},
        )
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 1)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 0)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N)]

    @tkw.wave(constraints)
    def broadcast_add(
        a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[M, ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
    ):
        lhs = tkw.read(a, elements_per_thread=LOAD_ELEMS_PER_THREAD)
        rhs = tkw.read(b, elements_per_thread=1)
        res = lhs + rhs
        tkw.write(res, c, elements_per_thread=STORE_ELEMS_PER_THREAD)

    shape = (256, 128)
    options = WaveCompileOptions(
        subs={
            M: shape[0],
            N: shape[1],
            BLOCK_M: 2,
            BLOCK_N: 128,
            LOAD_ELEMS_PER_THREAD: 2,
            STORE_ELEMS_PER_THREAD: 2,
            ADDRESS_SPACE: tkl.AddressSpace.GLOBAL_MEMORY.value,
        },
        canonicalize=True,
        compile_to_mlir=True,
    )
    options = set_default_compile_config(options)
    broadcast_add = wave_compile(options, broadcast_add)
    print(broadcast_add.asm)

    # CHECK-LABEL: test_broadcast_add
    # CHECK-DAG:     #[[map0:.*]] = affine_map<()[s0] -> (s0 * 2)>
    # CHECK-DAG:     #[[map2:.*]] = affine_map<()[s0] -> (s0 * 2 + 1)>
    # CHECK: func.func @broadcast_add
    # CHECK-SAME: (%[[ARG0:.+]]: !stream.binding, %[[ARG1:.+]]: !stream.binding, %{{.+}}: !stream.binding)
    # CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
    # CHECK: %[[workgroup_id_1:.*]] = stream.dispatch.workgroup.id[1] : index
    # CHECK: %[[thread_id_x:.*]] = gpu.thread_id  x

    # Slicing LHS
    # CHECK: %[[LHS:.+]] = stream.binding.subspan %[[ARG0]][%[[C0]]] : !stream.binding -> memref<256x128xf16
    # CHECK: %[[X_SLICE_0:.+]] = affine.apply #[[map0]]()[%[[workgroup_id_1]]]
    # CHECK: %[[LHS_0:.+]] = vector.load %[[LHS]][%[[X_SLICE_0]], %[[Y_SLICE:.+]]] : memref<256x128xf16, strided<[128, 1], offset: ?>>, vector<2xf16>
    # CHECK: %[[X_SLICE_1:.+]] = affine.apply #[[map2]]()[%[[workgroup_id_1]]]
    # CHECK: %[[LHS_1:.+]] = vector.load %[[LHS]][%[[X_SLICE_1]], %[[Y_SLICE]]] : memref<256x128xf16, strided<[128, 1], offset: ?>>, vector<2xf16>

    # Slicing RHS
    # CHECK: %[[RHS:.+]] = stream.binding.subspan %[[ARG1]][%[[C0]]] : !stream.binding -> memref<256xf16
    # CHECK: %[[RHS_0:.+]] = memref.load %[[RHS]][%[[X_SLICE_0]]] : memref<256xf16, strided<[1], offset: ?>>
    # CHECK: %[[RHS_1:.+]] = memref.load %[[RHS]][%[[X_SLICE_1]]] : memref<256xf16, strided<[1], offset: ?>>

    # 1st Broadcast RHS
    # CHECK: %[[BCAST_RHS_0:.+]] = vector.splat %[[RHS_0]] : vector<2xf16>

    # 2nd Broadcast RHS
    # CHECK: %[[BCAST_RHS_1:.+]] = vector.splat %[[RHS_1]] : vector<2xf16>

    # Broadcast-ADD RHS
    # CHECK: arith.addf %[[LHS_0]], %[[BCAST_RHS_0]] : vector<2xf16>
    # CHECK: arith.addf %[[LHS_1]], %[[BCAST_RHS_1]] : vector<2xf16>


@run_test
def test_binary_lowerings():
    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=64, waves_per_block=(1, 1, 1), vector_shapes={M: 16, N: 16}
        )
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 2)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]

    @tkw.wave(constraints)
    def binary_lowerings(
        a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
    ):
        a_reg = tkw.read(a, elements_per_thread=4)
        b_reg = tkw.read(b, elements_per_thread=4)
        res = a_reg - b_reg
        res = res * a_reg
        res = res / b_reg
        res = tkw.minimum(a_reg, b_reg)
        res = tkw.atan2(res, a_reg)
        tkw.write(res, a, elements_per_thread=4)

    binary_lowerings = wave_compile(get_wave_compile_options(), binary_lowerings)
    print(binary_lowerings.asm)

    # CHECK-LABEL: func @binary_lowerings
    # CHECK: %[[SUB:.+]] = arith.subf
    # CHECK: %[[MUL:.+]] = arith.mulf %[[SUB]]
    # CHECK: %[[DIV:.+]] = arith.divf %[[MUL]]
    # CHECK: %[[MINIMUM:.+]] = arith.minimumf
    # CHECK: %[[ATAN2:.+]] = math.atan2 %[[MINIMUM]]


@run_test
def test_int_comparisons():
    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=64, waves_per_block=(1, 1, 1), vector_shapes={M: 16, N: 16}
        )
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 2)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]

    @tkw.wave(constraints)
    def cmp_lowerings(
        a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.i32],
        b: tkl.Memory[M, N, ADDRESS_SPACE, tkl.i32],
    ):
        a_reg = tkw.read(a, elements_per_thread=4)
        b_reg = tkw.read(b, elements_per_thread=4)
        sgt = a_reg > b_reg
        s1 = tkw.select(sgt, a_reg, b_reg)
        slt = a_reg < b_reg
        s2 = tkw.select(slt, a_reg, b_reg)
        sge = s1 >= s2
        s3 = tkw.select(sge, s1, s2)
        sle = s1 <= s2
        s4 = tkw.select(sle, s1, s2)
        res = s1 + s2 + s3 + s4
        tkw.write(res, a, elements_per_thread=4)

    cmp_lowerings = wave_compile(get_wave_compile_options(), cmp_lowerings)
    print(cmp_lowerings.asm)

    # CHECK-LABEL: @cmp_lowerings
    # CHECK: arith.cmpi sgt
    # CHECK: arith.select
    # CHECK: arith.cmpi slt
    # CHECK: arith.select
    # CHECK: arith.cmpi sge
    # CHECK: arith.select


@run_test
def test_verbose_int_comparisons():
    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=64, waves_per_block=(1, 1, 1), vector_shapes={M: 16, N: 16}
        )
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 2)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]

    @tkw.wave(constraints)
    def verbose_cmp_lowerings(
        a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.i32],
        b: tkl.Memory[M, N, ADDRESS_SPACE, tkl.i32],
        c: tkl.Memory[M, N, ADDRESS_SPACE, tkw.i1],
    ):
        a_reg = tkw.read(a, elements_per_thread=4)
        b_reg = tkw.read(b, elements_per_thread=4)
        sgt = tkw.gt(a_reg, b_reg)
        s1 = tkw.select(sgt, a_reg, b_reg)
        slt = tkw.lt(a_reg, b_reg)
        s2 = tkw.select(slt, a_reg, b_reg)
        sge = tkw.ge(s1, s2)
        s3 = tkw.select(sge, s1, s2)
        sle = tkw.le(s1, s2)
        s4 = tkw.select(sle, s1, s2)
        res_eq = tkw.eq(s3, s4)
        res = s1 + s2 + s3 + s4
        tkw.write(res, a, elements_per_thread=4)
        tkw.write(res_eq, c, elements_per_thread=4)

    verbose_cmp_lowerings = wave_compile(
        get_wave_compile_options(), verbose_cmp_lowerings
    )
    print(verbose_cmp_lowerings.asm)

    # CHECK-LABEL: @verbose_cmp_lowerings
    # CHECK: arith.cmpi sgt
    # CHECK: arith.select
    # CHECK: arith.cmpi slt
    # CHECK: arith.select
    # CHECK: arith.cmpi sge
    # CHECK: arith.select
    # CHECK: arith.cmpi eq


@run_test
def test_float_comparisons():
    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=64, waves_per_block=(1, 1, 1), vector_shapes={M: 16, N: 16}
        )
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 2)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]

    @tkw.wave(constraints)
    def cmpf_lowerings(
        a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f32],
        b: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f32],
    ):
        a_reg = tkw.read(a, elements_per_thread=4)
        b_reg = tkw.read(b, elements_per_thread=4)
        sgt = a_reg > b_reg
        s1 = tkw.select(sgt, a_reg, b_reg)
        slt = a_reg < b_reg
        s2 = tkw.select(slt, a_reg, b_reg)
        sge = s1 >= s2
        s3 = tkw.select(sge, s1, s2)
        sle = s1 <= s2
        s4 = tkw.select(sle, s1, s2)
        res = s1 + s2 + s3 + s4
        tkw.write(res, a, elements_per_thread=4)

    cmpf_lowerings = wave_compile(get_wave_compile_options(), cmpf_lowerings)
    print(cmpf_lowerings.asm)

    # CHECK-LABEL: @cmpf_lowerings
    # CHECK: arith.cmpf ogt
    # CHECK: arith.select
    # CHECK: arith.cmpf olt
    # CHECK: arith.select
    # CHECK: arith.cmpf oge
    # CHECK: arith.select
    # CHECK: arith.cmpf ole
    # CHECK: arith.select


# TODO: Something is broken in codegen and we are getting int in place of fx.Node
# @launch
@pytest.mark.skip(reason="getitem: Currently only stub implementation")
def test_get_item():
    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=64, waves_per_block=(1, 1, 1), vector_shapes={M: 16, N: 16}
        ),
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 2)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]

    @tkw.wave(constraints)
    def get_item(a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16]):
        res = a[0]
        tkw.write(res, a, elements_per_thread=4)

    with pytest.raises(
        NotImplementedError, match="getitem: Currently only stub implementation"
    ):
        get_item(a)


# TODO: Add more tests once we have more than a stub implementation.


@run_test
def test_register_codegen_i32():
    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(1, 1, 1),
            vector_shapes={M: 1, N: 27},
        )
    ]
    constraints += [tkw.WorkgroupConstraint(M, 1, 1)]
    constraints += [tkw.WorkgroupConstraint(N, 27, 0)]
    constraints += [tkw.WaveConstraint(M, 1)]
    constraints += [tkw.WaveConstraint(N, 27)]

    @tkw.wave(constraints)
    def register_codegen_i32(
        b: tkl.Memory[M, N, ADDRESS_SPACE, tkl.i32],
    ):
        tid = tkl.Register[M, N, tkl.i32](THREAD_0)
        reg = tkl.Register[M, N, tkl.i32](3)
        res = tid + reg
        tkw.write(res, b)

    options = WaveCompileOptions(
        subs={
            M: 1,
            N: 27,
            ADDRESS_SPACE: tkl.AddressSpace.GLOBAL_MEMORY.value,
        },
        canonicalize=True,
        compile_to_mlir=True,
    )
    register_codegen_i32 = wave_compile(options, register_codegen_i32)
    print(register_codegen_i32.asm)

    # CHECK-LABEL: @register_codegen_i32

    # Setting up constant register
    # CHECK:  %[[CST:.+]] = arith.constant dense<3> : vector<1xi32>

    # Setting up THREAD_0 as a register
    # CHECK:  %[[tid:.+]] = gpu.thread_id  x
    # CHECK:  %[[cast:.+]] = arith.index_cast %[[tid]] : index to i32
    # CHECK:  %[[vector_tid:.+]] = vector.splat %[[cast]] : vector<1xi32>

    # Check for addition operation of two registers
    # CHECK: arith.addi %[[vector_tid]], %[[CST]] : vector<1xi32>


@run_test
def test_scalar_codegen_f32():
    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(1, 1, 1),
            vector_shapes={M: 1, N: 27},
        )
    ]
    constraints += [tkw.WorkgroupConstraint(M, 1, 1)]
    constraints += [tkw.WorkgroupConstraint(N, 27, 0)]
    constraints += [tkw.WaveConstraint(M, 1)]
    constraints += [tkw.WaveConstraint(N, 27)]

    @tkw.wave(constraints)
    def scalar_codegen_f32(
        a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f32],
        c: tkl.f32,  # type: ignore
        d: tkl.f32,  # type: ignore
        b: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f32],
    ):
        res = tkw.read(a)
        c = tkw.broadcast(c, target_shape=[M, N])
        d = tkw.broadcast(d, target_shape=[M, N])
        temp = tkl.Register[M, N, tkl.f32](1.0) * c
        res = res + temp + d
        tkw.write(res, b)

    options = WaveCompileOptions(
        subs={
            M: 1,
            N: 27,
            ADDRESS_SPACE: tkl.AddressSpace.GLOBAL_MEMORY.value,
        },
        canonicalize=True,
        compile_to_mlir=True,
    )
    scalar_codegen_f32 = wave_compile(options, scalar_codegen_f32)
    print(scalar_codegen_f32.asm)

    # Passed scalars' dtype
    # CHECK: func.func @scalar_codegen_f32(
    # CHECK-SAME: %arg2: f32, %arg3: f32)

    # Broadcast and add
    # CHECK: vector.splat %arg2 : vector<1xf32>
    # CHECK: vector.splat %arg3 : vector<1xf32>
    # CHECK: arith.addf

    # Final dispatch args dtype
    # CHECK: flow.dispatch @scalar_codegen_f32::@scalar_codegen_f32(
    # CHECK-SAME: %arg0, %arg1, %arg2, %arg3)


@run_test
def test_scalar_codegen_i32():
    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(1, 1, 1),
            vector_shapes={M: 1, N: 27},
        )
    ]
    constraints += [tkw.WorkgroupConstraint(M, 1, 1)]
    constraints += [tkw.WorkgroupConstraint(N, 27, 0)]
    constraints += [tkw.WaveConstraint(M, 1)]
    constraints += [tkw.WaveConstraint(N, 27)]

    @tkw.wave(constraints)
    def scalar_codegen_i32(
        a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.i32],
        c: tkl.i32,  # type: ignore
        d: tkl.i32,  # type: ignore
        b: tkl.Memory[M, N, ADDRESS_SPACE, tkl.i32],
    ):
        res = tkw.read(a)
        c = tkw.broadcast(c, target_shape=[M, N])
        d = tkw.broadcast(d, target_shape=[M, N])
        temp = tkl.Register[M, N, tkl.i32](1) * c
        res = res + temp + d
        tkw.write(res, b)

    options = WaveCompileOptions(
        subs={
            M: 1,
            N: 27,
            ADDRESS_SPACE: tkl.AddressSpace.GLOBAL_MEMORY.value,
        },
        canonicalize=True,
        compile_to_mlir=True,
    )
    scalar_codegen_i32 = wave_compile(options, scalar_codegen_i32)
    print(scalar_codegen_i32.asm)

    # Passed scalars' dtype: i32
    # CHECK: func.func @scalar_codegen_i32(
    # CHECK-SAME: %arg2: i32, %arg3: i32)

    # Broadcast and add
    # CHECK: vector.splat %arg2 : vector<1xi32>
    # CHECK: vector.splat %arg3 : vector<1xi32>
    # CHECK: arith.addi

    # Final dispatch args dtype
    # CHECK: flow.dispatch @scalar_codegen_i32::@scalar_codegen_i32(
    # CHECK-SAME: %arg0, %arg1, %arg2, %arg3)


#  This kernel copies of data from a into b if tid.x < threshold.
#  This test is important to ensure:
#  1. tkw.Scalar can handle index expressions correctly.
#  2. Scalars in Wave can be used for comparison/binaryOps
#     as well as on select ops.
@run_test
def test_scalar_cond_copy():
    M = tkl.sym.M
    N = tkl.sym.N
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    wave_size = 64
    BLOCK_M = 1
    # Tile size cannot be dynamic, so we use a fixed value here.
    BLOCK_N = 64

    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=wave_size,
            waves_per_block=(1, 1, 1),
            vector_shapes={M: BLOCK_M, N: BLOCK_N},
        )
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 1)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 0)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N)]

    thresh_value = 12

    @tkw.wave(constraints)
    def scalar_cond_copy(
        a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
    ):
        zero = tkw.scalar(0.0, tkl.f16)
        one = tkw.scalar(1.0, tkl.f16)

        tid = tkw.scalar(THREAD_0, tkl.i32)
        thresh = tkw.scalar(thresh_value, tkl.i32)

        mask = tkw.select(tid < thresh, one, zero)
        mask_broadcast = tkw.broadcast(mask, target_shape=[M, N])

        a_reg = tkw.read(a)
        res = a_reg * mask_broadcast
        tkw.write(res, b)

    options = WaveCompileOptions(
        subs={
            M: 1,
            N: 64,
            ADDRESS_SPACE: tkl.AddressSpace.GLOBAL_MEMORY.value,
        },
        canonicalize=True,
        compile_to_mlir=True,
    )
    scalar_cond_copy = wave_compile(options, scalar_cond_copy)
    print(scalar_cond_copy.asm)

    # CHECK-LABEL: @scalar_cond_copy

    # mask values
    # CHECK: %[[one:.+]] = arith.constant 1.000000e+00 : f16
    # CHECK: %[[zero:.+]] = arith.constant 0.000000e+00 : f16

    # Condition and mask selection
    # CHECK: %[[tidx:.+]] = gpu.thread_id  x
    # CHECK: %[[tidx_i32:.+]] = arith.index_cast %[[tidx]] : index to i32
    # CHECK: %[[cond:.+]] = arith.cmpi slt, %[[tidx_i32]], %c12
    # CHECK: %[[mask:.+]] = arith.select %[[cond]], %cst, %cst_0 : f16
    # CHECK: %[[splat_mask:.+]] = vector.splat %[[mask]] : vector<1xf16>

    # Apply mask
    # CHECK: arith.mulf {{.*}}, %[[splat_mask]]


@run_test
def test_scanop_cumsum():
    M = tkl.sym.M
    N = tkl.sym.N
    BLOCK_M = 1
    BLOCK_N = 64
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(1, 1, 1),
            vector_shapes={M: 1, N: 64},
        )
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 1)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 0)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N)]

    @tkw.wave(constraints)
    def scanop_cumsum(
        a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, tkl.f16],
    ):
        lhs = tkw.read(a)
        res = tkw.cumsum(lhs, dim=N)
        tkw.write(res, c)

    options = WaveCompileOptions(
        subs={
            M: 1,
            N: 64,
            BLOCK_M: 1,
            BLOCK_N: 64,
            ADDRESS_SPACE: tkl.AddressSpace.GLOBAL_MEMORY.value,
        },
        canonicalize=True,
        compile_to_mlir=True,
    )
    scanop_cumsum = wave_compile(options, scanop_cumsum)
    print(scanop_cumsum.asm)

    # CHECK-LABEL: func.func @scanop_cumsum

    # Shuffle-based scan: using XOR lane masks
    # CHECK: gpu.shuffle up {{.*}}, {{.*}}, {{.*}} : vector<1xf16>
    # CHECK: affine.apply

    # Conditional mask: comparison with offset
    # CHECK: arith.cmpi sge
    # CHECK: vector.splat {{.*}} : vector<1xi1>
    # CHECK: arith.select {{.*}} : vector<1xi1>, vector<1xf16>

    # Accumulation
    # CHECK: arith.addf {{.*}} : vector<1xf16>

    # Final store
    # CHECK: vector.store {{.*}} : memref<1x64xf16

    # Dispatch
    # CHECK: flow.dispatch @scanop_cumsum


@run_test
def test_atomic_min():
    M = tkl.sym.M
    N = tkl.sym.N
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    wave_size = 64
    BLOCK_M = 2
    BLOCK_N = 256
    num_waves = 2
    shape = (2, 256)

    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=wave_size,
            waves_per_block=(1, num_waves, 1),
            vector_shapes={
                M: int(BLOCK_M / num_waves),
                N: BLOCK_N,
            },
        )
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 1)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 0)]
    constraints += [tkw.WaveConstraint(M, 1)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N)]

    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)
    mapping = tkw.IndexMapping(
        num_iterators=2,
        inputs={M: sympy.Integer(0), N: j},
        outputs={M: i, N: j},
    )
    read_mapping = tkw.IndexMapping(
        num_iterators=2, inputs={M: sympy.Integer(0), N: j}, outputs={M: i, N: j}
    )

    @tkw.wave(constraints)
    def atomic_min_codegen(
        a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.i32],
        c: tkl.Memory[M, N, ADDRESS_SPACE, tkl.i32],
    ):
        res = tkw.read(a)
        shmem = tkw.allocate(
            shape=(M, N),
            distributed_shape=(1, BLOCK_N),
            dtype=tkl.i32,
        )
        inf_reg = tkl.Register[M, N, tkl.i32](1e6)
        tkw.write(inf_reg, shmem)
        res = tkw.atomic_min(res, shmem, mapping=mapping)
        res = tkw.read(shmem, mapping=read_mapping)
        tkw.write(res, c)

    options = WaveCompileOptions(
        subs={
            M: shape[0],
            N: shape[1],
            ADDRESS_SPACE: tkl.AddressSpace.GLOBAL_MEMORY.value,
        },
        canonicalize=True,
        use_buffer_load_ops=False,
        use_buffer_store_ops=False,
        compile_to_mlir=True,
        minimize_shared_allocs=False,
    )

    atomic_min_codegen = wave_compile(options, atomic_min_codegen)
    print(atomic_min_codegen.asm)

    # CHECK-LABEL: test_atomic_min
    # CHECK-DAG:     #[[map0:.*]] = affine_map<()[s0] -> (s0 * 4)>
    # CHECK-DAG:     #[[map1:.*]] = affine_map<()[s0] -> (s0 * 4 + 1)>
    # CHECK-DAG:     #[[map2:.*]] = affine_map<()[s0] -> (s0 * 4 + 2)>
    # CHECK-DAG:     #[[map3:.*]] = affine_map<()[s0] -> (s0 * 4 + 3)>
    # CHECK:         func.func @atomic_min_codegen
    # CHECK-DAG:        %[[C0:.+]] = arith.constant 0 : index
    # CHECK-DAG:        %[[thread_id_x:.*]] = gpu.thread_id  x
    # CHECK-DAG:        %[[thread_id_y:.*]] = gpu.thread_id  y
    # CHECK-DAG:        %[[val_0:.+]] = affine.apply #[[map0]]()[%[[thread_id_x]]]
    # CHECK:            %[[alloc:.*]] = memref.alloc
    # CHECK:            vector.store %{{.*}}, %[[alloc]][%[[thread_id_y]], %[[val_0]]]
    # CHECK:            %[[atm_0:.+]] = memref.atomic_rmw mins %{{.*}}, %[[alloc]][%[[C0]], %[[val_0]]]
    # CHECK-DAG:        %[[val_1:.+]] = affine.apply #[[map1]]()[%[[thread_id_x]]]
    # CHECK:            %[[atm_1:.+]] = memref.atomic_rmw mins %{{.*}}, %[[alloc]][%[[C0]], %[[val_1]]]
    # CHECK-DAG:        %[[val_2:.+]] = affine.apply #[[map2]]()[%[[thread_id_x]]]
    # CHECK:            %[[atm_2:.+]] = memref.atomic_rmw mins %{{.*}}, %[[alloc]][%[[C0]], %[[val_2]]]
    # CHECK-DAG:        %[[val_3:.+]] = affine.apply #[[map3]]()[%[[thread_id_x]]]
    # CHECK:            %[[atm_3:.+]] = memref.atomic_rmw mins %{{.*}}, %[[alloc]][%[[C0]], %[[val_3]]]
    # CHECK:            amdgpu.lds_barrier
    # CHECK:            vector.load %[[alloc]][%[[C0]], %[[val_0]]]
