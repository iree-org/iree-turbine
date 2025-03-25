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
        tkw.read(a, elements_per_thread=16)

    read = wave_compile(get_wave_compile_options(), read)
    print(read.asm)

    # CHECK-LABEL:    func.func @read
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
    # CHECK-DAG:        %[[C16:.+]] = arith.constant 16 : index
    # CHECK:            %[[D1:.+]] = arith.muli %[[WORKGROUP_ID_0]], %[[C16]] overflow<nsw, nuw> : index
    # CHECK-DAG:        %[[C16_0:.+]] = arith.constant 16 : index
    # CHECK-DAG:        %[[C1:.+]] = arith.constant 1 : index
    # CHECK-DAG:        %[[C64:.+]] = arith.constant 64 : index
    # CHECK:            %[[D2:.+]] = arith.divsi %[[THREAD_ID_X]], %[[C64]] : index
    # CHECK:            %[[D3:.+]] = arith.muli %[[D2]], %[[C16_0]] overflow<nsw, nuw> : index
    # CHECK:            %[[D4:.+]] = arith.addi %[[D3]], %[[D1]] overflow<nsw, nuw> : index
    # CHECK:            %[[D4_1:.+]] = arith.addi %[[D4]], %[[THREAD_ID_X]] overflow<nsw, nuw> : index
    # CHECK-DAG:        %[[C16_1:.+]] = arith.constant 16 : index
    # CHECK:            %[[D5:.+]] = arith.muli %[[WORKGROUP_ID_1]], %[[C16_1]] overflow<nsw, nuw> : index
    # CHECK-DAG:        %[[C32:.+]] = arith.constant 32 : index
    # CHECK:            %[[D6:.+]] = arith.muli %[[THREAD_ID_Y]], %[[C32]] overflow<nsw, nuw> : index
    # CHECK:            %[[D7:.+]] = arith.addi %[[D6]], %[[D5]] overflow<nsw, nuw> : index
    # CHECK:            %[[D8:.+]] = vector.load %[[D0]][%[[D4_1]], %[[D7]]] : memref<16x16xf16, strided<[16, 1], offset: ?>>,
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

    # CHECK-LABEL:    func.func @read_mapped
    # CHECK-SAME:       (%[[ARG0:[a-zA-Z0-9_]+]]: !stream.binding)
    # CHECK:            %[[WORKGROUP_ID_0:.+]] = stream.dispatch.workgroup.id[0] : index
    # CHECK:            %[[WORKGROUP_ID_1:.+]] = stream.dispatch.workgroup.id[1] : index
    # CHECK:            %[[WORKGROUP_ID_2:.+]] = stream.dispatch.workgroup.id[2] : index
    # CHECK-DAG:        %[[THREAD_ID_X:.+]] = gpu.thread_id  x
    # CHECK-DAG:        %[[THREAD_ID_Y:.+]] = gpu.thread_id  y
    # CHECK-DAG:        %[[THREAD_ID_Z:.+]] = gpu.thread_id  z
    # CHECK-DAG:        %[[C0:.+]] = arith.constant 0 : index
    # CHECK:            %[[ARR:.+]] = stream.binding.subspan %[[ARG0]][%[[C0]]] : !stream.binding -> memref<16x16xf16,
    # CHECK-SAME:         strided<[16, 1], offset: ?>>
    # CHECK-DAG:        %[[MASK:.+]] = arith.constant dense<true> : vector<16xi1>
    # CHECK-DAG:        %[[C16:.+]] = arith.constant 16 : index
    # CHECK:            %[[D0:.+]] = arith.muli %[[THREAD_ID_X]], %[[C16]] overflow<nsw, nuw> : index
    # CHECK-DAG:        %[[C16_0:.+]] = arith.constant 16 : index
    # CHECK:            %[[D1:.+]] = arith.muli %[[WORKGROUP_ID_0]], %[[C16_0]] overflow<nsw, nuw> : index
    # CHECK-DAG:        %[[C16_1:.+]] = arith.constant 16 : index
    # CHECK-DAG:        %[[C64:.+]] = arith.constant 64 : index
    # CHECK:            %[[D2:.+]] = arith.divsi %[[THREAD_ID_X]], %[[C64]] : index
    # CHECK:            %[[D3:.+]] = arith.muli %[[D2]], %[[C16_1]] overflow<nsw, nuw> : index
    # CHECK:            %[[D4:.+]] = arith.addi %[[D3]], %[[D1]] overflow<nsw, nuw> : index
    # CHECK:            %[[D5:.+]] = arith.addi %[[D4]], %[[D0]] overflow<nsw, nuw> : index
    # CHECK-DAG:        %[[C16_2:.+]] = arith.constant 16 : index
    # CHECK:            %[[D6:.+]] = arith.muli %[[WORKGROUP_ID_1]], %[[C16_2]] overflow<nsw, nuw> : index
    # CHECK-DAG:        %[[C17:.+]] = arith.constant 17 : index
    # CHECK:            %[[D7:.+]] = arith.muli %[[THREAD_ID_Y]], %[[C17]] overflow<nsw, nuw> : index
    # CHECK:            %[[D8:.+]] = arith.addi %[[D7]], %[[D6]] overflow<nsw, nuw> : index
    # CHECK-DAG:        %[[CST:.+]] = arith.constant dense<[0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240]> : vector<16xindex>
    # CHECK-DAG:        %[[CST_2:.+]] = arith.constant 0.000000e+00 : f16
    # CHECK-COUNT-16:   vector.maskedload


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
    # CHECK-COUNT-1:    memref.reinterpret_cast
    # CHECK-COUNT-16:   amdgpu.raw_buffer_load


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
        res = tkw.read(a, elements_per_thread=16)
        tkw.write(res, b, elements_per_thread=16)

    read_write = wave_compile(get_wave_compile_options(canonicalize=True), read_write)
    print(read_write.asm)

    # CHECK-LABEL:    func.func @read_write
    # CHECK-SAME:       (%[[ARG0:[a-zA-Z0-9_]+]]: !stream.binding, %[[ARG1:[a-zA-Z0-9_]+]]: !stream.binding)
    # CHECK-DAG:        %[[C32:.+]] = arith.constant 32 : index
    # CHECK-DAG:        %[[C64:.+]] = arith.constant 64 : index
    # CHECK-DAG:        %[[C16:.+]] = arith.constant 16 : index
    # CHECK-DAG:        %[[C0:.+]] = arith.constant 0 : index
    # CHECK:            %[[WORKGROUP_ID_0:.+]] = stream.dispatch.workgroup.id[0] : index
    # CHECK:            %[[WORKGROUP_ID_1:.+]] = stream.dispatch.workgroup.id[1] : index
    # CHECK-DAG:        %[[THREAD_ID_X:.+]] = gpu.thread_id  x
    # CHECK-DAG:        %[[THREAD_ID_Y:.+]] = gpu.thread_id  y
    # CHECK:            %[[D0:.+]] = stream.binding.subspan %[[ARG0]][%[[C0]]] : !stream.binding -> memref<16x16xf16,
    # CHECK-SAME:         strided<[16, 1], offset: ?>>
    # CHECK:            %[[D1:.+]] = arith.muli %[[WORKGROUP_ID_0]], %[[C16]] overflow<nsw, nuw> : index
    # CHECK:            %[[D2:.+]] = arith.divsi %[[THREAD_ID_X]], %[[C64]] : index
    # CHECK:            %[[D3:.+]] = arith.muli %[[D2]], %[[C16]] overflow<nsw, nuw> : index
    # CHECK:            %[[D4:.+]] = arith.addi %[[D3]], %[[D1]] overflow<nsw, nuw> : index
    # CHECK:            %[[D5:.+]] = arith.addi %[[D4]], %[[THREAD_ID_X]] overflow<nsw, nuw> : index
    # CHECK:            %[[D6:.+]] = arith.muli %[[WORKGROUP_ID_1]], %[[C16]] overflow<nsw, nuw> : index
    # CHECK:            %[[D7:.+]] = arith.muli %[[THREAD_ID_Y]], %[[C32]] overflow<nsw, nuw> : index
    # CHECK:            %[[D8:.+]] = arith.addi %[[D7]], %[[D6]] overflow<nsw, nuw> : index
    # CHECK:            %[[D9:.+]] = vector.load %[[D0]][%[[D5]], %[[D8]]] : memref<16x16xf16, strided<[16, 1], offset: ?>>,
    # CHECK-SAME:         vector<16xf16>
    # CHECK:            %[[D10:.+]] = stream.binding.subspan %[[ARG1]][%[[C0]]] : !stream.binding -> memref<16x16xf16,
    # CHECK-SAME:         strided<[16, 1], offset: ?>>
    # CHECK:            vector.store %[[D9]], %[[D10]][%[[D5]], %[[D8]]] : memref<16x16xf16, strided<[16, 1], offset: ?>>,
    # CHECK-SAME:         vector<16xf16>


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
        tkw.write(res, c, elements_per_thread=16)

    read_write_diagonal = wave_compile(
        get_wave_compile_options(canonicalize=True), read_write_diagonal
    )
    print(read_write_diagonal.asm)

    # CHECK-LABEL:    func.func @read_write_diagonal
    # CHECK-SAME:       (%[[ARG0:[a-zA-Z0-9_]+]]: !stream.binding)
    # CHECK-DAG:        %[[C32:.+]] = arith.constant 32 : index
    # CHECK-DAG:        %[[C64:.+]] = arith.constant 64 : index
    # CHECK-DAG:        %[[C16:.+]] = arith.constant 16 : index
    # CHECK-DAG:        %[[C0:.+]] = arith.constant 0 : index
    # CHECK-DAG:        %[[ONE:.+]] = arith.constant dense<1.000000e+00> : vector<16xf16>
    # CHECK-DAG:        %[[ZERO:.+]] = arith.constant dense<0.000000e+00> : vector<16xf16>
    # CHECK-DAG:        %[[CST:.*]] = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]> : vector<16xindex>
    # CHECK:            %[[WORKGROUP_ID_0:.+]] = stream.dispatch.workgroup.id[0] : index
    # CHECK:            %[[WORKGROUP_ID_1:.+]] = stream.dispatch.workgroup.id[1] : index
    # CHECK-DAG:        %[[THREAD_ID_X:.+]] = gpu.thread_id  x
    # CHECK-DAG:        %[[THREAD_ID_Y:.+]] = gpu.thread_id  y
    # CHECK:            %[[D1:.+]] = arith.muli %[[WORKGROUP_ID_0]], %[[C16]] overflow<nsw, nuw> : index
    # CHECK:            %[[D2:.+]] = arith.divsi %[[THREAD_ID_X]], %[[C64]] : index
    # CHECK:            %[[D3:.+]] = arith.muli %[[D2]], %[[C16]] overflow<nsw, nuw> : index
    # CHECK:            %[[D4:.+]] = arith.addi %[[D3]], %[[D1]] overflow<nsw, nuw> : index
    # CHECK:            %[[BASE_INDEX_X:.+]] = arith.addi %[[D4]], %[[THREAD_ID_X]] overflow<nsw, nuw> : index
    # CHECK:            %[[D5:.*]] = vector.splat %[[BASE_INDEX_X]] : vector<1xindex>
    # CHECK:            %[[D6:.*]] = arith.index_cast %[[D5]] : vector<1xindex> to vector<1xi64>
    # CHECK:            %[[D7:.*]] = vector.extract %[[D6]][0] : i64 from vector<1xi64>
    # CHECK:            %[[D8:.*]] = vector.splat %[[D7]] : vector<16xi64>
    # CHECK:            %[[D9:.*]] = arith.muli %[[WORKGROUP_ID_1]], %[[C16]] overflow<nsw, nuw> : index
    # CHECK:            %[[D10:.*]] = arith.muli %[[THREAD_ID_Y]], %[[C32]] overflow<nsw, nuw> : index
    # CHECK:            %[[BASE_INDEX_Y:.*]] = arith.addi %[[D10]], %[[D9]] overflow<nsw, nuw> : index
    # CHECK:            %[[D12:.*]] = vector.splat %[[BASE_INDEX_Y]] : vector<16xindex>
    # CHECK:            %[[D13:.*]] = arith.addi %[[D12]], %[[CST]] overflow<nsw, nuw> : vector<16xindex>
    # CHECK:            %[[D14:.*]] = arith.index_cast %[[D13]] : vector<16xindex> to vector<16xi64>
    # CHECK:            %[[D15:.*]] = arith.cmpi sge, %[[D8]], %[[D14]] : vector<16xi64>
    # CHECK:            %[[MASK_VAL:.*]] = arith.select %15, %[[ZERO]], %[[ONE]] : vector<16xi1>, vector<16xf16>
    # CHECK:            %[[OUTPUT:.+]] = stream.binding.subspan %{{.*}}[%[[C0]]] : !stream.binding -> memref<16x16xf16, strided<[16, 1], offset: ?>>
    # CHECK:            vector.store %[[MASK_VAL]], %[[OUTPUT]][%[[BASE_INDEX_X]], %[[BASE_INDEX_Y]]] : memref<16x16xf16, strided<[16, 1], offset: ?>>, vector<16xf16>


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
        res = tkw.read(a, elements_per_thread=4)
        tkw.write(res, b, elements_per_thread=4)

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

    # CHECK-LABEL:    func.func @read_write_masked
    # CHECK-SAME:       (%[[ARG0:[a-zA-Z0-9_]+]]: !stream.binding, %[[ARG1:[a-zA-Z0-9_]+]]: !stream.binding)
    # CHECK-DAG:        %[[CST:.*]] = arith.constant dense<0.000000e+00> : vector<4xf16>
    # CHECK-DAG:        %[[CST_0:.*]] = arith.constant dense<3> : vector<4xindex>
    # CHECK-DAG:        %[[CST_1:.*]] = arith.constant dense<[0, 1, 2, 3]> : vector<4xindex>
    # CHECK-DAG:        %[[C0:.*]] = arith.constant 0 : index
    # CHECK-DAG:        %[[C1:.*]] = arith.constant 1 : index
    # CHECK-DAG:        %[[C4:.*]] = arith.constant 4 : index
    # CHECK-DAG:        %[[C8:.*]] = arith.constant 8 : index
    # CHECK-DAG:        %[[C64:.*]] = arith.constant 64 : index
    # CHECK:            %[[WORKGROUP_ID_0:.*]] = stream.dispatch.workgroup.id[0] : index
    # CHECK:            %[[WORKGROUP_ID_1:.*]] = stream.dispatch.workgroup.id[1] : index
    # CHECK-DAG:        %[[THREAD_ID_X:.*]] = gpu.thread_id  x
    # CHECK-DAG:        %[[THREAD_ID_Y:.*]] = gpu.thread_id  y
    # CHECK:            %[[D0:.*]] = stream.binding.subspan %[[ARG0]][%[[C0]]] : !stream.binding -> memref<1x3xf16,
    # CHECK-SAME:         strided<[3, 1], offset: ?>>
    # CHECK:            %[[D1:.*]] = arith.muli %[[WORKGROUP_ID_0]], %[[C4]] overflow<nsw, nuw> : index
    # CHECK:            %[[D2:.*]] = arith.divsi %[[THREAD_ID_X]], %[[C64]] : index
    # CHECK:            %[[D3:.*]] = arith.muli %[[D2]], %[[C4]] overflow<nsw, nuw> : index
    # CHECK:            %[[D4:.*]] = arith.addi %[[D3]], %[[D1]] overflow<nsw, nuw> : index
    # CHECK:            %[[D5:.*]] = arith.addi %[[D4]], %[[THREAD_ID_X]] overflow<nsw, nuw> : index
    # CHECK:            %[[D6:.*]] = arith.muli %[[WORKGROUP_ID_1]], %[[C4]] overflow<nsw, nuw> : index
    # CHECK:            %[[D7:.*]] = arith.muli %[[THREAD_ID_Y]], %[[C8]] overflow<nsw, nuw> : index
    # CHECK:            %[[D8:.*]] = arith.addi %[[D7]], %[[D6]] overflow<nsw, nuw> : index
    # CHECK:            %[[D9:.*]] = vector.splat %[[D8]] : vector<4xindex>
    # CHECK:            %[[D10:.*]] = arith.addi %[[D9]], %[[CST_1]] overflow<nsw, nuw> : vector<4xindex>
    # CHECK:            %[[D11:.*]] = arith.cmpi slt, %[[D10]], %[[CST_0]] : vector<4xindex>
    # CHECK:            %[[D12:.*]] = arith.cmpi slt, %[[D5]], %[[C1]] : index
    # CHECK:            %[[D13:.*]] = vector.splat %[[D12]] : vector<4xi1>
    # CHECK:            %[[D14:.*]] = arith.andi %[[D11]], %[[D13]] : vector<4xi1>
    # CHECK:            %[[D15:.*]] = vector.maskedload %[[D0]][%[[D5]], %[[D8]]], %[[D14]], %[[CST]] : memref<1x3xf16,
    # CHECK-SAME:         strided<[3, 1], offset: ?>>, vector<4xi1>, vector<4xf16> into vector<4xf16>
    # CHECK:            %[[D16:.*]] = stream.binding.subspan %arg1[%[[C0]]] : !stream.binding -> memref<1x3xf16,
    # CHECK-SAME:         strided<[3, 1], offset: ?>>
    # CHECK:            vector.maskedstore %[[D16]][%[[D5]], %[[D8]]], %[[D14]], %[[D15]] : memref<1x3xf16,
    # CHECK-SAME:         strided<[3, 1], offset: ?>>, vector<4xi1>, vector<4xf16>


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
        res = tkw.read(a, elements_per_thread=4)
        tkw.write(res, b, elements_per_thread=4)

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
        res = tkw.read(a, elements_per_thread=16)
        tkw.write(res, b, mapping=mapping, elements_per_thread=16)

    read_write_mapping = wave_compile(
        get_wave_compile_options(canonicalize=True), read_write_mapping
    )
    print(read_write_mapping.asm)

    # CHECK-LABEL:    func.func @read_write_mapping
    # CHECK-SAME:       (%[[ARG0:[a-zA-Z0-9_]+]]: !stream.binding, %[[ARG1:[a-zA-Z0-9_]+]]: !stream.binding)
    # CHECK-DAG:        %[[C32:.+]] = arith.constant 32 : index
    # CHECK-DAG:        %[[C64:.+]] = arith.constant 64 : index
    # CHECK-DAG:        %[[C16:.+]] = arith.constant 16 : index
    # CHECK-DAG:        %[[C0:.+]] = arith.constant 0 : index
    # CHECK:            %[[WORKGROUP_ID_0:.+]] = stream.dispatch.workgroup.id[0] : index
    # CHECK:            %[[WORKGROUP_ID_1:.+]] = stream.dispatch.workgroup.id[1] : index
    # CHECK-DAG:        %[[THREAD_ID_X:.+]] = gpu.thread_id  x
    # CHECK-DAG:        %[[THREAD_ID_Y:.+]] = gpu.thread_id  y
    # CHECK:            %[[D0:.+]] = stream.binding.subspan %[[ARG0]][%[[C0]]] : !stream.binding -> memref<16x16xf16,
    # CHECK-SAME:         strided<[16, 1], offset: ?>>
    # CHECK:            %[[D1:.+]] = arith.muli %[[WORKGROUP_ID_0]], %[[C16]] overflow<nsw, nuw> : index
    # CHECK:            %[[D2:.+]] = arith.divsi %[[THREAD_ID_X]], %[[C64]] : index
    # CHECK:            %[[D3:.+]] = arith.muli %[[D2]], %[[C16]] overflow<nsw, nuw> : index
    # CHECK:            %[[D4:.+]] = arith.addi %[[D3]], %[[D1]] overflow<nsw, nuw> : index
    # CHECK:            %[[D5:.+]] = arith.addi %[[D4]], %[[THREAD_ID_X]] overflow<nsw, nuw> : index
    # CHECK:            %[[D6:.+]] = arith.muli %[[WORKGROUP_ID_1]], %[[C16]] overflow<nsw, nuw> : index
    # CHECK:            %[[D7:.+]] = arith.muli %[[THREAD_ID_Y]], %[[C32]] overflow<nsw, nuw> : index
    # CHECK:            %[[D8:.+]] = arith.addi %[[D7]], %[[D6]] overflow<nsw, nuw> : index
    # CHECK:            %[[D9:.+]] = vector.load %[[D0]][%[[D5]], %[[D8]]] : memref<16x16xf16, strided<[16, 1], offset: ?>>,
    # CHECK-SAME:         vector<16xf16>
    # CHECK:            %[[D10:.+]] = stream.binding.subspan %[[ARG1]][%[[C0]]] : !stream.binding -> memref<16x16xf16,
    # CHECK-SAME:         strided<[16, 1], offset: ?>>
    # CHECK-COUNT-16:   vector.store


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
        offset = tkw.read(off, elements_per_thread=16)
        res = tkw.read(
            a,
            mapping=mapping,
            mapping_dynamic_vals=(offset,),
            elements_per_thread=16,
        )
        tkw.write(res, b, elements_per_thread=16)

    read_write_dynamic_mapping = wave_compile(
        get_wave_compile_options(canonicalize=True), read_write_dynamic_mapping
    )
    print(read_write_dynamic_mapping.asm)

    # CHECK-LABEL:    func.func @read_write_dynamic_mapping
    # CHECK-SAME:       (%[[ARG0:.*]]: !stream.binding, %[[ARG1:.*]]: !stream.binding, %[[ARG2:.*]]: !stream.binding)
    # CHECK-DAG:        %[[D0:.*]] = arith.constant 0 : index
    # CHECK-DAG:        %[[C16:.*]] = arith.constant 16 : index
    # CHECK-DAG:        %[[C32:.*]] = arith.constant 32 : index
    # CHECK-DAG:        %[[C64:.*]] = arith.constant 64 : index
    # CHECK-DAG:        %[[C256:.*]] = arith.constant 256 : index
    # CHECK:            %[[D0:.*]] = stream.binding.subspan %[[ARG1]][%[[C0]]] : !stream.binding -> memref<16x16xi32, strided<[16, 1], offset: ?>>
    # CHECK:            %[[D9:.*]] = vector.load %[[D0]][%[[D5:.*]], %[[D8:.*]]] : memref<16x16xi32, strided<[16, 1], offset: ?>>, vector<16xi32>
    # CHECK:            %[[D10:.*]] = stream.binding.subspan %[[ARG0]][%[[C0]]] : !stream.binding -> memref<16x16xf16, strided<[16, 1], offset: ?>>
    # CHECK:            %[[D11:.*]] = arith.index_cast %[[D9]] : vector<16xi32> to vector<16xindex>
    # CHECK:            %[[D13:.*]] = arith.muli %{{.*}}, %[[C16]] overflow<nsw, nuw> : index
    # CHECK:            %[[D14:.*]] = arith.muli %{{.*}}, %[[C256]] overflow<nsw, nuw> : index
    # CHECK:            %[[D15:.*]] = arith.muli %{{.*}}, %[[C256]] overflow<nsw, nuw> : index
    # CHECK:            %[[D16:.*]] = arith.addi %[[D15]], %[[D14]] overflow<nsw, nuw> : index
    # CHECK:            %[[D17:.*]] = arith.addi %[[D16]], %[[D13]] overflow<nsw, nuw> : index
    # CHECK:            %[[D18:.*]] = vector.splat %[[D17]] : vector<16xindex>
    # CHECK:            %[[D19:.*]] = arith.addi %[[D18]], %[[D11]] overflow<nsw, nuw> : vector<16xindex>
    # CHECK-COUNT-16:   vector.load
    # CHECK:            %[[D21:.*]] = stream.binding.subspan %[[ARG2]][%[[C0]]] : !stream.binding -> memref<16x16xf16, strided<[16, 1], offset: ?>>
    # CHECK:            vector.store %{{.*}}, %[[D21]][%[[D5]], %[[D8]]] : memref<16x16xf16, strided<[16, 1], offset: ?>>, vector<16xf16>


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
        offset = tkw.read(off, elements_per_thread=1)
        res = tkw.read(
            a,
            mapping=mapping,
            mapping_dynamic_vals=(offset,),
            elements_per_thread=16,
        )
        tkw.write(res, b, elements_per_thread=16)

    read_write_dynamic_mapping_broadcast = wave_compile(
        get_wave_compile_options(canonicalize=True, additional_symbols={ONE: 1}),
        read_write_dynamic_mapping_broadcast,
    )
    print(read_write_dynamic_mapping_broadcast.asm)

    # CHECK-LABEL:    func.func @read_write_dynamic_mapping_broadcast
    # CHECK:            %[[OFF:.*]] = vector.load %{{.*}}[%[[M:.*]], %{{.*}}] : memref<16x1xi32, strided<[1, 1], offset: ?>>, vector<1xi32>
    # CHECK:            %[[IDX:.*]] = arith.index_cast %[[OFF]] : vector<1xi32> to vector<1xindex>
    # CHECK:            %[[IDX1:.*]] = vector.extract %[[IDX]][0] : index from vector<1xindex>
    # CHECK:            %[[RES:.*]] = vector.load %{{.*}}[%[[M]], %[[IDX1]]] : memref<16x16xf16, strided<[16, 1], offset: ?>>, vector<16xf16>
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
        offset1 = tkw.read(off1, elements_per_thread=1)
        offset2 = tkw.read(
            off2,
            mapping=mapping1,
            mapping_dynamic_vals=(offset1,),
            elements_per_thread=1,
        )
        res = tkw.read(
            a,
            mapping=mapping2,
            mapping_dynamic_vals=(offset2,),
            elements_per_thread=4,
        )
        tkw.write(res, b, elements_per_thread=4)

    read_write_dynamic_mapping_chain = wave_compile(
        get_wave_compile_options(
            canonicalize=True, additional_symbols={BLOCK_N: 4, SIZE1: 2, SIZE2: 4}
        ),
        read_write_dynamic_mapping_chain,
    )
    print(read_write_dynamic_mapping_chain.asm)

    # CHECK-LABEL:    func.func @read_write_dynamic_mapping_chain
    # CHECK:            %[[C8:.*]] = arith.constant 8 : index
    # CHECK:            %[[thread_id_y:.*]] = gpu.thread_id  y
    # CHECK:            %[[D7:.*]] = arith.addi %{{.*}}, %[[thread_id_y]] overflow<nsw, nuw> : index
    # CHECK:            %[[D8:.*]] = vector.load %{{.*}}[%[[D5:.*]], %[[D7]]] : memref<16x2xi32, strided<[2, 1], offset: ?>>, vector<1xi32>
    # CHECK:            %[[D10:.*]] = arith.index_cast %[[D8]] : vector<1xi32> to vector<1xindex>
    # CHECK:            %[[D11:.*]] = vector.extract %[[D10]][0] : index from vector<1xindex>
    # CHECK:            %[[D12:.*]] = vector.load %{{.*}}[%[[D5]], %[[D11]]] : memref<16x4xi32, strided<[4, 1], offset: ?>>, vector<1xi32>
    # CHECK:            %[[D14:.*]] = arith.index_cast %[[D12]] : vector<1xi32> to vector<1xindex>
    # CHECK:            %[[D15:.*]] = vector.extract %[[D14]][0] : index from vector<1xindex>
    # CHECK:            %[[D16:.*]] = vector.load %{{.*}}[%[[D5]], %[[D15]]] : memref<16x16xf16, strided<[16, 1], offset: ?>>, vector<4xf16>
    # CHECK:            %[[D19:.*]] = arith.muli %[[thread_id_y]], %[[C8]] overflow<nsw, nuw> : index
    # CHECK:            %[[D20:.*]] = arith.addi %[[D19]], %{{.*}} overflow<nsw, nuw> : index
    # CHECK:            vector.store %[[D16]], %{{.*}}[%[[D5]], %[[D20]]] : memref<16x16xf16, strided<[16, 1], offset: ?>>, vector<4xf16>


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
        offset = tkw.read(off, elements_per_thread=1)
        tkw.set_symbol(S, offset)
        res = tkw.read(
            a,
            mapping=mapping,
            elements_per_thread=1,
        )
        tkw.write(res, b, elements_per_thread=1)

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
    #       CHECK:      %[[O1:.*]] = vector.load %[[A2]][%[[M:.*]], %[[N:.*]]] : memref<16x16xi32, strided<[16, 1], offset: ?>>, vector<1xi32>
    #       CHECK:      %[[O2:.*]] = arith.index_cast %[[O1]] : vector<1xi32> to vector<1xindex>
    #       CHECK:      %[[O3:.*]] = vector.extract %[[O2]][0] : index from vector<1xindex>
    #       CHECK:      %[[A1:.*]] = stream.binding.subspan %[[ARG0]][%[[C0]]] : !stream.binding -> memref<?x16xf16, strided<[16, 1], offset: ?>>{%arg3}
    #       CHECK:      %[[RES:.*]] = vector.load %[[A1]][%[[O3]], %[[N]]] : memref<?x16xf16, strided<[16, 1], offset: ?>>, vector<1xf16>
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
        offset = tkw.read(off, elements_per_thread=1)
        offset = tkw.apply_expr(offset, lambda a: M - a - 1)
        tkw.set_symbol(S, offset)
        res = tkw.read(
            a,
            mapping=mapping,
            elements_per_thread=1,
        )
        tkw.write(res, b, elements_per_thread=1)

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
    #   CHECK-DAG:      %[[CST:.*]] = arith.constant dense<15> : vector<1xindex>
    #   CHECK-DAG:      %[[C0:.*]] = arith.constant 0 : index
    #       CHECK:      %[[A2:.*]] = stream.binding.subspan %[[ARG1]][%[[C0]]] : !stream.binding -> memref<16x16xi32, strided<[16, 1], offset: ?>>
    #       CHECK:      %[[O1:.*]] = vector.load %[[A2]][%[[M:.*]], %[[N:.*]]] : memref<16x16xi32, strided<[16, 1], offset: ?>>, vector<1xi32>
    #       CHECK:      %[[O2:.*]] = arith.index_cast %[[O1]] : vector<1xi32> to vector<1xindex>
    #       CHECK:      %[[O3:.*]] = arith.subi %[[CST]], %[[O2]] : vector<1xindex>
    #       CHECK:      %[[O4:.*]] = vector.extract %[[O3]][0] : index from vector<1xindex>
    #       CHECK:      %[[A1:.*]] = stream.binding.subspan %[[ARG0]][%[[C0]]] : !stream.binding -> memref<?x16xf16, strided<[16, 1], offset: ?>>{%arg3}
    #       CHECK:      %[[RES:.*]] = vector.load %[[A1]][%[[O4]], %[[N]]] : memref<?x16xf16, strided<[16, 1], offset: ?>>, vector<1xf16>
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
    #   CHECK-DAG:      %[[CST:.*]] = arith.constant dense<0> : vector<1xindex>
    #   CHECK-DAG:      %[[C0:.*]] = arith.constant 0 : index
    #       CHECK:      %[[A1:.*]] = stream.binding.subspan %[[ARG0]][%[[C0]]] : !stream.binding -> memref<16x16xf16, strided<[16, 1], offset: ?>>
    #       CHECK:      %[[RES:.*]] = vector.load %[[A1]][%[[M:.*]], %[[N:.*]]] : memref<16x16xf16, strided<[16, 1], offset: ?>>, vector<1xf16>
    #       CHECK:      %[[A2:.*]] = stream.binding.subspan %[[ARG1]][%[[C0]]] : !stream.binding -> memref<16x16xi32, strided<[16, 1], offset: ?>>
    #       CHECK:      %[[O1:.*]] = vector.load %[[A2]][%[[M]], %[[N]]] : memref<16x16xi32, strided<[16, 1], offset: ?>>, vector<1xi32>
    #       CHECK:      %[[O2:.*]] = arith.index_cast %[[O1]] : vector<1xi32> to vector<1xindex>
    #       CHECK:      %[[O3:.*]] = arith.cmpi sgt, %[[O2]], %[[CST]] : vector<1xindex>
    #       CHECK:      %[[O4:.*]] = vector.extract %[[O3]][0] : i1 from vector<1xi1>
    #       CHECK:      scf.if %[[O4]] {
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

    # CHECK-LABEL:    func.func @dynamic_copy
    # CHECH-SAME:       %[[ARG0:.*]]: !stream.binding, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index
    # CHECK-SAME:       attributes {translation_info = #[[TRANSLATION:.+]]} {
    # CHECK-DAG:        %[[CST:.+]] = arith.constant dense<0.000000e+00> : vector<16xf16>
    # CHECK-DAG:        %[[CST_0:.+]] = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]> :
    # CHECK-SAME:         vector<16xindex>
    # CHECK-DAG:        %[[C32:.+]] = arith.constant 32 : index
    # CHECK-DAG:        %[[C64:.+]] = arith.constant 64 : index
    # CHECK-DAG:        %[[C16]] = arith.constant 16 : index
    # CHECK-DAG:        %[[C0:.+]] = arith.constant 0 : index
    # CHECK:            %[[WORKGROUP_ID_0:.+]] = stream.dispatch.workgroup.id[0] : index
    # CHECK:            %[[WORKGROUP_ID_1:.+]] = stream.dispatch.workgroup.id[1] : index
    # CHECK-DAG:        %[[THREAD_ID_X:.+]] = gpu.thread_id  x
    # CHECK-DAG:        %[[THREAD_ID_Y:.+]] = gpu.thread_id  y
    # CHECK:            %[[D0]] = stream.binding.subspan %[[ARG0]][%[[C0]]] : !stream.binding -> memref<?x?xf16, strided<[?, 1], offset: ?>>{%[[ARG1]],
    # CHECK-SAME:         %[[ARG2]]}
    # CHECK:            %[[D1]] = arith.muli %[[WORKGROUP_ID_0]], %[[C16]] overflow<nsw, nuw> : index
    # CHECK:            %[[D2:.+]] = arith.divsi %[[THREAD_ID_X]], %[[C64]] : index
    # CHECK:            %[[D3:.+]] = arith.muli %[[D2]], %[[C16]] overflow<nsw, nuw> : index
    # CHECK:            %[[D4:.+]] = arith.addi %[[D3]], %[[D1]] overflow<nsw, nuw> : index
    # CHECK:            %[[D5:.+]] = arith.addi %[[D4]], %[[THREAD_ID_X]] overflow<nsw, nuw> : index
    # CHECK:            %[[D6:.+]] = arith.muli %[[WORKGROUP_ID_1]], %[[C16]] overflow<nsw, nuw> : index
    # CHECK:            %[[D7:.+]] = arith.muli %[[THREAD_ID_Y]], %[[C32]] overflow<nsw, nuw> : index
    # CHECK:            %[[D8:.+]] = arith.addi %[[D7]], %[[D6]] overflow<nsw, nuw> : index
    # CHECK:            %[[D9:.+]] = vector.splat %[[D8]] : vector<16xindex>
    # CHECK:            %[[D10:.+]] = arith.addi %[[D9]], %[[CST_0]] overflow<nsw, nuw> : vector<16xindex>
    # CHECK:            %[[D11:.+]] = vector.splat %[[ARG2]] : vector<16xindex>
    # CHECK:            %[[D12:.+]] = arith.cmpi slt, %[[D10]], %[[D11]] : vector<16xindex>
    # CHECK:            %[[D13:.+]] = arith.cmpi slt, %[[D5]], %[[ARG1]] : index
    # CHECK:            %[[D14:.+]] = vector.splat %[[D13]] : vector<16xi1>
    # CHECK:            %[[D15:.+]] = arith.andi %[[D12]], %[[D14]] : vector<16xi1>
    # CHECK:            %[[D16:.+]] = vector.maskedload %[[D0]][%[[D5]], %[[D8]]], %[[D15]], %[[CST]] : memref<?x?xf16, strided<[?, 1], offset: ?>>,
    # CHECK-SAME:         vector<16xi1>, vector<16xf16> into vector<16xf16>
    # CHECK:            vector.maskedstore %[[D0]][%[[D5]], %[[D8]]], %[[D15]], %[[D16]] : memref<?x?xf16, strided<[?, 1], offset: ?>>, vector<16xi1>,
    # CHECK-SAME:         vector<16xf16>
    # CHECK:            return


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
        a_reg = tkw.read(a, elements_per_thread=16)
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
        a_reg = tkw.read(a, elements_per_thread=16)
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
        res = tkw.roundeven(res)
        tkw.write(res, a, elements_per_thread=4)
        tkw.write(res_b, b, elements_per_thread=4)

    test = wave_compile(get_wave_compile_options(), test)
    print(test.asm)
    # CHECK-LABEL: func @test
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

    # Tests roundeven
    # CHECK: %[[ROUNDEVEN:.+]] = math.roundeven %[[TANH]]


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

    # CHECK-LABEL: func @test
    # CHECK-DAG: %[[C1:.+]] = arith.constant 1 : i32
    # CHECK-DAG: %[[C2:.+]] = arith.constant 2 : i32
    # CHECK-DAG: %[[C4:.+]] = arith.constant 4 : i32
    # CHECK-DAG: %[[C8:.+]] = arith.constant 8 : i32
    # CHECK-DAG: %[[C16:.+]] = arith.constant 16 : i32
    # CHECK-DAG: %[[C32:.+]] = arith.constant 32 : i32
    # Elementwise
    # CHECK: arith.mulf {{.*}} : vector<2xf16>
    # Local Reduction
    # CHECK: arith.addf {{.*}} : vector<1xf16>
    # Global Reduction
    # CHECK: gpu.shuffle  xor %{{.+}}, %[[C1]], %{{.+}} : f32
    # CHECK: arith.addf {{.*}} : vector<1xf16>
    # CHECK: gpu.shuffle  xor %{{.+}}, %[[C2]], %{{.+}} : f32
    # CHECK: arith.addf {{.*}} : vector<1xf16>
    # CHECK: gpu.shuffle  xor %{{.+}}, %[[C4]], %{{.+}} : f32
    # CHECK: arith.addf {{.*}} : vector<1xf16>
    # CHECK: gpu.shuffle  xor %{{.+}}, %[[C8]], %{{.+}} : f32
    # CHECK: arith.addf {{.*}} : vector<1xf16>
    # CHECK: gpu.shuffle  xor %{{.+}}, %[[C16]], %{{.+}} : f32
    # CHECK: arith.addf {{.*}} : vector<1xf16>
    # CHECK: gpu.shuffle  xor %{{.+}}, %[[C32]], %{{.+}} : f32
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

    # CHECK-LABEL: func @test
    # CHECK: %[[LHS:.+]] = vector.load {{.*}} : memref<256x128xf16
    # CHECK: %[[RHS:.+]] = vector.load {{.*}} : memref<256x128xf16
    # Reduce all sources locally.
    # CHECK: %[[SRC_REDUC:.+]] = arith.addf %[[LHS]], %[[RHS]] : vector<2xf16>
    # Do Local Reductions.
    # CHECK: %[[LOCAL_REDUC0:.+]] = vector.extract_strided_slice %[[SRC_REDUC]] {offsets = [0], sizes = [1], strides = [1]}
    # CHECK: %[[LOCAL_REDUC1:.+]] = vector.extract_strided_slice %[[SRC_REDUC]] {offsets = [1], sizes = [1], strides = [1]}
    # CHECK: %[[REDUC_0:.+]] = arith.addf %[[LOCAL_REDUC0]], %[[LOCAL_REDUC1]] : vector<1xf16>
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

        @tkw.reduction(N, init_args=[init_max])
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

    # CHECK-LABEL: func @test
    # CHECK-DAG: %[[C0_IDX:.+]] = arith.constant 0 : index
    # CHECK-DAG: %[[C4_IDX:.+]] = arith.constant 4 : index
    # CHECK-DAG: %[[C1_IDX:.+]] = arith.constant 1 : index
    # CHECK-DAG: %[[INIT:.+]] = arith.constant dense<0xFC00> : vector<1xf16>

    # Tile Reduction Loop
    # CHECK: %[[TILED:.+]]:2 = scf.for %[[ITER:.+]] = %[[C0_IDX]] to %[[C4_IDX]] step %[[C1_IDX]]
    # CHECK-SAME: iter_args(%[[ACC0:.+]] = %[[INIT]], %[[ACC1:.+]] = %[[INIT]]) -> (vector<1xf16>, vector<1xf16>) {
    # 1st Expanded Local Reduction
    # CHECK: arith.maximumf {{.*}} : vector<1xf16>
    # 1st Expanded Global Reduction
    # CHECK-COUNT-6: gpu.shuffle  xor
    # 1st Expanded Accumulator Reduction
    # CHECK: %[[ACC_REDUCE_0:.+]] = arith.maximumf %[[ACC0]], %{{.*}}

    # 2nd Expanded Local Reduction
    # CHECK: arith.maximumf {{.*}} : vector<1xf16>
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

        @tkw.reduction(N, init_args=[init_max])
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

    # CHECK-LABEL: func @test
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
    # CHECK: arith.maximumf {{.*}} : vector<1xf16>
    # Global Reduction
    # CHECK: gpu.shuffle  xor %{{.+}}, %[[C1]], %{{.+}} : f32
    # CHECK: arith.maximumf {{.*}} : vector<1xf16>
    # CHECK: gpu.shuffle  xor %{{.+}}, %[[C2]], %{{.+}} : f32
    # CHECK: arith.maximumf {{.*}} : vector<1xf16>
    # CHECK: gpu.shuffle  xor %{{.+}}, %[[C4]], %{{.+}} : f32
    # CHECK: arith.maximumf {{.*}} : vector<1xf16>
    # CHECK: gpu.shuffle  xor %{{.+}}, %[[C8]], %{{.+}} : f32
    # CHECK: arith.maximumf {{.*}} : vector<1xf16>
    # CHECK: gpu.shuffle  xor %{{.+}}, %[[C16]], %{{.+}} : f32
    # CHECK: arith.maximumf {{.*}} : vector<1xf16>
    # CHECK: gpu.shuffle  xor %{{.+}}, %[[C32]], %{{.+}} : f32
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
        init_min = tkl.Register[M, tkl.f16](1e6)

        @tkw.reduction(N, init_args=[init_min])
        def repeat(
            partial_min: tkl.Register[M, tkl.f16],
        ) -> tkl.Register[M, tkl.f16]:
            lhs = tkw.read(a, elements_per_thread=ELEMS_PER_THREAD)
            rhs = tkw.read(b, elements_per_thread=ELEMS_PER_THREAD)
            res = lhs * rhs
            partial_min = tkw.min(res, partial_min, dim=N)
            return partial_min

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

    # CHECK-LABEL: func @test
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
    # CHECK: arith.minimumf {{.*}} : vector<1xf16>
    # Global Reduction
    # CHECK: gpu.shuffle  xor %{{.+}}, %[[C1]], %{{.+}} : f32
    # CHECK: arith.minimumf {{.*}} : vector<1xf16>
    # CHECK: gpu.shuffle  xor %{{.+}}, %[[C2]], %{{.+}} : f32
    # CHECK: arith.minimumf {{.*}} : vector<1xf16>
    # CHECK: gpu.shuffle  xor %{{.+}}, %[[C4]], %{{.+}} : f32
    # CHECK: arith.minimumf {{.*}} : vector<1xf16>
    # CHECK: gpu.shuffle  xor %{{.+}}, %[[C8]], %{{.+}} : f32
    # CHECK: arith.minimumf {{.*}} : vector<1xf16>
    # CHECK: gpu.shuffle  xor %{{.+}}, %[[C16]], %{{.+}} : f32
    # CHECK: arith.minimumf {{.*}} : vector<1xf16>
    # CHECK: gpu.shuffle  xor %{{.+}}, %[[C32]], %{{.+}} : f32
    # CHECK: %[[GLOBAL_REDUCE:.+]] = arith.minimumf {{.*}} : vector<1xf16>
    # Accumulator Reduction
    # CHECK: %[[ACC_REDUCE:.+]] = arith.minimumf %[[ACC]], %[[GLOBAL_REDUCE]]
    # CHECK: scf.yield %[[ACC_REDUCE]] : vector<1xf16>


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

        @tkw.reduction(N, init_args=[init_max, init_sum])
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

    # CHECK-LABEL: func @test
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
    # CHECK: arith.maximumf {{.*}} : vector<1xf16>
    # 1st Expanded Global Max Reduction
    # CHECK-COUNT-6: gpu.shuffle  xor
    # 1st Expanded Accumulator Max Reduction
    # CHECK: %[[ACC_MAX_0:.+]] = arith.maximumf %[[ACC0]], %{{.*}}

    # 2nd Expanded Local Max Reduction
    # CHECK: arith.maximumf {{.*}} : vector<1xf16>
    # 2nd Expanded Global Max Reduction
    # CHECK-COUNT-6: gpu.shuffle  xor
    # 2nd Expanded Accumulator Max Reduction
    # CHECK: %[[ACC_MAX_1:.+]] = arith.maximumf %[[ACC1]], %{{.*}}

    # 1st Expanded Local Sum Reduction
    # CHECK: arith.addf {{.*}} : vector<1xf16>
    # 1st Expanded Global Sum Reduction
    # CHECK-COUNT-6: gpu.shuffle  xor
    # 1st Expanded Accumulator Sum Reduction
    # CHECK: %[[ACC_SUM_0:.+]] = arith.addf %[[ACC2]], %{{.*}}

    # 2nd Expanded Local Sum Reduction
    # CHECK: arith.addf {{.*}} : vector<1xf16>
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

        @tkw.reduction(N, init_args=[init_max, init_sum])
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

    # CHECK-LABEL: func @test
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
    # CHECK: %[[EXP2_SLICE_0:.+]] = vector.extract_strided_slice %[[EXP2]] {offsets = [0], sizes = [1], strides = [1]} : vector<2xf32> to vector<1xf32>
    # CHECK: %[[EXP2_SLICE_1:.+]] = vector.extract_strided_slice %[[EXP2]] {offsets = [1], sizes = [1], strides = [1]} : vector<2xf32> to vector<1xf32>
    # CHECK: arith.addf %[[EXP2_SLICE_0]], %[[EXP2_SLICE_1]]
    # CHECK-COUNT-6: gpu.shuffle xor


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

    # CHECK-LABEL: func.func @explicit_broadcast
    # CHECK-SAME: (%[[ARG0:.+]]: !stream.binding, %[[ARG1:.+]]: !stream.binding, %{{.+}}: !stream.binding)
    # CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
    # CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index

    # Slicing LHS
    # CHECK: %[[LHS:.+]] = stream.binding.subspan %[[ARG0]][%[[C0]]] : !stream.binding -> memref<256x128xf16
    # CHECK: %[[LHS_0:.+]] = vector.load %[[LHS]][%[[X_SLICE_0:.+]], %[[Y_SLICE:.+]]] : memref<256x128xf16, strided<[128, 1], offset: ?>>, vector<2xf16>
    # CHECK: %[[X_SLICE_1:.+]] = arith.addi %[[X_SLICE_0]], %c1 overflow<nsw, nuw> : index
    # CHECK: %[[LHS_1:.+]] = vector.load %[[LHS]][%[[X_SLICE_1]], %[[Y_SLICE]]] : memref<256x128xf16, strided<[128, 1], offset: ?>>, vector<2xf16>

    # Slicing RHS
    # CHECK: %[[RHS:.+]] = stream.binding.subspan %[[ARG1]][%[[C0]]] : !stream.binding -> memref<256xf16
    # CHECK: %[[RHS_0:.+]] = vector.load %[[RHS]][%[[X_SLICE_0]]] : memref<256xf16, strided<[1], offset: ?>>, vector<1xf16>
    # CHECK: %[[RHS_1:.+]] = vector.load %[[RHS]][%[[X_SLICE_1]]] : memref<256xf16, strided<[1], offset: ?>>, vector<1xf16>

    # 1st Broadcast RHS
    # CHECK: %[[EXTRACT_0:.+]] = vector.extract %[[RHS_0]][0] : f16 from vector<1xf16>
    # CHECK: %[[BCAST_RHS_0:.+]] = vector.splat %[[EXTRACT_0]] : vector<2xf16>

    # 2nd Broadcast RHS
    # CHECK: %[[EXTRACT_1:.+]] = vector.extract %[[RHS_1]][0] : f16 from vector<1xf16>
    # CHECK: %[[BCAST_RHS_1:.+]] = vector.splat %[[EXTRACT_1]] : vector<2xf16>

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

    # CHECK-LABEL: func.func @broadcast_add
    # CHECK-SAME: (%[[ARG0:.+]]: !stream.binding, %[[ARG1:.+]]: !stream.binding, %{{.+}}: !stream.binding)
    # CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
    # CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index

    # Slicing LHS
    # CHECK: %[[LHS:.+]] = stream.binding.subspan %[[ARG0]][%[[C0]]] : !stream.binding -> memref<256x128xf16
    # CHECK: %[[LHS_0:.+]] = vector.load %[[LHS]][%[[X_SLICE_0:.+]], %[[Y_SLICE:.+]]] : memref<256x128xf16, strided<[128, 1], offset: ?>>, vector<2xf16>
    # CHECK: %[[X_SLICE_1:.+]] = arith.addi %[[X_SLICE_0]], %c1 overflow<nsw, nuw> : index
    # CHECK: %[[LHS_1:.+]] = vector.load %[[LHS]][%[[X_SLICE_1]], %[[Y_SLICE]]] : memref<256x128xf16, strided<[128, 1], offset: ?>>, vector<2xf16>

    # Slicing RHS
    # CHECK: %[[RHS:.+]] = stream.binding.subspan %[[ARG1]][%[[C0]]] : !stream.binding -> memref<256xf16
    # CHECK: %[[RHS_0:.+]] = vector.load %[[RHS]][%[[X_SLICE_0]]] : memref<256xf16, strided<[1], offset: ?>>, vector<1xf16>
    # CHECK: %[[RHS_1:.+]] = vector.load %[[RHS]][%[[X_SLICE_1]]] : memref<256xf16, strided<[1], offset: ?>>, vector<1xf16>

    # 1st Broadcast RHS
    # CHECK: %[[EXTRACT_0:.+]] = vector.extract %[[RHS_0]][0] : f16 from vector<1xf16>
    # CHECK: %[[BCAST_RHS_0:.+]] = vector.splat %[[EXTRACT_0]] : vector<2xf16>

    # 2nd Broadcast RHS
    # CHECK: %[[EXTRACT_1:.+]] = vector.extract %[[RHS_1]][0] : f16 from vector<1xf16>
    # CHECK: %[[BCAST_RHS_1:.+]] = vector.splat %[[EXTRACT_1]] : vector<2xf16>

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
        tkw.write(res, a, elements_per_thread=4)

    binary_lowerings = wave_compile(get_wave_compile_options(), binary_lowerings)
    print(binary_lowerings.asm)

    # CHECK-LABEL: func @binary_lowerings
    # CHECK: %[[SUB:.+]] = arith.subf
    # CHECK: %[[MUL:.+]] = arith.mulf %[[SUB]]
    # CHECK: %[[DIV:.+]] = arith.divf %[[MUL]]
    # CHECK: %[[MINIMUM:.+]] = arith.minimumf


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
        res = s1 + s2 + s3 + s4
        tkw.write(res, a, elements_per_thread=4)

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
