# RUN: python %s | FileCheck %s

import pytest
from typing import Callable
import iree.turbine.kernel as tk
import iree.turbine.kernel.lang as tkl
import iree.turbine.kernel.wave as tkw
from iree.turbine.kernel.lang.global_symbols import *
from iree.turbine.kernel.wave.utils import run_test
import torch

M = tkl.sym.M
N = tkl.sym.N
K = tkl.sym.K
B = tkl.sym.B
BLOCK_M = tkl.sym.BLOCK_M
BLOCK_N = tkl.sym.BLOCK_N
BLOCK_K = tkl.sym.BLOCK_K
BLOCK_B = tkl.sym.BLOCK_B
LOAD_ELEMS_PER_THREAD = tkl.sym.LOAD_ELEM_PER_THREAD
STORE_ELEMS_PER_THREAD = tkl.sym.STORE_ELEM_PER_THREAD
ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
ADDRESS_SPACE_0 = tkl.sym.ADDRESS_SPACE_0


def codegen_test_context(canonicalize: bool = False, dynamic_symbols=[]):
    bindings = {
        M: 16,
        N: 16,
        K: 16,
        BLOCK_M: 16,
        BLOCK_N: 16,
        BLOCK_K: 16,
        ADDRESS_SPACE: tkl.AddressSpace.SHARED_MEMORY.value,
    }

    # Remove dynamic symbols from the bindings.
    for sym in dynamic_symbols:
        if sym in bindings:
            del bindings[sym]

    return tk.gen.TestLaunchContext(
        bindings, canonicalize=canonicalize, dynamic_symbols=dynamic_symbols
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
    def test(a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16]):
        tkw.read(a, elements_per_thread=16)

    with codegen_test_context():
        a = torch.randn(16, 16, dtype=torch.float16)
        print(test(a).module_op)

        # CHECK:          func.func @test(%[[ARG0:[a-zA-Z0-9_]+]]: !stream.binding)
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
    def test(a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16]):
        tkw.read(a, mapping=mapping, elements_per_thread=16)

    with codegen_test_context():
        a = torch.randn(16, 16, dtype=torch.float16)
        print(test(a).module_op)

        # CHECK:          func.func @test(%[[ARG0:[a-zA-Z0-9_]+]]: !stream.binding)
        # CHECK:            %[[WORKGROUP_ID_0:.+]] = stream.dispatch.workgroup.id[0] : index
        # CHECK:            %[[WORKGROUP_ID_1:.+]] = stream.dispatch.workgroup.id[1] : index
        # CHECK:            %[[WORKGROUP_ID_2:.+]] = stream.dispatch.workgroup.id[2] : index
        # CHECK-DAG:        %[[THREAD_ID_X:.+]] = gpu.thread_id  x
        # CHECK-DAG:        %[[THREAD_ID_Y:.+]] = gpu.thread_id  y
        # CHECK-DAG:        %[[THREAD_ID_Z:.+]] = gpu.thread_id  z
        # CHECK-DAG:        %[[C0:.+]] = arith.constant 0 : index
        # CHECK:            %[[ARR:.+]] = stream.binding.subspan %[[ARG0]][%[[C0]]] : !stream.binding -> memref<16x16xf16,
        # CHECK-SAME:         strided<[16, 1], offset: ?>>
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
        # CHECK:            %[[CST:.+]] = arith.constant dense<[0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240]> : vector<16xindex>
        # CHECK:            %[[MASK:.+]] = vector.constant_mask [16] : vector<16xi1>
        # CHECK-DAG:        %[[CST_2:.+]] = arith.constant 0.000000e+00 : f16
        # CHECK:            %[[D9:.+]] = vector.splat %[[CST_2]] : vector<16xf16>
        # CHECK:            %[[D10:.+]] = vector.gather %[[ARR]][%[[D5]], %[[D8]]] [%[[CST]]], %[[MASK]], %[[D9]] :
        # CHECK-SAME:         memref<16x16xf16, strided<[16, 1], offset: ?>>, vector<16xindex>, vector<16xi1>, vector<16xf16>
        # CHECK-SAME:         into vector<16xf16>


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
    def test(
        a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
    ):
        res = tkw.read(a, elements_per_thread=16)
        tkw.write(res, b, elements_per_thread=16)

    with codegen_test_context(canonicalize=True):
        a = torch.randn(16, 16, dtype=torch.float16)
        b = torch.zeros(16, 16, dtype=torch.float16)
        print(test(a, b).module_op)

        # CHECK:          func.func @test(%[[ARG0:[a-zA-Z0-9_]+]]: !stream.binding, %[[ARG1:[a-zA-Z0-9_]+]]: !stream.binding)
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
    def test(
        a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
    ):
        res = tkw.read(a, elements_per_thread=4)
        tkw.write(res, b, elements_per_thread=4)

    with tk.gen.TestLaunchContext(
        {
            M: 1,
            N: 3,
            BLOCK_M: 4,
            BLOCK_N: 4,
            ADDRESS_SPACE: tkl.AddressSpace.SHARED_MEMORY.value,
        },
        canonicalize=True,
    ):
        a = torch.randn(4, 4, dtype=torch.float16)
        b = torch.zeros(4, 4, dtype=torch.float16)
        print(test(a, b).module_op)

        # CHECK:          func.func @test(%[[ARG0:[a-zA-Z0-9_]+]]: !stream.binding, %[[ARG1:[a-zA-Z0-9_]+]]: !stream.binding)
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
    def test(
        a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[M, N, ADDRESS_SPACE_0, tkl.f16],
    ):
        res = tkw.read(a, elements_per_thread=4)
        tkw.write(res, b, elements_per_thread=4)

    with tk.gen.TestLaunchContext(
        {
            M: 1,
            N: 3,
            BLOCK_M: 4,
            BLOCK_N: 4,
            ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
            ADDRESS_SPACE_0: GLOBAL_ADDRESS_SPACE,
        },
        canonicalize=True,
    ):
        a = torch.randn(4, 4, dtype=torch.float16)
        b = torch.zeros(4, 4, dtype=torch.float16)
        print(test(a, b).module_op)

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
    def test(
        a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[N, M, ADDRESS_SPACE, tkl.f16],
    ):
        res = tkw.read(a, elements_per_thread=16)
        tkw.write(res, b, mapping=mapping, elements_per_thread=16)

    with codegen_test_context(canonicalize=True):
        a = torch.randn(16, 16, dtype=torch.float16)
        b = torch.zeros(16, 16, dtype=torch.float16)
        print(test(a, b).module_op)

        # CHECK:          func.func @test(%[[ARG0:[a-zA-Z0-9_]+]]: !stream.binding, %[[ARG1:[a-zA-Z0-9_]+]]: !stream.binding)
        # CHECK:            %[[CST:.+]] = arith.constant dense<[0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208,
        # CHECK-SAME:         224, 240]> : vector<16xindex>
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
        # CHECK:            %[[D11:.+]] = vector.constant_mask [16] : vector<16xi1>
        # CHECK:            vector.scatter %[[D10]][%[[D8]], %[[D5]]] [%[[CST]]], %[[D11]], %[[D9]] : memref<16x16xf16,
        # CHECK-SAME:         strided<[16, 1], offset: ?>>, vector<16xindex>, vector<16xi1>, vector<16xf16>


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
    def test(a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16]):
        b = tkw.read(a, elements_per_thread=16)
        tkw.write(b, a, elements_per_thread=16)

    with codegen_test_context(canonicalize=True, dynamic_symbols=[M, N]):
        a = torch.randn(16, 16, dtype=torch.float16)
        print(test(a).module_op)

    # CHECK:        stream.executable.export public @test workgroups(%[[ARG0:.*]]: index, %[[ARG1:.*]]:
    # CHECK-SAME:       index) -> (index, index, index) {
    # CHECK-DAG:        %[[C1:.+]] = arith.constant 1 : index
    # CHECK-DAG:        %[[C16:.+]] = arith.constant 16 : index
    # CHECK:            %[[D0:.+]] = arith.ceildivsi %[[ARG0]], %[[C16]] : index
    # CHECK:            %[[D1:.+]] = arith.ceildivsi %[[ARG1]], %[[C16]] : index
    # CHECK:            stream.return %[[D0]], %[[D1]], %[[C1]] : index, index, index
    # CHECK:          }
    # CHECK:          func.func @test(%[[ARG0:.*]]: !stream.binding, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index)
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
    # CHECK:            %[[D0]] = stream.binding.subspan %[[ARG0]][%[[C0]]] : !stream.binding -> memref<?x?xf16>{%[[ARG1]],
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
    # CHECK:            %[[D16:.+]] = vector.maskedload %[[D0]][%[[D5]], %[[D8]]], %[[D15]], %[[CST]] : memref<?x?xf16>,
    # CHECK-SAME:         vector<16xi1>, vector<16xf16> into vector<16xf16>
    # CHECK:            vector.maskedstore %[[D0]][%[[D5]], %[[D8]]], %[[D15]], %[[D16]] : memref<?x?xf16>, vector<16xi1>,
    # CHECK-SAME:         vector<16xf16>
    # CHECK:            return


@run_test
def test_gemm():
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.TilingConstraint(K, BLOCK_K)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 2)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]

    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(2, 2, 1),
            mma_type=tkw.MMAType.F32_16x16x16_F16,
        )
    ]

    @tkw.wave(constraints)
    def gemm(
        a: tkl.Memory[M, K, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[N, K, ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[M, N, ADDRESS_SPACE_0, tkl.f32],
    ):
        c_reg = tkl.Register[M, N, tkl.f32](0.0)

        @tkw.reduction(K, init_args=[c_reg])
        def repeat(acc: tkl.Register[M, N, tkl.f32]) -> tkl.Register[M, N, tkl.f32]:
            a_reg = tkw.read(a, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            b_reg = tkw.read(b, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            acc = tkw.mma(a_reg, b_reg, acc)
            return acc

        tkw.write(repeat, c, elements_per_thread=STORE_ELEMS_PER_THREAD)

    with tk.gen.TestLaunchContext(
        {
            M: 64,
            N: 128,
            K: 64,
            BLOCK_M: 32,
            BLOCK_N: 32,
            BLOCK_K: 16,
            LOAD_ELEMS_PER_THREAD: 4,
            STORE_ELEMS_PER_THREAD: 4,
            ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
            ADDRESS_SPACE_0: GLOBAL_ADDRESS_SPACE,
        },
        canonicalize=True,
    ):
        a = torch.randn(64, 32, dtype=torch.float16)
        b = torch.randn(128, 32, dtype=torch.float16)
        c = torch.zeros(64, 128, dtype=torch.float32)
        print(gemm(a, b, c).module_op)

        # CHECK:          func.func @gemm(%[[ARG0:[a-zA-Z0-9_]+]]: !stream.binding, %[[ARG1:[a-zA-Z0-9_]+]]: !stream.binding,
        # CHECK-SAME:       %[[ARG2:[a-zA-Z0-9_]+]]: !stream.binding) attributes {translation_info = #[[TRANSLATION:.+]]} {
        # CHECK-DAG:        %[[C3:.+]] = arith.constant 3 : index
        # CHECK-DAG:        %[[C2:.+]] = arith.constant 2 : index
        # CHECK-DAG:        %[[C32:.+]] = arith.constant 32 : index
        # CHECK-DAG:        %[[C64:.+]] = arith.constant 64 : index
        # CHECK-DAG:        %[[C16:.+]] = arith.constant 16 : index
        # CHECK-DAG:        %[[C1:.+]] = arith.constant 1 : index
        # CHECK-DAG:        %[[C4:.+]] = arith.constant 4 : index
        # CHECK-DAG:        %[[C0:.+]] = arith.constant 0 : index
        # CHECK-DAG:        %[[CST:.+]] = arith.constant dense<0.000000e+00> : vector<4xf32>
        # CHECK:            %[[WORKGROUP_ID_0:.+]] = stream.dispatch.workgroup.id[0] : index
        # CHECK:            %[[WORKGROUP_ID_1:.+]] = stream.dispatch.workgroup.id[1] : index
        # CHECK-DAG:        %[[THREAD_ID_X:.+]] = gpu.thread_id  x
        # CHECK-DAG:        %[[THREAD_ID_Y:.+]] = gpu.thread_id  y
        # CHECK:            %[[ALLOC:.+]] = memref.alloc() : memref<32x20xf16, #[[GPU:.+]].address_space<workgroup>>
        # CHECK:            %[[ALLOC_0:.+]] = memref.alloc() : memref<32x20xf16, #[[GPU]].address_space<workgroup>>
        # CHECK:            %[[D0:.+]] = stream.binding.subspan %[[ARG0]][%[[C0]]] : !stream.binding -> memref<64x64xf16,
        # CHECK-SAME:         strided<[64, 1], offset: ?>>
        # CHECK:            %[[D1:.+]] = stream.binding.subspan %[[ARG1]][%[[C0]]] : !stream.binding -> memref<128x64xf16,
        # CHECK-SAME:         strided<[64, 1], offset: ?>>
        # CHECK:            %[[D2:.+]] = arith.divsi %[[THREAD_ID_X]], %[[C64]] : index
        # CHECK:            %[[D3:.+]] = arith.muli %[[D2]], %[[C16]] overflow<nsw, nuw> : index
        # CHECK:            %[[D4:.+]] = arith.muli %[[WORKGROUP_ID_0]], %[[C32]] overflow<nsw, nuw> : index
        # CHECK:            %[[D5:.+]] = arith.remsi %[[THREAD_ID_X]], %[[C16]] : index
        # CHECK:            %[[D6:.+]] = arith.addi %[[D5]], %[[D4]] overflow<nsw, nuw> : index
        # CHECK:            %[[D7:.+]] = arith.addi %[[D6]], %[[D3]] overflow<nsw, nuw> : index
        # CHECK:            %[[D8:.+]] = arith.remsi %[[THREAD_ID_X]], %[[C64]] : index
        # CHECK:            %[[D9:.+]] = arith.divsi %[[D8]], %[[C16]] : index
        # CHECK:            %[[D10:.+]] = arith.muli %[[D9]], %[[C4]] overflow<nsw, nuw> : index
        # CHECK:            %[[D11:.+]] = arith.addi %[[D5]], %[[D3]] overflow<nsw, nuw> : index
        # CHECK:            %[[D12:.+]] = arith.muli %[[THREAD_ID_Y]], %[[C16]] overflow<nsw, nuw> : index
        # CHECK:            %[[D13:.+]] = arith.muli %[[WORKGROUP_ID_1]], %[[C32]] overflow<nsw, nuw> : index
        # CHECK:            %[[D14:.+]] = arith.addi %[[D5]], %[[D13]] overflow<nsw, nuw> : index
        # CHECK:            %[[D15:.+]] = arith.addi %[[D14]], %[[D12]] overflow<nsw, nuw> : index
        # CHECK:            %[[D16:.+]] = arith.addi %[[D5]], %[[D12]] overflow<nsw, nuw> : index
        # CHECK:            %[[D17:.+]] = scf.for %[[ARG3:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C4]] step %[[C1]]
        # CHECK-SAME:         iter_args(%[[ARG4:[a-zA-Z0-9_]+]] = %[[CST]]) -> (vector<4xf32>) {
        # CHECK:              %[[D39:.+]] = arith.muli %[[ARG3]], %[[C16]] overflow<nsw, nuw> : index
        # CHECK:              %[[D40:.+]] = arith.addi %[[D39]], %[[D10]] overflow<nsw, nuw> : index
        # CHECK:              %[[D41:.+]] = vector.load %[[D0]][%[[D7]], %[[D40]]] : memref<64x64xf16, strided<[64, 1], offset:
        # CHECK-SAME:           ?>>, vector<4xf16>
        # CHECK:              vector.store %[[D41]], %[[ALLOC]][%[[D11]], %[[D10]]] : memref<32x20xf16,
        # CHECK-SAME:           #[[GPU]].address_space<workgroup>>, vector<4xf16>
        # CHECK:              amdgpu.lds_barrier
        # CHECK:              %[[D42:.+]] = vector.load %[[ALLOC]][%[[D11]], %[[D10]]] : memref<32x20xf16,
        # CHECK-SAME:           #[[GPU]].address_space<workgroup>>, vector<4xf16>
        # CHECK:              %[[D43:.+]] = vector.load %[[D1]][%[[D15]], %[[D40]]] : memref<128x64xf16, strided<[64, 1],
        # CHECK-SAME:           offset: ?>>, vector<4xf16>
        # CHECK:              amdgpu.lds_barrier
        # CHECK:              vector.store %[[D43]], %[[ALLOC_0]][%[[D16]], %[[D10]]] : memref<32x20xf16,
        # CHECK-SAME:           #[[GPU]].address_space<workgroup>>, vector<4xf16>
        # CHECK:              amdgpu.lds_barrier
        # CHECK:              %[[D44:.+]] = vector.load %[[ALLOC_0]][%[[D16]], %[[D10]]] : memref<32x20xf16,
        # CHECK-SAME:           #[[GPU]].address_space<workgroup>>, vector<4xf16>
        # CHECK:              %[[D45:.+]] = amdgpu.mfma %[[D42]] * %[[D44]] + %[[ARG4]] {blocks = 1 : i32, k = 16 : i32, m = 16
        # CHECK-SAME:           : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        # CHECK:              scf.yield %[[D45]] : vector<4xf32>
        # CHECK:            }
        # CHECK:            %[[D18:.+]] = vector.extract_strided_slice %[[D17]] {offsets = [0], sizes = [1], strides = [1]} :
        # CHECK-SAME:         vector<4xf32> to vector<1xf32>
        # CHECK:            %[[D19:.+]] = stream.binding.subspan %[[ARG2]][%[[C0]]] : !stream.binding -> memref<64x128xf32,
        # CHECK-SAME:         strided<[128, 1], offset: ?>>
        # CHECK:            %[[D20:.+]] = arith.remsi %[[THREAD_ID_X]], %[[C64]] : index
        # CHECK:            %[[D21:.+]] = arith.divsi %[[D20]], %[[C16]] : index
        # CHECK:            %[[D22:.+]] = arith.muli %[[D21]], %[[C4]] overflow<nsw, nuw> : index
        # CHECK:            %[[D23:.+]] = arith.divsi %[[THREAD_ID_X]], %[[C64]] : index
        # CHECK:            %[[D24:.+]] = arith.muli %[[D23]], %[[C16]] overflow<nsw, nuw> : index
        # CHECK:            %[[D25:.+]] = arith.muli %[[WORKGROUP_ID_0]], %[[C32]] overflow<nsw, nuw> : index
        # CHECK:            %[[D26:.+]] = arith.addi %[[D25]], %[[D24]] overflow<nsw, nuw> : index
        # CHECK:            %[[D27:.+]] = arith.addi %[[D26]], %[[D22]] overflow<nsw, nuw> : index
        # CHECK:            %[[D28:.+]] = arith.muli %[[THREAD_ID_Y]], %[[C16]] overflow<nsw, nuw> : index
        # CHECK:            %[[D29:.+]] = arith.muli %[[WORKGROUP_ID_1]], %[[C32]] overflow<nsw, nuw> : index
        # CHECK:            %[[D30:.+]] = arith.remsi %[[THREAD_ID_X]], %[[C16]] : index
        # CHECK:            %[[D31:.+]] = arith.addi %[[D30]], %[[D29]] overflow<nsw, nuw> : index
        # CHECK:            %[[D32:.+]] = arith.addi %[[D31]], %[[D28]] overflow<nsw, nuw> : index
        # CHECK:            vector.store %[[D18]], %[[D19]][%[[D27]], %[[D32]]] : memref<64x128xf32, strided<[128, 1], offset:
        # CHECK-SAME:         ?>>, vector<1xf32>
        # CHECK:            %[[D33:.+]] = vector.extract_strided_slice %[[D17]] {offsets = [1], sizes = [1], strides = [1]} :
        # CHECK-SAME:         vector<4xf32> to vector<1xf32>
        # CHECK:            %[[D34:.+]] = arith.addi %[[D27]], %[[C1]] overflow<nsw, nuw> : index
        # CHECK:            vector.store %[[D33]], %[[D19]][%[[D34]], %[[D32]]] : memref<64x128xf32, strided<[128, 1], offset:
        # CHECK-SAME:         ?>>, vector<1xf32>
        # CHECK:            %[[D35:.+]] = vector.extract_strided_slice %[[D17]] {offsets = [2], sizes = [1], strides = [1]} :
        # CHECK-SAME:         vector<4xf32> to vector<1xf32>
        # CHECK:            %[[D36:.+]] = arith.addi %[[D27]], %[[C2]] overflow<nsw, nuw> : index
        # CHECK:            vector.store %[[D35]], %[[D19]][%[[D36]], %[[D32]]] : memref<64x128xf32, strided<[128, 1], offset:
        # CHECK-SAME:         ?>>, vector<1xf32>
        # CHECK:            %[[D37:.+]] = vector.extract_strided_slice %[[D17]] {offsets = [3], sizes = [1], strides = [1]} :
        # CHECK-SAME:         vector<4xf32> to vector<1xf32>
        # CHECK:            %[[D38:.+]] = arith.addi %[[D27]], %[[C3]] overflow<nsw, nuw> : index
        # CHECK:            vector.store %[[D37]], %[[D19]][%[[D38]], %[[D32]]] : memref<64x128xf32, strided<[128, 1], offset:
        # CHECK-SAME:         ?>>, vector<1xf32>
        # CHECK:            return


@run_test
def test_batched_gemm():
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.WorkgroupConstraint(B, BLOCK_B, 2)]
    constraints += [tkw.TilingConstraint(K, BLOCK_K)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 2)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]

    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(2, 2, 1),
            mma_type=tkw.MMAType.F32_16x16x16_F16,
            vector_shapes={B: 0},
        )
    ]

    @tkw.wave(constraints)
    def batched_gemm(
        a: tkl.Memory[B, M, K, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[B, N, K, ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[B, M, N, ADDRESS_SPACE_0, tkl.f32],
    ):
        c_reg = tkl.Register[B, M, N, tkl.f32](0.0)

        @tkw.reduction(K, init_args=[c_reg])
        def repeat(
            acc: tkl.Register[B, M, N, tkl.f32]
        ) -> tkl.Register[B, M, N, tkl.f32]:
            a_reg = tkw.read(a, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            b_reg = tkw.read(b, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            acc = tkw.mma(a_reg, b_reg, acc)
            return acc

        tkw.write(repeat, c, elements_per_thread=STORE_ELEMS_PER_THREAD)

    with tk.gen.TestLaunchContext(
        {
            B: 12,
            M: 64,
            N: 128,
            K: 64,
            BLOCK_M: 32,
            BLOCK_N: 32,
            BLOCK_K: 16,
            BLOCK_B: 1,
            LOAD_ELEMS_PER_THREAD: 4,
            STORE_ELEMS_PER_THREAD: 4,
            ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
            ADDRESS_SPACE_0: GLOBAL_ADDRESS_SPACE,
        },
        canonicalize=True,
    ):
        a = torch.randn(12, 64, 32, dtype=torch.float16)
        b = torch.randn(12, 128, 32, dtype=torch.float16)
        c = torch.zeros(12, 64, 128, dtype=torch.float32)
        print(batched_gemm(a, b, c).module_op)

        # CHECK:          func.func @batched_gemm(%[[ARG0:[a-zA-Z0-9_]+]]: !stream.binding, %[[ARG1:[a-zA-Z0-9_]+]]:
        # CHECK-SAME:       !stream.binding, %[[ARG2:[a-zA-Z0-9_]+]]: !stream.binding) attributes {translation_info =
        # CHECK-SAME:       #[[TRANSLATION:.+]]} {
        # CHECK-DAG:        %[[C3:.+]] = arith.constant 3 : index
        # CHECK-DAG:        %[[C2:.+]] = arith.constant 2 : index
        # CHECK-DAG:        %[[C32:.+]] = arith.constant 32 : index
        # CHECK-DAG:        %[[C64:.+]] = arith.constant 64 : index
        # CHECK-DAG:        %[[C16:.+]] = arith.constant 16 : index
        # CHECK-DAG:        %[[C1:.+]] = arith.constant 1 : index
        # CHECK-DAG:        %[[C4:.+]] = arith.constant 4 : index
        # CHECK-DAG:        %[[C0:.+]] = arith.constant 0 : index
        # CHECK-DAG:        %[[CST:.+]] = arith.constant dense<0.000000e+00> : vector<4xf32>
        # CHECK:            %[[WORKGROUP_ID_0:.+]] = stream.dispatch.workgroup.id[0] : index
        # CHECK:            %[[WORKGROUP_ID_1:.+]] = stream.dispatch.workgroup.id[1] : index
        # CHECK:            %[[WORKGROUP_ID_2:.+]] = stream.dispatch.workgroup.id[2] : index
        # CHECK-DAG:        %[[THREAD_ID_X:.+]] = gpu.thread_id  x
        # CHECK-DAG:        %[[THREAD_ID_Y:.+]] = gpu.thread_id  y
        # CHECK:            %[[ALLOC:.+]] = memref.alloc() : memref<1x32x20xf16, #[[GPU:.+]].address_space<workgroup>>
        # CHECK:            %[[ALLOC_0:.+]] = memref.alloc() : memref<1x32x20xf16, #[[GPU]].address_space<workgroup>>
        # CHECK:            %[[D0:.+]] = stream.binding.subspan %[[ARG0]][%[[C0]]] : !stream.binding -> memref<12x64x64xf16,
        # CHECK-SAME:         strided<[4096, 64, 1], offset: ?>>
        # CHECK:            %[[D1:.+]] = stream.binding.subspan %[[ARG1]][%[[C0]]] : !stream.binding -> memref<12x128x64xf16,
        # CHECK-SAME:         strided<[8192, 64, 1], offset: ?>>
        # CHECK:            %[[D2:.+]] = arith.divsi %[[THREAD_ID_X]], %[[C64]] : index
        # CHECK:            %[[D3:.+]] = arith.muli %[[D2]], %[[C16]] overflow<nsw, nuw> : index
        # CHECK:            %[[D4:.+]] = arith.muli %[[WORKGROUP_ID_0]], %[[C32]] overflow<nsw, nuw> : index
        # CHECK:            %[[D5:.+]] = arith.remsi %[[THREAD_ID_X]], %[[C16]] : index
        # CHECK:            %[[D6:.+]] = arith.addi %[[D5]], %[[D4]] overflow<nsw, nuw> : index
        # CHECK:            %[[D7:.+]] = arith.addi %[[D6]], %[[D3]] overflow<nsw, nuw> : index
        # CHECK:            %[[D8:.+]] = arith.remsi %[[THREAD_ID_X]], %[[C64]] : index
        # CHECK:            %[[D9:.+]] = arith.divsi %[[D8]], %[[C16]] : index
        # CHECK:            %[[D10:.+]] = arith.muli %[[D9]], %[[C4]] overflow<nsw, nuw> : index
        # CHECK:            %[[D11:.+]] = arith.addi %[[D5]], %[[D3]] overflow<nsw, nuw> : index
        # CHECK:            %[[D12:.+]] = arith.muli %[[THREAD_ID_Y]], %[[C16]] overflow<nsw, nuw> : index
        # CHECK:            %[[D13:.+]] = arith.muli %[[WORKGROUP_ID_1]], %[[C32]] overflow<nsw, nuw> : index
        # CHECK:            %[[D14:.+]] = arith.addi %[[D5]], %[[D13]] overflow<nsw, nuw> : index
        # CHECK:            %[[D15:.+]] = arith.addi %[[D14]], %[[D12]] overflow<nsw, nuw> : index
        # CHECK:            %[[D16:.+]] = arith.addi %[[D5]], %[[D12]] overflow<nsw, nuw> : index
        # CHECK:            %[[D17:.+]] = scf.for %[[ARG3:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C4]] step %[[C1]]
        # CHECK-SAME:         iter_args(%[[ARG4:[a-zA-Z0-9_]+]] = %[[CST]]) -> (vector<4xf32>) {
        # CHECK:              %[[D39:.+]] = arith.muli %[[ARG3]], %[[C16]] overflow<nsw, nuw> : index
        # CHECK:              %[[D40:.+]] = arith.addi %[[D39]], %[[D10]] overflow<nsw, nuw> : index
        # CHECK:              %[[D41:.+]] = vector.load %[[D0]][%[[WORKGROUP_ID_2]], %[[D7]], %[[D40]]] : memref<12x64x64xf16,
        # CHECK-SAME:           strided<[4096, 64, 1], offset: ?>>, vector<4xf16>
        # CHECK:              vector.store %[[D41]], %[[ALLOC]][%[[C0]], %[[D11]], %[[D10]]] : memref<1x32x20xf16,
        # CHECK-SAME:           #[[GPU]].address_space<workgroup>>, vector<4xf16>
        # CHECK:              amdgpu.lds_barrier
        # CHECK:              %[[D42:.+]] = vector.load %[[ALLOC]][%[[C0]], %[[D11]], %[[D10]]] : memref<1x32x20xf16,
        # CHECK-SAME:           #[[GPU]].address_space<workgroup>>, vector<4xf16>
        # CHECK:              %[[D43:.+]] = vector.load %[[D1]][%[[WORKGROUP_ID_2]], %[[D15]], %[[D40]]] : memref<12x128x64xf16,
        # CHECK-SAME:           strided<[8192, 64, 1], offset: ?>>, vector<4xf16>
        # CHECK:              amdgpu.lds_barrier
        # CHECK:              vector.store %[[D43]], %[[ALLOC_0]][%[[C0]], %[[D16]], %[[D10]]] : memref<1x32x20xf16,
        # CHECK-SAME:           #[[GPU]].address_space<workgroup>>, vector<4xf16>
        # CHECK:              amdgpu.lds_barrier
        # CHECK:              %[[D44:.+]] = vector.load %[[ALLOC_0]][%[[C0]], %[[D16]], %[[D10]]] : memref<1x32x20xf16,
        # CHECK-SAME:           #[[GPU]].address_space<workgroup>>, vector<4xf16>
        # CHECK:              %[[D45:.+]] = amdgpu.mfma %[[D42]] * %[[D44]] + %[[ARG4]] {blocks = 1 : i32, k = 16 : i32, m = 16
        # CHECK-SAME:           : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        # CHECK:              scf.yield %[[D45]] : vector<4xf32>
        # CHECK:            }
        # CHECK:            %[[D18:.+]] = vector.extract_strided_slice %[[D17]] {offsets = [0], sizes = [1], strides = [1]} :
        # CHECK-SAME:         vector<4xf32> to vector<1xf32>
        # CHECK:            %[[D19:.+]] = stream.binding.subspan %[[ARG2]][%[[C0]]] : !stream.binding -> memref<12x64x128xf32,
        # CHECK-SAME:         strided<[8192, 128, 1], offset: ?>>
        # CHECK:            %[[D20:.+]] = arith.remsi %[[THREAD_ID_X]], %[[C64]] : index
        # CHECK:            %[[D21:.+]] = arith.divsi %[[D20]], %[[C16]] : index
        # CHECK:            %[[D22:.+]] = arith.muli %[[D21]], %[[C4]] overflow<nsw, nuw> : index
        # CHECK:            %[[D23:.+]] = arith.divsi %[[THREAD_ID_X]], %[[C64]] : index
        # CHECK:            %[[D24:.+]] = arith.muli %[[D23]], %[[C16]] overflow<nsw, nuw> : index
        # CHECK:            %[[D25:.+]] = arith.muli %[[WORKGROUP_ID_0]], %[[C32]] overflow<nsw, nuw> : index
        # CHECK:            %[[D26:.+]] = arith.addi %[[D25]], %[[D24]] overflow<nsw, nuw> : index
        # CHECK:            %[[D27:.+]] = arith.addi %[[D26]], %[[D22]] overflow<nsw, nuw> : index
        # CHECK:            %[[D28:.+]] = arith.muli %[[THREAD_ID_Y]], %[[C16]] overflow<nsw, nuw> : index
        # CHECK:            %[[D29:.+]] = arith.muli %[[WORKGROUP_ID_1]], %[[C32]] overflow<nsw, nuw> : index
        # CHECK:            %[[D30:.+]] = arith.remsi %[[THREAD_ID_X]], %[[C16]] : index
        # CHECK:            %[[D31:.+]] = arith.addi %[[D30]], %[[D29]] overflow<nsw, nuw> : index
        # CHECK:            %[[D32:.+]] = arith.addi %[[D31]], %[[D28]] overflow<nsw, nuw> : index
        # CHECK:            vector.store %[[D18]], %[[D19]][%[[WORKGROUP_ID_2]], %[[D27]], %[[D32]]] : memref<12x64x128xf32,
        # CHECK-SAME:         strided<[8192, 128, 1], offset: ?>>, vector<1xf32>
        # CHECK:            %[[D33:.+]] = vector.extract_strided_slice %[[D17]] {offsets = [1], sizes = [1], strides = [1]} :
        # CHECK-SAME:         vector<4xf32> to vector<1xf32>
        # CHECK:            %[[D34:.+]] = arith.addi %[[D27]], %[[C1]] overflow<nsw, nuw> : index
        # CHECK:            vector.store %[[D33]], %[[D19]][%[[WORKGROUP_ID_2]], %[[D34]], %[[D32]]] : memref<12x64x128xf32,
        # CHECK-SAME:         strided<[8192, 128, 1], offset: ?>>, vector<1xf32>
        # CHECK:            %[[D35:.+]] = vector.extract_strided_slice %[[D17]] {offsets = [2], sizes = [1], strides = [1]} :
        # CHECK-SAME:         vector<4xf32> to vector<1xf32>
        # CHECK:            %[[D36:.+]] = arith.addi %[[D27]], %[[C2]] overflow<nsw, nuw> : index
        # CHECK:            vector.store %[[D35]], %[[D19]][%[[WORKGROUP_ID_2]], %[[D36]], %[[D32]]] : memref<12x64x128xf32,
        # CHECK-SAME:         strided<[8192, 128, 1], offset: ?>>, vector<1xf32>
        # CHECK:            %[[D37:.+]] = vector.extract_strided_slice %[[D17]] {offsets = [3], sizes = [1], strides = [1]} :
        # CHECK-SAME:         vector<4xf32> to vector<1xf32>
        # CHECK:            %[[D38:.+]] = arith.addi %[[D27]], %[[C3]] overflow<nsw, nuw> : index
        # CHECK:            vector.store %[[D37]], %[[D19]][%[[WORKGROUP_ID_2]], %[[D38]], %[[D32]]] : memref<12x64x128xf32,
        # CHECK-SAME:         strided<[8192, 128, 1], offset: ?>>, vector<1xf32>
        # CHECK:            return


@run_test
def test_chained_gemm():
    K1 = tkl.sym.K1
    K2 = tkl.sym.K2
    BLOCK_K2 = tkl.sym.BLOCK_K2

    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.WorkgroupConstraint(B, BLOCK_B, 2)]
    constraints += [tkw.TilingConstraint(K2, BLOCK_K2)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 2)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]

    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(2, 2, 1),
            mma_type=tkw.MMAType.F32_16x16x16_F16,
            vector_shapes={B: 0},
        )
    ]

    @tkw.wave(constraints)
    def chained_gemm(
        q: tkl.Memory[B, M, K1, ADDRESS_SPACE, tkl.f16],
        k: tkl.Memory[B, K2, K1, ADDRESS_SPACE, tkl.f16],
        v: tkl.Memory[B, N, K2, ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[B, M, N, ADDRESS_SPACE_0, tkl.f32],
    ):
        c_reg = tkl.Register[B, M, N, tkl.f32](0.0)

        @tkw.reduction(K2, init_args=[c_reg])
        def repeat(
            acc: tkl.Register[B, M, N, tkl.f32]
        ) -> tkl.Register[B, M, N, tkl.f32]:
            inner_acc = tkl.Register[B, K2, M, tkl.f32](0.0)
            q_reg = tkw.read(q, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            k_reg = tkw.read(k, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            kq_reg = tkw.mma(k_reg, q_reg, inner_acc)
            qk_reg = tkw.permute(kq_reg, target_shape=[B, M, K2])
            qk_cast_reg = tkw.cast(qk_reg, tkl.f16)
            v_reg = tkw.read(v, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            acc = tkw.mma(qk_cast_reg, v_reg, acc)
            return acc

        tkw.write(repeat, c, elements_per_thread=STORE_ELEMS_PER_THREAD)

    with tk.gen.TestLaunchContext(
        {
            M: 128,
            N: 128,
            K1: 32,
            K2: 256,
            B: 8,
            BLOCK_M: 64,
            BLOCK_N: 64,
            BLOCK_K2: 32,
            BLOCK_B: 1,
            LOAD_ELEMS_PER_THREAD: 4,
            STORE_ELEMS_PER_THREAD: 4,
            ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
            ADDRESS_SPACE_0: GLOBAL_ADDRESS_SPACE,
        },
        canonicalize=True,
    ):
        q = torch.randn(8, 64, 64, dtype=torch.float16)
        k = torch.randn(8, 256, 64, dtype=torch.float16)
        v = torch.zeros(8, 128, 256, dtype=torch.float16)
        output = torch.zeros(8, 64, 128, dtype=torch.float32)
        print(chained_gemm(q, k, v, output).module_op)

        # CHECK:           func.func @chained_gemm(
        # CHECK:             {{.*}} = scf.for
        # CHECK-COUNT-8:       {{.*}} = amdgpu.mfma
        # CHECK-COUNT-4:       {{.*}} = arith.truncf
        # CHECK-COUNT-8:       {{.*}} = amdgpu.mfma
        # CHECK:             scf.yield


@run_test
def test_gemm_pipelined():
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.TilingConstraint(K, BLOCK_K)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 2)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]

    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(2, 2, 1),
            mma_type=tkw.MMAType.F32_16x16x16_F16,
        )
    ]

    @tkw.wave(constraints)
    def gemm_pipelined(
        a: tkl.Memory[M, K, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[N, K, ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[M, N, ADDRESS_SPACE_0, tkl.f32],
    ):
        c_reg = tkl.Register[M, N, tkl.f32](0.0)

        @tkw.reduction(K, init_args=[c_reg])
        def repeat(acc: tkl.Register[M, N, tkl.f32]) -> tkl.Register[M, N, tkl.f32]:
            a_reg = tkw.read(a, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            b_reg = tkw.read(b, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            acc = tkw.mma(a_reg, b_reg, acc)
            return acc

        tkw.write(repeat, c, elements_per_thread=STORE_ELEMS_PER_THREAD)

    with tk.gen.TestLaunchContext(
        {
            M: 128,
            N: 128,
            K: 128,
            BLOCK_M: 64,
            BLOCK_N: 64,
            BLOCK_K: 32,
            LOAD_ELEMS_PER_THREAD: 4,
            STORE_ELEMS_PER_THREAD: 4,
            ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
            ADDRESS_SPACE_0: GLOBAL_ADDRESS_SPACE,
            READ_SHARED_DELAY: 1,
            WRITE_SHARED_DELAY: 1,
            READ_GLOBAL_DELAY: 2,
            WRITE_GLOBAL_DELAY: 2,
            MMA_DELAY: 1,
            SHARED_MEMORY_UNITS: 4,
            GLOBAL_MEMORY_UNITS: 4,
            MMA_UNITS: 4,
        },
        canonicalize=True,
        schedule=True,
        use_scheduling_barriers=True,
    ):
        a = torch.randn(64, 32, dtype=torch.float16)
        b = torch.randn(128, 32, dtype=torch.float16)
        c = torch.zeros(64, 128, dtype=torch.float32)
        print(gemm_pipelined(a, b, c).module_op)

        # CHECK:          func.func @gemm_pipelined
        # CHECK-COUNT-2:    vector.load
        # CHECK-COUNT-2:    vector.store
        # CHECK-COUNT-1:    amdgpu.lds_barrier
        # CHECK-COUNT-10:   vector.load
        # CHECK-COUNT-4:    amdgpu.mfma
        # CHECK-COUNT-1:    amdgpu.lds_barrier
        # CHECK-COUNT-2:    vector.store
        # CHECK-COUNT-1:    scf.for
        # CHECK-COUNT-4:    amdgpu.mfma
        # CHECK-COUNT-1:    amdgpu.lds_barrier
        # CHECK-COUNT-6:    vector.load
        # CHECK-COUNT-3:    llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"
        # CHECK-COUNT-4:    vector.load
        # CHECK-COUNT-1:    llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"
        # CHECK-COUNT-4:    amdgpu.mfma
        # CHECK-COUNT-1:    amdgpu.lds_barrier
        # CHECK-COUNT-2:    vector.store
        # CHECK-COUNT-2:    llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"
        # CHECK-COUNT-1:    scf.yield
        # CHECK-COUNT-4:    amdgpu.mfma
        # CHECK-COUNT-1:    amdgpu.lds_barrier
        # CHECK-COUNT-8:    vector.load
        # CHECK-COUNT-8:    amdgpu.mfma


# This test is used to check three things
# 1. Reduction with multiple different types(MMA, ReduceOp) of iterArg works
# 2. ReduceOp lowering works using constraints from MMA (not just vector_shape).
# 3. We can propagate layout of multiple Reduction results through IterArg/GetResult
#    and observe that broadcast is being generated to resolve binaryOp.
@run_test
def test_gemm_and_reduce():
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.TilingConstraint(K, BLOCK_K)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 2)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]

    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(2, 2, 1),
            mma_type=tkw.MMAType.F32_16x16x16_F16,
        )
    ]

    @tkw.wave(constraints)
    def gemm(
        a: tkl.Memory[M, K, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[N, K, ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[M, N, ADDRESS_SPACE_0, tkl.f32],
    ):
        c_reg = tkl.Register[M, N, tkl.f32](0.0)
        init_max = tkl.Register[M, tkl.f16](-1e6)

        @tkw.reduction(K, init_args=[init_max, c_reg])
        def repeat(
            partial_max: tkl.Register[M, tkl.f16], acc: tkl.Register[M, N, tkl.f32]
        ) -> tkl.Register[M, N, tkl.f32]:
            a_reg = tkw.read(a, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            b_reg = tkw.read(b, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            partial_max = tkw.max(a_reg, partial_max, dim=K)
            acc = tkw.mma(a_reg, b_reg, acc)
            return partial_max, acc

        res_max, res_mm = repeat
        res = res_mm / tkw.cast(res_max, tkl.f32)
        tkw.write(res, c, elements_per_thread=STORE_ELEMS_PER_THREAD)

    with tk.gen.TestLaunchContext(
        {
            M: 64,
            N: 128,
            K: 64,
            BLOCK_M: 32,
            BLOCK_N: 32,
            BLOCK_K: 16,
            LOAD_ELEMS_PER_THREAD: 4,
            STORE_ELEMS_PER_THREAD: 4,
            ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
            ADDRESS_SPACE_0: GLOBAL_ADDRESS_SPACE,
        },
        canonicalize=True,
    ):
        a = torch.randn(64, 32, dtype=torch.float16)
        b = torch.randn(128, 32, dtype=torch.float16)
        c = torch.zeros(64, 128, dtype=torch.float32)
        print(gemm(a, b, c).module_op)
        # CHECK-DAG: %[[C0_IDX:.+]] = arith.constant 0 : index
        # CHECK-DAG: %[[C4_IDX:.+]] = arith.constant 4 : index
        # CHECK-DAG: %[[C1_IDX:.+]] = arith.constant 1 : index

        # Tile Reduction Loop
        # Note: Shape is 32x20 instead of 32x16 because of padding to avoid bank conflicts
        # CHECK: %[[LOOP:.+]]:2 = scf.for %[[ITER:.+]] = %[[C0_IDX]] to %[[C4_IDX]] step %[[C1_IDX]]
        # CHECK-SAME: iter_args(%[[ACC0:.+]] = %{{.*}}, %[[ACC1:.+]] = {{.*}})
        # CHECK-COUNT-2: vector.load{{.*}} memref<32x20xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        # CHECK-COUNT-2: gpu.shuffle  xor
        #         CHECK: %[[MAX:.+]] = arith.maximumf %[[ACC0]], %{{.*}}
        #         CHECK: %[[MMA:.+]] = amdgpu.mfma %{{.*}} * %{{.*}} + %[[ACC1]]
        #         CHECK: scf.yield %[[MAX]], %[[MMA]] : vector<1xf16>, vector<4xf32>
        # CHECK: %[[MAX_EXT:.+]] = arith.extf %[[LOOP]]#0 : vector<1xf16> to vector<1xf32>
        # CHECK: %[[BCAST_SRC:.+]] = vector.extract %[[MAX_EXT]][0] : f32 from vector<1xf32>
        # CHECK: %[[BROADCAST:.+]] = vector.splat %19 : vector<4xf32>
        # CHECK: arith.divf %[[LOOP]]#1, %[[BROADCAST]] : vector<4xf32>


@run_test
def test_igemm():
    n, c, h, w = 2, 640, 64, 64
    cf, hf, wf, nf = c, 3, 3, 640
    stride = 1
    padding = 0

    x = torch.randn(n, c, h, w, dtype=torch.float16)
    we = torch.randn(nf, cf, hf, wf, dtype=torch.float16)

    h_out = (h + 2 * padding - hf) // stride + 1
    w_out = (w + 2 * padding - wf) // stride + 1
    res_shape = (n, nf, h_out, w_out)
    out = torch.zeros(res_shape, dtype=torch.float32)

    sym = tkl.sym
    N, C, H, W = sym.N, sym.C, sym.H, sym.W
    NF, HF, WF = sym.NF, sym.HF, sym.WF

    H_OUT = (H + 2 * padding - HF) // stride + 1
    W_OUT = (W + 2 * padding - WF) // stride + 1
    SZ_OUT = H_OUT * W_OUT

    K = HF * WF * C
    M = SZ_OUT * N

    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)

    x_mapping = tkw.IndexMapping(
        num_iterators=2,
        inputs={
            N: i // SZ_OUT,
            C: j // (HF * WF),
            H: (i % SZ_OUT) % W_OUT * stride + (j % (HF * WF)) % WF,
            W: (i % SZ_OUT) // W_OUT * stride + (j % (HF * WF)) // WF,
        },
        outputs={M: i, K: j},
    )
    w_mapping = tkw.IndexMapping(
        num_iterators=2,
        inputs={NF: i % NF, C: j // (HF * WF), HF: j % WF, WF: (j % (HF * WF)) // WF},
        outputs={NF: i, K: j},
    )
    out_mapping = tkw.IndexMapping(
        num_iterators=2,
        inputs={M: i, NF: j},
        outputs={
            N: i // SZ_OUT,
            NF: j,
            H_OUT: (i % SZ_OUT) % W_OUT,
            W_OUT: (i % SZ_OUT) // W_OUT,
        },
    )

    # Workgroup tile sizes
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_K = 16
    # Address space (for GPU, shared(1) or global(0))
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
    # Other hyperparameters
    ELEMS_PER_THREAD = tkl.sym.ELEMS_PER_THREAD

    # layout == "nhwc_hwcf"
    x_type = tkl.Memory[N, H, W, C, ADDRESS_SPACE, tkl.f16]
    we_type = tkl.Memory[HF, WF, C, NF, ADDRESS_SPACE, tkl.f16]
    out_type = tkl.Memory[N, H_OUT, W_OUT, NF, GLOBAL_ADDRESS_SPACE, tkl.f32]
    x = torch.permute(x, (0, 2, 3, 1)).contiguous()
    we = torch.permute(we, (2, 3, 1, 0)).contiguous()
    out = torch.permute(out, (0, 2, 3, 1)).contiguous()

    # Expose user-constraints
    constraints: list[tkw.Constraint] = []
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(NF, BLOCK_N, 1)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M)]
    constraints += [tkw.WaveConstraint(NF, BLOCK_N)]
    constraints += [tkw.TilingConstraint(K, BLOCK_K)]

    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(1, 1, 1),
        )
    ]

    @tkw.wave(constraints)
    def conv(
        x: x_type,
        we: we_type,
        out: out_type,
    ):
        c_reg = tkl.Register[M, NF, tkl.f32](0.0)

        @tkw.reduction(K, init_args=[c_reg])
        def repeat(acc: tkl.Register[M, NF, tkl.f32]) -> tkl.Register[M, NF, tkl.f32]:
            a_reg = tkw.read(
                x,
                mapping=x_mapping,
                elements_per_thread=ELEMS_PER_THREAD,
            )
            b_reg = tkw.read(
                we,
                mapping=w_mapping,
                elements_per_thread=ELEMS_PER_THREAD,
            )
            acc = tkw.mma(a_reg, b_reg, acc)
            return acc

        tkw.write(
            repeat, out, mapping=out_mapping, elements_per_thread=ELEMS_PER_THREAD
        )

    with tk.gen.TestLaunchContext(
        {
            N: n,
            C: c,
            W: w,
            H: h,
            NF: nf,
            WF: wf,
            HF: hf,
            BLOCK_M: 16,
            BLOCK_N: 16,
            ELEMS_PER_THREAD: 4,
            ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        },
        canonicalize=True,
    ):
        print(conv(x, we, out).module_op)
        #      CHECK: func @conv
        #  CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index

        # Check we are setting gather start indices to 0
        #      CHECK: %{{.*}} = vector.gather %{{.*}}[%[[C0]], %[[C0]], %[[C0]], %[[C0]]] [%{{.*}}], %{{.*}}, %{{.*}} : memref<2x64x64x640xf16
        #      CHECK: %{{.*}} = vector.gather %{{.*}}[%[[C0]], %[[C0]], %[[C0]], %[[C0]]] [%{{.*}}], %{{.*}}, %{{.*}} : memref<3x3x640x640xf16


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
    def test(a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16]):
        a_reg = tkw.read(a, elements_per_thread=16)
        res = a_reg + a_reg

    with codegen_test_context():
        a = torch.randn(16, 16, dtype=torch.float16)
        print(test(a).module_op)
        # CHECK: func @test
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

    with codegen_test_context():
        a = torch.ones(16, 16, dtype=torch.int32)
        print(test(a).module_op)
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
    def test(a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16]):
        a_reg = tkw.read(a, elements_per_thread=4)
        res = -a_reg
        res = tkw.exp2(res)
        tkw.write(res, a, elements_per_thread=4)

    a = torch.randn(16, 16, dtype=torch.float16)
    with codegen_test_context():
        print(test(a).module_op)
        # CHECK: %[[NEG:.+]] = arith.negf
        # CHECK: math.exp2 %[[NEG]]


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

    config = {"backend": "rocm", "device": "hip", "target": "gfx942"}

    shape = (256, 128)
    a = torch.randn(shape, dtype=torch.float16)
    b = torch.randn(shape, dtype=torch.float16)
    c = torch.zeros((shape[0],), dtype=torch.float16)
    with tk.gen.TestLaunchContext(
        {
            M: shape[0],
            N: shape[1],
            BLOCK_M: 1,
            BLOCK_N: 128,
            ELEMS_PER_THREAD: 2,
            ADDRESS_SPACE: tkl.AddressSpace.GLOBAL_MEMORY.value,
        },
        canonicalize=True,
    ):
        print(test(a, b, c).module_op)
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

    config = {"backend": "rocm", "device": "hip", "target": "gfx942"}

    shape = (256, 128)
    a = torch.randn(shape, dtype=torch.float16)
    b = torch.randn(shape, dtype=torch.float16)
    c = torch.zeros((shape[0],), dtype=torch.float16)
    with tk.gen.TestLaunchContext(
        {
            M: shape[0],
            N: shape[1],
            BLOCK_M: 1,
            BLOCK_N: 128,
            ELEMS_PER_THREAD: 2,
            ADDRESS_SPACE: tkl.AddressSpace.GLOBAL_MEMORY.value,
        },
        canonicalize=True,
    ):
        print(test(a, b, c).module_op)
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

    config = {"backend": "rocm", "device": "hip", "target": "gfx942"}

    shape = (256, 512)
    a = torch.randn(shape, dtype=torch.float16)
    c = torch.zeros((shape[0],), dtype=torch.float16)
    with tk.gen.TestLaunchContext(
        {
            M: shape[0],
            N: shape[1],
            BLOCK_M: 2,
            BLOCK_N: 128,
            ELEMS_PER_THREAD: 2,
            ADDRESS_SPACE: tkl.AddressSpace.GLOBAL_MEMORY.value,
        },
        canonicalize=True,
    ):
        print(test(a, c).module_op)
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

    config = {"backend": "rocm", "device": "hip", "target": "gfx942"}

    shape = (256, 512)
    a = torch.randn(shape, dtype=torch.float16)
    b = torch.randn(shape, dtype=torch.float16)
    c = torch.zeros((shape[0],), dtype=torch.float16)
    with tk.gen.TestLaunchContext(
        {
            M: shape[0],
            N: shape[1],
            BLOCK_M: 1,
            BLOCK_N: 128,
            ELEMS_PER_THREAD: 2,
            ADDRESS_SPACE: tkl.AddressSpace.GLOBAL_MEMORY.value,
        },
        canonicalize=True,
    ):
        print(test(a, b, c).module_op)
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

    config = {"backend": "rocm", "device": "hip", "target": "gfx942"}

    shape = (256, 512)
    a = torch.randn(shape, dtype=torch.float16)
    c = torch.zeros((shape[0],), dtype=torch.float16)
    d = torch.zeros((shape[0],), dtype=torch.float16)
    with tk.gen.TestLaunchContext(
        {
            M: shape[0],
            N: shape[1],
            BLOCK_M: 2,
            BLOCK_N: 128,
            ELEMS_PER_THREAD: 2,
            ADDRESS_SPACE: tkl.AddressSpace.GLOBAL_MEMORY.value,
        },
        canonicalize=True,
    ):
        print(test(a, c).module_op)
        # CHECK-DAG: %[[C0_IDX:.+]] = arith.constant 0 : index
        # CHECK-DAG: %[[C4_IDX:.+]] = arith.constant 4 : index
        # CHECK-DAG: %[[C1_IDX:.+]] = arith.constant 1 : index
        # CHECK-DAG: %[[INIT_MAX:.+]] = arith.constant dense<0xFC00> : vector<1xf16>
        # CHECK-DAG: %[[INIT_SUM:.+]] = arith.constant dense<0.000000e+00> : vector<1xf16>

        # Tile Reduction Loop
        # CHECK: %[[TILED:.+]]:4 = scf.for %[[ITER:.+]] = %[[C0_IDX]] to %[[C4_IDX]] step %[[C1_IDX]]
        # CHECK-SAME: iter_args(%[[ACC0:.+]] = %[[INIT_MAX]], %[[ACC1:.+]] = %[[INIT_SUM]], %[[ACC2:.+]] = %[[INIT_MAX]], %[[ACC3:.+]] = %[[INIT_SUM]])
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
        # CHECK: %[[ACC_MAX_1:.+]] = arith.maximumf %[[ACC2]], %{{.*}}

        # 1st Expanded Local Sum Reduction
        # CHECK: arith.addf {{.*}} : vector<1xf16>
        # 1st Expanded Global Sum Reduction
        # CHECK-COUNT-6: gpu.shuffle  xor
        # 1st Expanded Accumulator Sum Reduction
        # CHECK: %[[ACC_SUM_0:.+]] = arith.addf %[[ACC1]], %{{.*}}

        # 2nd Expanded Local Sum Reduction
        # CHECK: arith.addf {{.*}} : vector<1xf16>
        # 2nd Expanded Global Sum Reduction
        # CHECK-COUNT-6: gpu.shuffle  xor
        # 2nd Expanded Accumulator Sum Reduction
        # CHECK: %[[ACC_SUM_1:.+]] = arith.addf %[[ACC3]], %{{.*}}

        # CHECK: scf.yield %[[ACC_MAX_0]], %[[ACC_SUM_0]], %[[ACC_MAX_1]], %[[ACC_SUM_1]]


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

    config = {"backend": "rocm", "device": "hip", "target": "gfx942"}

    shape = (256, 1024)
    a = torch.randn(shape, dtype=torch.float32)
    c = torch.zeros((shape[0],), dtype=torch.float32)
    with tk.gen.TestLaunchContext(
        {
            M: shape[0],
            N: shape[1],
            BLOCK_M: 1,
            BLOCK_N: 128,
            LOAD_ELEMS_PER_THREAD: 2,
            ADDRESS_SPACE: tkl.AddressSpace.GLOBAL_MEMORY.value,
        },
        canonicalize=True,
        run=False,
        run_config=config,
    ):
        print(test(a, c).module_op)
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
    def test(
        a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[M, ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
    ):
        lhs = tkw.read(a, elements_per_thread=LOAD_ELEMS_PER_THREAD)
        rhs = tkw.read(b, elements_per_thread=1)
        res = lhs + rhs
        tkw.write(res, c, elements_per_thread=STORE_ELEMS_PER_THREAD)

    config = {"backend": "rocm", "device": "hip", "target": "gfx942"}

    shape = (256, 128)
    a = torch.ones(shape, dtype=torch.float16)
    b = torch.ones(shape[0], dtype=torch.float16)
    c = torch.zeros(shape, dtype=torch.float16)
    with tk.gen.TestLaunchContext(
        {
            M: shape[0],
            N: shape[1],
            BLOCK_M: 2,
            BLOCK_N: 128,
            LOAD_ELEMS_PER_THREAD: 2,
            STORE_ELEMS_PER_THREAD: 2,
            ADDRESS_SPACE: tkl.AddressSpace.GLOBAL_MEMORY.value,
        },
        canonicalize=True,
        run=False,
        run_config=config,
    ):
        print(test(a, b, c).module_op)
        # CHECK: func.func @test(%[[ARG0:.+]]: !stream.binding, %[[ARG1:.+]]: !stream.binding, %{{.+}}: !stream.binding)
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
    def test(
        a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
    ):
        a_reg = tkw.read(a, elements_per_thread=4)
        b_reg = tkw.read(b, elements_per_thread=4)
        res = a_reg - b_reg
        res = res * a_reg
        res = res / b_reg
        tkw.write(res, a, elements_per_thread=4)

    a = torch.randn(16, 16, dtype=torch.float16)
    b = torch.randn(16, 16, dtype=torch.float16)
    with codegen_test_context():
        print(test(a, b).module_op)
        # CHECK: %[[SUB:.+]] = arith.subf
        # CHECK: %[[MUL:.+]] = arith.mulf %[[SUB]]
        # CHECK: %[[DIV:.+]] = arith.divf %[[MUL]]


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
    def test(a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16]):
        res = a[0]
        tkw.write(res, a, elements_per_thread=4)

    a = torch.randn(16, 16, dtype=torch.float16)
    with pytest.raises(
        NotImplementedError, match="getitem: Currently only stub implementation"
    ):
        test(a)


# TODO: Add more tests once we have more than a stub implementation.
