# RUN: python %s | FileCheck %s

import pytest
from typing import Callable
import shark_turbine.kernel as tk
import shark_turbine.kernel.lang as tkl
import shark_turbine.kernel.wave as tkw
from shark_turbine.kernel.lang.global_symbols import *
from shark_turbine.kernel.wave.utils import run_test
import torch

M = tkl.sym.M
N = tkl.sym.N
K = tkl.sym.K
BLOCK_M = tkl.sym.BLOCK_M
BLOCK_N = tkl.sym.BLOCK_N
BLOCK_K = tkl.sym.BLOCK_K
LOAD_ELEMS_PER_THREAD = tkl.sym.LOAD_ELEM_PER_THREAD
STORE_ELEMS_PER_THREAD = tkl.sym.STORE_ELEM_PER_THREAD
ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
ADDRESS_SPACE_0 = tkl.sym.ADDRESS_SPACE_0


def launch(func: Callable[[], None]) -> Callable[[], None]:
    """
    Run a function as part of the test suite in a test launch context.
    Provides default values for the hyperparameters.
    """
    if __name__ == "__main__":
        with tk.gen.TestLaunchContext(
            {
                M: 16,
                N: 16,
                K: 16,
                BLOCK_M: 16,
                BLOCK_N: 16,
                BLOCK_K: 16,
                ADDRESS_SPACE: tkl.AddressSpace.SHARED_MEMORY.value,
            }
        ):
            func()
    return func


def codegen_test_context():
    return tk.gen.TestLaunchContext(
        {
            M: 16,
            N: 16,
            K: 16,
            BLOCK_M: 16,
            BLOCK_N: 16,
            BLOCK_K: 16,
            ADDRESS_SPACE: tkl.AddressSpace.SHARED_MEMORY.value,
        }
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
        # CHECK: func.func @test(%[[ARG0:.+]]: !stream.binding)
        # CHECK: %[[WG_0:.+]] = stream.dispatch.workgroup.id[0]
        # CHECK: %[[WG_1:.+]] = stream.dispatch.workgroup.id[1]
        # CHECK: %[[T0:.+]] = gpu.thread_id  x
        # CHECK: %[[T1:.+]] = gpu.thread_id  y
        # CHECK: %[[DATA:.+]] = stream.binding.subspan %[[ARG0]]
        # CHECK: %[[C16:.+]] = arith.constant 16 : index
        # CHECK: %[[WG0_OFF:.+]] = arith.muli %[[WG_0]], %[[C16]]
        # CHECK: %[[C4:.+]] = arith.constant 4 : index
        # CHECK: %[[T0_OFF:.+]] = arith.divsi %[[T0]], %[[C4]]
        # CHECK: %[[IDX_X:.+]] = arith.addi %[[T0_OFF]], %[[WG0_OFF]]
        # CHECK: %[[C16_0:.+]] = arith.constant 16 : index
        # CHECK: %[[T1_OFF:.+]] = arith.muli %[[T1]], %[[C16_0]] : index
        # CHECK: %[[C16_1:.+]] = arith.constant 16 : index
        # CHECK: %[[WG1_OFF:.+]] = arith.muli %[[WG_1]], %[[C16_1]]
        # CHECK: %[[IDX_Y:.+]] = arith.addi %[[WG1_OFF]], %[[T1_OFF]]
        # CHECK: vector.load %[[DATA]][%[[IDX_X]], %[[IDX_Y]]] : memref<16x16xf16, strided<[16, 1], offset: ?>>, vector<16xf16>


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
        # CHECK: func.func @test(%[[ARG0:.+]]: !stream.binding)
        # CHECK: %[[WG_0:.+]] = stream.dispatch.workgroup.id[0]
        # CHECK: %[[WG_1:.+]] = stream.dispatch.workgroup.id[1]
        # CHECK: %[[T0:.+]] = gpu.thread_id  x
        # CHECK: %[[T1:.+]] = gpu.thread_id  y
        # CHECK: %[[DATA:.+]] = stream.binding.subspan %[[ARG0]]
        # CHECK: %[[C16:.+]] = arith.constant 16 : index
        # CHECK: %[[WG0_OFF:.+]] = arith.muli %[[WG_0]], %[[C16]]
        # CHECK: %[[C4:.+]] = arith.constant 4 : index
        # CHECK: %[[T0_OFF:.+]] = arith.divsi %[[T0]], %[[C4]]
        # CHECK: %[[IDX_X:.+]] = arith.addi %[[T0_OFF]], %[[WG0_OFF]]
        # CHECK: %[[C16_0:.+]] = arith.constant 16 : index
        # CHECK: %[[T1_OFF:.+]] = arith.muli %[[T1]], %[[C16_0]] : index
        # CHECK: %[[C16_1:.+]] = arith.constant 16 : index
        # CHECK: %[[WG1_OFF:.+]] = arith.muli %[[WG_1]], %[[C16_1]]
        # CHECK: %[[IDX_Y:.+]] = arith.addi %[[WG1_OFF]], %[[T1_OFF]]
        # CHECK: %[[OFF:.+]] = arith.constant dense<[0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240]> : vector<16xindex>
        # CHECK: %[[MASK:.+]] = vector.constant_mask [16] : vector<16xi1>
        # CHECK: %[[PASSTHRU:.+]] = vector.splat %{{.*}} : vector<16xf16>
        # CHECK: %[[RES:.+]] = vector.gather %[[DATA]][%[[IDX_X]], %[[IDX_Y]]] [%[[OFF]]], %[[MASK]], %[[PASSTHRU]] : memref<16x16xf16, strided<[16, 1], offset: ?>>, vector<16xindex>, vector<16xi1>, vector<16xf16> into vector<16xf16>


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

    with codegen_test_context():
        a = torch.randn(16, 16, dtype=torch.float16)
        b = torch.zeros(16, 16, dtype=torch.float16)
        print(test(a, b).module_op)
        # CHECK: func.func @test(%[[ARG0:.+]]: !stream.binding, %[[ARG1:.+]]: !stream.binding)
        # CHECK: %[[WG_0:.+]] = stream.dispatch.workgroup.id[0]
        # CHECK: %[[WG_1:.+]] = stream.dispatch.workgroup.id[1]
        # CHECK: %[[T0:.+]] = gpu.thread_id  x
        # CHECK: %[[T1:.+]] = gpu.thread_id  y

        # CHECK: %[[RES:.+]] = vector.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<16x16xf16, strided<[16, 1], offset: ?>>, vector<16xf16>

        # CHECK: %[[OUT:.+]] = stream.binding.subspan %[[ARG1]]

        # CHECK: %[[C16:.+]] = arith.constant 16 : index
        # CHECK: %[[WG0_OFF:.+]] = arith.muli %[[WG_0]], %[[C16]]
        # CHECK: %[[C4:.+]] = arith.constant 4 : index
        # CHECK: %[[T0_OFF:.+]] = arith.divsi %[[T0]], %[[C4]]
        # CHECK: %[[IDX_X:.+]] = arith.addi %[[T0_OFF]], %[[WG0_OFF]]
        # CHECK: %[[C16_0:.+]] = arith.constant 16 : index
        # CHECK: %[[T1_OFF:.+]] = arith.muli %[[T1]], %[[C16_0]] : index
        # CHECK: %[[C16_1:.+]] = arith.constant 16 : index
        # CHECK: %[[WG1_OFF:.+]] = arith.muli %[[WG_1]], %[[C16_1]]
        # CHECK: %[[IDX_Y:.+]] = arith.addi %[[WG1_OFF]], %[[T1_OFF]]
        # CHECK: vector.store %[[RES]], %[[OUT]][%[[IDX_X]], %[[IDX_Y]]] : memref<16x16xf16, strided<[16, 1], offset: ?>>, vector<16xf16>


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

    with codegen_test_context():
        a = torch.randn(16, 16, dtype=torch.float16)
        b = torch.zeros(16, 16, dtype=torch.float16)
        print(test(a, b).module_op)
        # CHECK: func.func @test(%[[ARG0:.+]]: !stream.binding, %[[ARG1:.+]]: !stream.binding)
        # CHECK: %[[WG_0:.+]] = stream.dispatch.workgroup.id[0]
        # CHECK: %[[WG_1:.+]] = stream.dispatch.workgroup.id[1]
        # CHECK: %[[T0:.+]] = gpu.thread_id  x
        # CHECK: %[[T1:.+]] = gpu.thread_id  y

        # CHECK: %[[RES:.+]] = vector.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<16x16xf16, strided<[16, 1], offset: ?>>, vector<16xf16>

        # CHECK: %[[OUT:.+]] = stream.binding.subspan %[[ARG1]]

        # CHECK: %[[C16:.+]] = arith.constant 16 : index
        # CHECK: %[[T1_OFF:.+]] = arith.muli %[[T1]], %[[C16]] : index
        # CHECK: %[[C16_0:.+]] = arith.constant 16 : index
        # CHECK: %[[WG1_OFF:.+]] = arith.muli %[[WG_1]], %[[C16_0]] : index
        # CHECK: %[[IDX_Y:.+]] = arith.addi %[[WG1_OFF]], %[[T1_OFF]] : index
        # CHECK: %[[C16_1:.+]] = arith.constant 16 : index
        # CHECK: %[[WG0_OFF:.+]] = arith.muli %[[WG_0]], %[[C16_1]] : index
        # CHECK: %[[C4:.+]] = arith.constant 4 : index
        # CHECK: %[[T0_OFF:.+]] = arith.divsi %[[T0]], %[[C4]] : index
        # CHECK: %[[IDX_X:.+]] = arith.addi %[[T0_OFF]], %[[WG0_OFF]] : index
        # CHECK: %[[OFF:.+]] = arith.constant dense<[0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240]> : vector<16xindex>
        # CHECK: %[[MASK:.+]] = vector.constant_mask [16] : vector<16xi1>
        # CHECK: vector.scatter %[[OUT]][%[[IDX_Y]], %[[IDX_X]]] [%[[OFF]]], %[[MASK]], %[[RES]] : memref<16x16xf16, strided<[16, 1], offset: ?>>, vector<16xindex>, vector<16xi1>, vector<16xf16>


@run_test
def test_mma():
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
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
    def mma(
        a: tkl.Memory[M, K, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[N, K, ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[M, N, ADDRESS_SPACE_0, tkl.f32],
    ):
        c_reg = tkl.Register[M, N, tkl.f32](0.0)
        a_reg = tkw.read(a, elements_per_thread=LOAD_ELEMS_PER_THREAD)
        b_reg = tkw.read(b, elements_per_thread=LOAD_ELEMS_PER_THREAD)
        acc = tkw.mma(a_reg, b_reg, c_reg)
        tkw.write(acc, c, elements_per_thread=STORE_ELEMS_PER_THREAD)

    with tk.gen.TestLaunchContext(
        {
            M: 64,
            N: 128,
            K: 16,
            BLOCK_M: 32,
            BLOCK_N: 32,
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
        print(mma(a, b, c).module_op)

        # CHECK:          func.func @mma(%[[ARG0:[a-zA-Z0-9_]+]]: !stream.binding, %[[ARG1:[a-zA-Z0-9_]+]]: !stream.binding,
        # CHECK-SAME:       %[[ARG2:[a-zA-Z0-9_]+]]: !stream.binding
        # CHECK-DAG:        %[[C3:.+]] = arith.constant 3 : index
        # CHECK-DAG:        %[[C2:.+]] = arith.constant 2 : index
        # CHECK-DAG:        %[[C64:.+]] = arith.constant 64 : index
        # CHECK-DAG:        %[[C16:.+]] = arith.constant 16 : index
        # CHECK-DAG:        %[[C4:.+]] = arith.constant 4 : index
        # CHECK-DAG:        %[[C1:.+]] = arith.constant 1 : index
        # CHECK-DAG:        %[[C32:.+]] = arith.constant 32 : index
        # CHECK-DAG:        %[[C0:.+]] = arith.constant 0 : index
        # CHECK-DAG:        %[[CST:.+]] = arith.constant dense<0.000000e+00> : vector<4xf32>
        # CHECK:            %[[WORKGROUP_ID_0:.+]] = stream.dispatch.workgroup.id[0] : index
        # CHECK:            %[[WORKGROUP_ID_1:.+]] = stream.dispatch.workgroup.id[1] : index
        # CHECK-DAG:        %[[THREAD_ID_X:.+]] = gpu.thread_id  x
        # CHECK-DAG:        %[[THREAD_ID_Y:.+]] = gpu.thread_id  y
        # CHECK-DAG:        %[[THREAD_ID_Z:.+]] = gpu.thread_id  z
        # CHECK:            %[[D0:.+]] = stream.binding.subspan %[[ARG0]][%[[C0]]] : !stream.binding -> memref<64x16xf16,
        # CHECK-SAME:         strided<[16, 1], offset: ?>>
        # CHECK:            %[[D1:.+]] = arith.muli %[[WORKGROUP_ID_0]], %[[C32]] : index
        # CHECK:            %[[D2:.+]] = arith.divsi %[[THREAD_ID_X]], %[[C4]] : index
        # CHECK:            %[[D3:.+]] = arith.addi %[[D2]], %[[D1]] : index
        # CHECK:            %[[D4:.+]] = vector.load %[[D0]][%[[D3]], %[[C0]]] : memref<64x16xf16, strided<[16, 1], offset:
        # CHECK-SAME:         ?>>, vector<4xf16>
        # CHECK:            %[[ALLOC:.+]] = memref.alloc() : memref<32x16xf16, #[[GPU:.+]].address_space<workgroup>>
        # CHECK:            vector.store %[[D4]], %[[ALLOC]][%[[D3]], %[[C0]]] : memref<32x16xf16,
        # CHECK-SAME:         #[[GPU]].address_space<workgroup>>, vector<4xf16>
        # CHECK:            %[[D5:.+]] = vector.load %[[ALLOC]][%[[D3]], %[[C0]]] : memref<32x16xf16,
        # CHECK-SAME:         #[[GPU]].address_space<workgroup>>, vector<4xf16>
        # CHECK:            %[[D6:.+]] = stream.binding.subspan %[[ARG1]][%[[C0]]] : !stream.binding -> memref<128x16xf16,
        # CHECK-SAME:         strided<[16, 1], offset: ?>>
        # CHECK:            %[[D7:.+]] = arith.muli %[[THREAD_ID_Y]], %[[C16]] : index
        # CHECK:            %[[D8:.+]] = arith.muli %[[WORKGROUP_ID_1]], %[[C32]] : index
        # CHECK:            %[[D9:.+]] = arith.addi %[[D8]], %[[D7]] : index
        # CHECK:            %[[D10:.+]] = vector.load %[[D6]][%[[D9]], %[[C0]]] : memref<128x16xf16, strided<[16, 1], offset:
        # CHECK-SAME:         ?>>, vector<4xf16>
        # CHECK:            %[[ALLOC_0:.+]] = memref.alloc() : memref<32x16xf16, #[[GPU]].address_space<workgroup>>
        # CHECK:            vector.store %[[D10]], %[[ALLOC_0]][%[[D9]], %[[C0]]] : memref<32x16xf16,
        # CHECK-SAME:         #[[GPU]].address_space<workgroup>>, vector<4xf16>
        # CHECK:            %[[D11:.+]] = vector.load %[[ALLOC_0]][%[[D9]], %[[C0]]] : memref<32x16xf16,
        # CHECK-SAME:         #[[GPU]].address_space<workgroup>>, vector<4xf16>
        # CHECK:            %[[D12:.+]] = amdgpu.mfma %[[D5]] * %[[D11]] + %[[CST]] {blocks = 1 : i32, k = 16 : i32, m = 16 :
        # CHECK-SAME:         i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        # CHECK:            %[[D13:.+]] = vector.extract_strided_slice %[[D12]] {offsets = [0], sizes = [1], strides = [1]} :
        # CHECK-SAME:         vector<4xf32> to vector<1xf32>
        # CHECK:            %[[D14:.+]] = stream.binding.subspan %[[ARG2]][%[[C0]]] : !stream.binding -> memref<64x128xf32,
        # CHECK-SAME:         strided<[128, 1], offset: ?>>
        # CHECK:            %[[D15:.+]] = arith.divsi %[[THREAD_ID_X]], %[[C16]] : index
        # CHECK:            %[[D16:.+]] = arith.muli %[[D15]], %[[C4]] : index
        # CHECK:            %[[D17:.+]] = arith.muli %[[THREAD_ID_Y]], %[[C32]] : index
        # CHECK:            %[[D18:.+]] = arith.muli %[[THREAD_ID_Z]], %[[C64]] : index
        # CHECK:            %[[D19:.+]] = arith.addi %[[D2]], %[[D18]] : index
        # CHECK:            %[[D20:.+]] = arith.addi %[[D19]], %[[D1]] : index
        # CHECK:            %[[D21:.+]] = arith.addi %[[D20]], %[[D17]] : index
        # CHECK:            %[[D22:.+]] = arith.addi %[[D21]], %[[D16]] : index
        # CHECK:            %[[D23:.+]] = arith.remsi %[[THREAD_ID_X]], %[[C16]] : index
        # CHECK:            %[[D24:.+]] = arith.addi %[[D23]], %[[D8]] : index
        # CHECK:            %[[D25:.+]] = arith.addi %[[D24]], %[[D7]] : index
        # CHECK:            vector.store %[[D13]], %[[D14]][%[[D22]], %[[D25]]] : memref<64x128xf32, strided<[128, 1], offset:
        # CHECK-SAME:         ?>>, vector<1xf32>
        # CHECK:            %[[D26:.+]] = vector.extract_strided_slice %[[D12]] {offsets = [1], sizes = [1], strides = [1]} :
        # CHECK-SAME:         vector<4xf32> to vector<1xf32>
        # CHECK:            %[[D27:.+]] = arith.addi %[[D22]], %[[C1]] : index
        # CHECK:            vector.store %[[D26]], %[[D14]][%[[D27]], %[[D25]]] : memref<64x128xf32, strided<[128, 1], offset:
        # CHECK-SAME:         ?>>, vector<1xf32>
        # CHECK:            %[[D28:.+]] = vector.extract_strided_slice %[[D12]] {offsets = [2], sizes = [1], strides = [1]} :
        # CHECK-SAME:         vector<4xf32> to vector<1xf32>
        # CHECK:            %[[D29:.+]] = arith.addi %[[D22]], %[[C2]] : index
        # CHECK:            vector.store %[[D28]], %[[D14]][%[[D29]], %[[D25]]] : memref<64x128xf32, strided<[128, 1], offset:
        # CHECK-SAME:         ?>>, vector<1xf32>
        # CHECK:            %[[D30:.+]] = vector.extract_strided_slice %[[D12]] {offsets = [3], sizes = [1], strides = [1]} :
        # CHECK-SAME:         vector<4xf32> to vector<1xf32>
        # CHECK:            %[[D31:.+]] = arith.addi %[[D22]], %[[C3]] : index
        # CHECK:            vector.store %[[D30]], %[[D14]][%[[D31]], %[[D25]]] : memref<64x128xf32, strided<[128, 1], offset:
        # CHECK-SAME:         ?>>, vector<1xf32>


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
        # CHECK-DAG:        %[[C4:.+]] = arith.constant 4 : index
        # CHECK-DAG:        %[[C1:.+]] = arith.constant 1 : index
        # CHECK-DAG:        %[[C32:.+]] = arith.constant 32 : index
        # CHECK-DAG:        %[[C16:.+]] = arith.constant 16 : index
        # CHECK-DAG:        %[[C64:.+]] = arith.constant 64 : index
        # CHECK-DAG:        %[[C0:.+]] = arith.constant 0 : index
        # CHECK-DAG:        %[[CST:.+]] = arith.constant dense<0.000000e+00> : vector<4xf32>
        # CHECK:            %[[WORKGROUP_ID_0:.+]] = stream.dispatch.workgroup.id[0] : index
        # CHECK:            %[[WORKGROUP_ID_1:.+]] = stream.dispatch.workgroup.id[1] : index
        # CHECK-DAG:        %[[THREAD_ID_X:.+]] = gpu.thread_id  x
        # CHECK-DAG:        %[[THREAD_ID_Y:.+]] = gpu.thread_id  y
        # CHECK-DAG:        %[[THREAD_ID_Z:.+]] = gpu.thread_id  z
        # CHECK:            %[[ALLOC:.+]] = memref.alloc() : memref<32x64xf16, #[[GPU:.+]].address_space<workgroup>>
        # CHECK:            %[[ALLOC_0:.+]] = memref.alloc() : memref<32x64xf16, #[[GPU]].address_space<workgroup>>
        # CHECK:            %[[D0:.+]] = scf.for %[[ARG3:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C64]] step %[[C16]]
        # CHECK-SAME:         iter_args(%[[ARG4:[a-zA-Z0-9_]+]] = %[[CST]]) -> (vector<4xf32>) {
        # CHECK:              %[[D24:.+]] = stream.binding.subspan %[[ARG0]][%[[C0]]] : !stream.binding -> memref<64x64xf16,
        # CHECK-SAME:           strided<[64, 1], offset: ?>>
        # CHECK:              %[[D25:.+]] = stream.binding.subspan %[[ARG1]][%[[C0]]] : !stream.binding -> memref<128x64xf16,
        # CHECK-SAME:           strided<[64, 1], offset: ?>>
        # CHECK:              %[[D26:.+]] = arith.muli %[[WORKGROUP_ID_0]], %[[C32]] : index
        # CHECK:              %[[D27:.+]] = arith.divsi %[[THREAD_ID_X]], %[[C4]] : index
        # CHECK:              %[[D28:.+]] = arith.addi %[[D27]], %[[D26]] : index
        # CHECK:              %[[D29:.+]] = arith.muli %[[ARG3]], %[[C16]] : index
        # CHECK:              %[[D30:.+]] = vector.load %[[D24]][%[[D28]], %[[D29]]] : memref<64x64xf16, strided<[64, 1],
        # CHECK-SAME:           offset: ?>>, vector<4xf16>
        # CHECK:              vector.store %[[D30]], %[[ALLOC]][%[[D28]], %[[D29]]] : memref<32x64xf16,
        # CHECK-SAME:           #[[GPU]].address_space<workgroup>>, vector<4xf16>
        # CHECK:              %[[D31:.+]] = vector.load %[[ALLOC]][%[[D28]], %[[D29]]] : memref<32x64xf16,
        # CHECK-SAME:           #[[GPU]].address_space<workgroup>>, vector<4xf16>
        # CHECK:              %[[D32:.+]] = arith.muli %[[THREAD_ID_Y]], %[[C16]] : index
        # CHECK:              %[[D33:.+]] = arith.muli %[[WORKGROUP_ID_1]], %[[C32]] : index
        # CHECK:              %[[D34:.+]] = arith.addi %[[D33]], %[[D32]] : index
        # CHECK:              %[[D35:.+]] = vector.load %[[D25]][%[[D34]], %[[D29]]] : memref<128x64xf16, strided<[64, 1],
        # CHECK-SAME:           offset: ?>>, vector<4xf16>
        # CHECK:              vector.store %[[D35]], %[[ALLOC_0]][%[[D34]], %[[D29]]] : memref<32x64xf16,
        # CHECK-SAME:           #[[GPU]].address_space<workgroup>>, vector<4xf16>
        # CHECK:              %[[D36:.+]] = vector.load %[[ALLOC_0]][%[[D34]], %[[D29]]] : memref<32x64xf16,
        # CHECK-SAME:           #[[GPU]].address_space<workgroup>>, vector<4xf16>
        # CHECK:              %[[D37:.+]] = amdgpu.mfma %[[D31]] * %[[D36]] + %[[ARG4]] {blocks = 1 : i32, k = 16 : i32, m = 16
        # CHECK-SAME:           : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        # CHECK:              scf.yield %[[D37]] : vector<4xf32>
        # CHECK:            }
        # CHECK:            %[[D1:.+]] = vector.extract_strided_slice %[[D0]] {offsets = [0], sizes = [1], strides = [1]} :
        # CHECK-SAME:         vector<4xf32> to vector<1xf32>
        # CHECK:            %[[D2:.+]] = stream.binding.subspan %[[ARG2]][%[[C0]]] : !stream.binding -> memref<64x128xf32,
        # CHECK-SAME:         strided<[128, 1], offset: ?>>
        # CHECK:            %[[D3:.+]] = arith.divsi %[[THREAD_ID_X]], %[[C16]] : index
        # CHECK:            %[[D4:.+]] = arith.muli %[[D3]], %[[C4]] : index
        # CHECK:            %[[D5:.+]] = arith.muli %[[THREAD_ID_Y]], %[[C32]] : index
        # CHECK:            %[[D6:.+]] = arith.muli %[[WORKGROUP_ID_0]], %[[C32]] : index
        # CHECK:            %[[D7:.+]] = arith.muli %[[THREAD_ID_Z]], %[[C64]] : index
        # CHECK:            %[[D8:.+]] = arith.divsi %[[THREAD_ID_X]], %[[C4]] : index
        # CHECK:            %[[D9:.+]] = arith.addi %[[D8]], %[[D7]] : index
        # CHECK:            %[[D10:.+]] = arith.addi %[[D9]], %[[D6]] : index
        # CHECK:            %[[D11:.+]] = arith.addi %[[D10]], %[[D5]] : index
        # CHECK:            %[[D12:.+]] = arith.addi %[[D11]], %[[D4]] : index
        # CHECK:            %[[D13:.+]] = arith.muli %[[THREAD_ID_Y]], %[[C16]] : index
        # CHECK:            %[[D14:.+]] = arith.muli %[[WORKGROUP_ID_1]], %[[C32]] : index
        # CHECK:            %[[D15:.+]] = arith.remsi %[[THREAD_ID_X]], %[[C16]] : index
        # CHECK:            %[[D16:.+]] = arith.addi %[[D15]], %[[D14]] : index
        # CHECK:            %[[D17:.+]] = arith.addi %[[D16]], %[[D13]] : index
        # CHECK:            vector.store %[[D1]], %[[D2]][%[[D12]], %[[D17]]] : memref<64x128xf32, strided<[128, 1], offset:
        # CHECK-SAME:         ?>>, vector<1xf32>
        # CHECK:            %[[D18:.+]] = vector.extract_strided_slice %[[D0]] {offsets = [1], sizes = [1], strides = [1]} :
        # CHECK-SAME:         vector<4xf32> to vector<1xf32>
        # CHECK:            %[[D19:.+]] = arith.addi %[[D12]], %[[C1]] : index
        # CHECK:            vector.store %[[D18]], %[[D2]][%[[D19]], %[[D17]]] : memref<64x128xf32, strided<[128, 1], offset:
        # CHECK-SAME:         ?>>, vector<1xf32>
        # CHECK:            %[[D20:.+]] = vector.extract_strided_slice %[[D0]] {offsets = [2], sizes = [1], strides = [1]} :
        # CHECK-SAME:         vector<4xf32> to vector<1xf32>
        # CHECK:            %[[D21:.+]] = arith.addi %[[D12]], %[[C2]] : index
        # CHECK:            vector.store %[[D20]], %[[D2]][%[[D21]], %[[D17]]] : memref<64x128xf32, strided<[128, 1], offset:
        # CHECK-SAME:         ?>>, vector<1xf32>
        # CHECK:            %[[D22:.+]] = vector.extract_strided_slice %[[D0]] {offsets = [3], sizes = [1], strides = [1]} :
        # CHECK-SAME:         vector<4xf32> to vector<1xf32>
        # CHECK:            %[[D23:.+]] = arith.addi %[[D12]], %[[C3]] : index
        # CHECK:            vector.store %[[D22]], %[[D2]][%[[D23]], %[[D17]]] : memref<64x128xf32, strided<[128, 1], offset:
        # CHECK-SAME:         ?>>, vector<1xf32>


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
def test_reduce():
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


@launch
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
