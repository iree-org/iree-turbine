# RUN: python %s | FileCheck %s

import pytest
from typing import Callable
import shark_turbine.kernel as tk
import shark_turbine.kernel.lang as tkl
import shark_turbine.kernel.wave as tkw
from shark_turbine.kernel.lang.global_symbols import *
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


def run(func: Callable[[], None]) -> Callable[[], None]:
    """Run a function as part of the test suite."""
    if __name__ == "__main__":
        func()
        # Print a separator between tests
        print("-----")
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


@run
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


@run
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


@run
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


@run
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


@run
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

        # CHECK: func.func @mma(%[[ARG0:.+]]: !stream.binding, %[[ARG1:.+]]: !stream.binding, %[[ARG2:.+]]: !stream.binding) {
        # CHECK-DAG: %[[C16:.+]] = arith.constant 16 : index
        # CHECK-DAG: %[[C4:.+]] = arith.constant 4 : index
        # CHECK-DAG: %[[C32:.+]] = arith.constant 32 : index
        # CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
        # CHECK-DAG: %[[ACC:.+]] = arith.constant dense<0.000000e+00> : vector<4xf32>
        # CHECK: %[[WG0:.+]] = stream.dispatch.workgroup.id[0] : index
        # CHECK: %[[WG1:.+]] = stream.dispatch.workgroup.id[1] : index
        # CHECK: %[[TX:.+]] = gpu.thread_id  x
        # CHECK: %[[TY:.+]] = gpu.thread_id  y
        # CHECK: %[[R0:.+]] = stream.binding.subspan %[[ARG0]][%[[C0]]] : !stream.binding -> memref<64x16xf16, strided<[16, 1], offset: ?>>
        # CHECK: %[[R1:.+]] = arith.muli %[[WG0]], %[[C32]] : index
        # CHECK: %[[R2:.+]] = arith.divsi %[[TX]], %[[C4]] : index
        # CHECK: %[[R3:.+]] = arith.addi %[[R2]], %[[R1]] : index
        # CHECK: %[[R4:.+]] = vector.load %0[%[[R3]], %[[C0]]] : memref<64x16xf16, strided<[16, 1], offset: ?>>, vector<4xf16>
        # CHECK: %[[ALLOC:.+]] = memref.alloc() : memref<32x16xf16, #gpu.address_space<workgroup>>
        # CHECK: %[[R5:.+]] = arith.muli %[[WG0]], %[[C32]] : index
        # CHECK: %[[R6:.+]] = arith.divsi %[[TX]], %[[C4]] : index
        # CHECK: %[[R7:.+]] = arith.addi %[[R6]], %[[R5]] : index
        # CHECK: vector.store %4, %[[ALLOC]][%[[R7]], %[[C0]]] : memref<32x16xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        # CHECK: %[[R8:.+]] = arith.muli %[[WG0]], %[[C32]] : index
        # CHECK: %[[R9:.+]] = arith.divsi %[[TX]], %[[C4]] : index
        # CHECK: %[[R10:.+]] = arith.addi %[[R9]], %[[R8]] : index
        # CHECK: %[[R11:.+]] = vector.load %[[ALLOC]][%[[R10]], %[[C0]]] : memref<32x16xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        # CHECK: %[[R12:.+]] = stream.binding.subspan %[[ARG1]][%[[C0]]] : !stream.binding -> memref<128x16xf16, strided<[16, 1], offset: ?>>
        # CHECK: %[[R13:.+]] = arith.muli %[[TY]], %[[C16]] : index
        # CHECK: %[[R14:.+]] = arith.muli %[[WG1]], %[[C32]] : index
        # CHECK: %[[R15:.+]] = arith.addi %[[R14]], %[[R13]] : index
        # CHECK: %[[R16:.+]] = vector.load %[[R12]][%[[R15]], %[[C0]]] : memref<128x16xf16, strided<[16, 1], offset: ?>>, vector<4xf16>
        # CHECK: %[[ALLOC_0:.+]] = memref.alloc() : memref<32x16xf16, #gpu.address_space<workgroup>>
        # CHECK: %[[R17:.+]] = arith.muli %[[TY]], %[[C16]] : index
        # CHECK: %[[R18:.+]] = arith.muli %[[WG1]], %[[C32]] : index
        # CHECK: %[[R19:.+]] = arith.addi %[[R18]], %[[R17]] : index
        # CHECK: vector.store %16, %[[ALLOC_0]][%[[R19]], %[[C0]]] : memref<32x16xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        # CHECK: %[[R20:.+]] = arith.muli %[[TY]], %[[C16]] : index
        # CHECK: %[[R21:.+]] = arith.muli %[[WG1]], %[[C32]] : index
        # CHECK: %[[R22:.+]] = arith.addi %[[R21]], %[[R20]] : index
        # CHECK: %[[R23:.+]] = vector.load %[[ALLOC_0]][%[[R22]], %[[C0]]] : memref<32x16xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        # CHECK: %[[R24:.+]] = amdgpu.mfma %[[R11]] * %[[R23]] + %[[ACC]] {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        # CHECK: %[[R25:.+]] = stream.binding.subspan %[[ARG2]][%[[C0]]] : !stream.binding -> memref<64x128xf32, strided<[128, 1], offset: ?>>
        # CHECK: %[[R26:.+]] = arith.muli %[[WG0]], %[[C32]] : index
        # CHECK: %[[R27:.+]] = arith.divsi %[[TX]], %[[C4]] : index
        # CHECK: %[[R28:.+]] = arith.addi %[[R27]], %[[R26]] : index
        # CHECK: %[[R29:.+]] = arith.muli %[[TY]], %[[C16]] : index
        # CHECK: %[[R30:.+]] = arith.muli %[[WG1]], %[[C32]] : index
        # CHECK: %[[R31:.+]] = arith.addi %[[R30]], %[[R29]] : index
        # CHECK: vector.store %[[R24]], %[[R25]][%[[R28]], %[[R31]]] : memref<64x128xf32, strided<[128, 1], offset: ?>>, vector<4xf32>


@run
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


@run
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


@launch
@pytest.mark.skip(reason="neg: Currently only stub implementation")
def test_neg():
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
        res = -a
        tkw.write(res, a, elements_per_thread=4)

    a = torch.randn(16, 16, dtype=torch.float16)
    with pytest.raises(
        NotImplementedError, match="neg: Currently only stub implementation"
    ):
        test(a)


@launch
@pytest.mark.skip(reason="sub: Currently only stub implementation")
def test_sub():
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
        res = a - a
        tkw.write(res, a, elements_per_thread=4)

    a = torch.randn(16, 16, dtype=torch.float16)
    with pytest.raises(
        NotImplementedError, match="sub: Currently only stub implementation"
    ):
        test(a)


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
