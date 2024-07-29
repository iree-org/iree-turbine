# RUN: python %s

import pytest
from typing import Callable
import shark_turbine.kernel as tk
import shark_turbine.kernel.lang as tkl
import shark_turbine.kernel.wave as tkw
import torch

M = tkl.sym.M
N = tkl.sym.N
K = tkl.sym.K
BLOCK_M = tkl.sym.BLOCK_M
BLOCK_N = tkl.sym.BLOCK_N
BLOCK_K = tkl.sym.BLOCK_K
ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE


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


@launch
def test_read():
    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(threads_per_wave=64, waves_per_block=(1, 1, 1))
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]

    @tkw.wave(constraints)
    def test(a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16]):
        tkw.read(a)

    a = torch.randn(16, 16, dtype=torch.float16)
    with pytest.raises(
        NotImplementedError, match="Read: Currently only stub implementation"
    ):
        test(a)


@launch
def test_add():
    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(threads_per_wave=64, waves_per_block=(1, 1, 1))
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]

    @tkw.wave(constraints)
    def test(a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16]):
        res = a + a
        tkw.write(res, a, elements_per_thread=4)

    a = torch.randn(16, 16, dtype=torch.float16)
    with pytest.raises(
        NotImplementedError, match="add: Currently only stub implementation"
    ):
        test(a)


@launch
def test_neg():
    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(threads_per_wave=64, waves_per_block=(1, 1, 1))
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]

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
def test_sub():
    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(threads_per_wave=64, waves_per_block=(1, 1, 1))
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]

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
def test_get_item():
    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(threads_per_wave=64, waves_per_block=(1, 1, 1))
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]

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
