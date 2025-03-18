import unittest

import torch
from iree.turbine.kernel.boo import boo_kernel, KernelCacheManager
from iree.turbine.kernel._support.tracing import TestLaunchContext


@boo_kernel
def matmul_relu(lhs, rhs):
    mm = torch.matmul(lhs, rhs)
    return torch.nn.functional.relu(mm)


compile_config = {
    "target_backends": ("llvm-cpu",),
    "flags": ("--iree-llvmcpu-target-cpu=host",),
    "print_mlir": True,
}

run_config = {
    "device": "local-task",
}


def _run():
    with TestLaunchContext(compile_config=compile_config, run_config=run_config):
        output_types = (((2, 1), torch.float32),)

        # this should print the IR only once (for the first compilation)
        for i in range(10):
            gen = torch.manual_seed(seed=i)
            x = torch.randn([2, 3], dtype=torch.float32, generator=gen)
            w = torch.randn([3, 1], dtype=torch.float32, generator=gen)
            y = matmul_relu(x, w, output_types=output_types)
            y_ref = torch.nn.functional.relu(torch.matmul(x, w))
            assert torch.allclose(y, y_ref)

        # this should re-compile for the new dtype
        x, w = x.to(dtype=torch.float16), w.to(dtype=torch.float16)
        output_types = (((2, 1), torch.float16),)
        y = matmul_relu(x, w, output_types=output_types)
        y_ref = torch.nn.functional.relu(torch.matmul(x, w))
        assert torch.allclose(y, y_ref)

        # reset the cache with:
        # KernelCacheManager.reset("boo", "matmul_relu")


class BOOMatmulReluKernelTest(unittest.TestCase):
    def testMatmulRelu(self):
        _run()


if __name__ == "__main__":
    unittest.main()
