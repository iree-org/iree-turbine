import unittest

import torch
from iree.turbine.kernel.boo.alt_launchable import torch_fusion


@torch_fusion
def matmul_relu(lhs, rhs):
    mm = torch.matmul(lhs, rhs)
    return torch.nn.functional.relu(mm)


def _run():
    # this should print the IR only once (for the first compilation)
    for i in range(10):
        gen = torch.manual_seed(seed=i)
        x = torch.randn([2, 3], dtype=torch.float32, generator=gen)
        w = torch.randn([3, 1], dtype=torch.float32, generator=gen)
        _ = matmul_relu(x, w)

    # this should re-compile for the new dtype
    # x, w = x.to(dtype=torch.float16), w.to(dtype=torch.float16)
    # y = matmul_relu(x, w)
    # y_ref = torch.nn.functional.relu(torch.matmul(x, w))
    # assert torch.allclose(y, y_ref)

    # reset the cache with:
    # KernelCacheManager.reset("boo", "matmul_relu")


class BOOMatmulReluKernelTest(unittest.TestCase):
    def testMatmulRelu(self):
        _run()


if __name__ == "__main__":
    unittest.main()
