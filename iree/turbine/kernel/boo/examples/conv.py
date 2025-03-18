import unittest

import torch
from iree.turbine.kernel.boo import KernelCacheManager, BOOLaunchable
from iree.turbine.kernel.boo.conv_exports import ConvSignature
from iree.turbine.kernel._support.tracing import TestLaunchContext


def _get_test_signature(
    N: int, C: int, F: int, H: int, W: int, KH: int, KW: int, dtype=torch.float32
):
    return ConvSignature(
        input_shape=[N, H, W, C],
        kernel_shape=[F, KH, KW, C],
        dtype=dtype,
        shared_layout="NHWC",
    )


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
        N = 2
        C = 3
        F = 4
        H = 16
        W = 16
        KH = 3
        KW = 3
        dtype = torch.float32
        signature = _get_test_signature(N, C, F, H, W, KH, KW)
        conv = signature.get_nn_module()
        boo_conv = BOOLaunchable("default_conv_2d_fwd", conv)
        output_types = ((signature.output_shape, dtype),)

        # this should print the IR only once (for the first compilation)
        for i in range(10):
            args = signature.get_sample_conv_args(seed=i)
            y = boo_conv(*args, output_types=output_types)
            y_ref = conv(*args)
            assert torch.allclose(y, y_ref)

        # reset the cache with:
        KernelCacheManager.reset("boo", "default_conv_2d_fwd")


class BOOMatmulReluKernelTest(unittest.TestCase):
    def testMatmulRelu(self):
        _run()


if __name__ == "__main__":
    unittest.main()
