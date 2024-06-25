# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
from contextlib import contextmanager
import sys
from time import perf_counter

import torch
import torch.nn as nn

import shark_turbine.aot as aot

from shark_turbine.runtime import (
    Launchable,
)


@contextmanager
def report_time(banner):
    t1 = perf_counter()
    yield
    t2 = perf_counter()
    print(f"{banner}: {(t2 - t1) * 1000}ms")


class TestLinearModel(nn.Module):
    def forward(self, input, weight, bias):
        result = nn.functional.linear(input, weight, bias)
        scale = 1 / 256.0
        result = result * scale
        return nn.functional.tanh(result)


def test_sequential(nreps, device, dtype):
    print("*** run test_sequential:")
    print("------------------------")
    print("*** compiling model")
    torch.manual_seed(42)
    initial_input = torch.rand([512, 512], dtype=dtype)
    weight = torch.rand([512, 512], dtype=dtype)
    bias = torch.rand([512], dtype=dtype)
    eo = aot.export(TestLinearModel(), args=(initial_input, weight, bias))
    launcher = Launchable.jit_compile(str(eo.mlir_module))
    launcher.preload(device)
    print("*** preloaded")

    # Compute reference.
    ref_model = TestLinearModel()
    ref_input = initial_input.to(device)
    ref_weight = weight.to(device)
    ref_bias = bias.to(device)
    print("*** moved ref inputs to device")
    with report_time("reference total time"):
        for i in range(nreps):
            ref_output = ref_model(ref_input, ref_weight, ref_bias)
            ref_input = ref_output
        ref_output = ref_output.to(device="cpu").clone()
    print(ref_output)

    # Compute test.
    test_input = initial_input.to(device)
    test_weight = weight.to(device)
    test_bias = bias.to(device)
    print("*** moved inputs to device")

    with report_time("turbine total time"):
        for i in range(nreps):
            test_output = launcher(test_input, test_weight, test_bias)
            test_input = test_output
        test_output = test_output.to(device="cpu").clone()
    print(test_output)

    torch.testing.assert_close(ref_output, test_output)


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--nreps", type=int, default=100)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dtype", type=str, default="float16")
    args = parser.parse_args(argv)

    dtype = getattr(torch, args.dtype)
    device = torch.device(args.device)
    print("Running on device:", device)
    print("Using dtype:", dtype)
    test_sequential(nreps=args.nreps, device=device, dtype=dtype)


if __name__ == "__main__":
    main(sys.argv[1:])
