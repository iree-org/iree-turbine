# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch
from typing import Sequence
import math
import gc
import sys
from iree.turbine.kernel.boo.conv_exports import (
    miopen_parser as mio,
    get_launchable,
)


def run(cli_args: Sequence[str], gpu_id: int):
    parser = mio.get_miopen_parser()
    parser.add_argument(
        "--iter", type=int, help="Number of iterations to run", default=100
    )
    parser.add_argument(
        "--splat-input-value",
        default=None,
        type=int,
        help="use a splat value for inputs (defaults to random values)",
    )
    args = parser.parse_args(cli_args)
    sig = mio.get_signature(args)
    conv = get_launchable(sig)

    # get the number of available GPU's
    num_devices = 1 if gpu_id != -1 else torch.cuda.device_count()
    devices = (
        [f"cuda:{gpu_id}"]
        if gpu_id != -1
        else [f"cuda:{i}" for i in range(num_devices)]
    )
    iter_per_device = args.iter // num_devices
    rem_iter = args.iter % num_devices

    # generate sample conv args on each GPU
    per_device_data = [
        sig.get_sample_conv_args(
            seed=10,
            device=device,
            splat_value=args.splat_input_value,
        )
        for device in devices
    ]

    # determine an iter threshold to pause and collect garbage
    numel = 0
    if int(sig.mode) == 0:
        numel = math.prod(sig.output_shape)
    elif int(sig.mode) == 1:
        numel = math.prod(sig.input_shape)
    elif int(sig.mode) == 2:
        numel = math.prod(sig.kernel_shape)
    dtype_bytes = int(sig.dtype.itemsize)
    res_mem_bytes = numel * dtype_bytes
    # This is a rough threshold: Mi300x 192 GB memory divided by 2.
    mem_bytes_threshold = 96 * (10**9)
    iter_thresh = int(mem_bytes_threshold // res_mem_bytes)

    result = None
    for iter in range(iter_per_device + 1):
        for device_idx, conv_args in enumerate(per_device_data):
            if iter == iter_per_device and device_idx >= rem_iter:
                break
            result = conv(*conv_args)
        if (iter + 1) % iter_thresh == 0:
            print(f"Synchronizing all devices on iter {iter} and collecting garbage.")
            for i in range(num_devices):
                torch.cuda.synchronize(torch.device(f"cuda:{i}"))
            gc.collect()

    torch.cuda.synchronize()

    # Print the function name so it can be picked up by the pipe.
    print(sig.get_func_name())


if __name__ == "__main__":
    run(sys.argv[2:], int(sys.argv[1]))
