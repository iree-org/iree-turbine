# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import torch


def main():
    parser = argparse.ArgumentParser(
        description="""
Run a layer_norm with the IREE runtime.
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--sizes", "-s", type=str, required=True)
    parser.add_argument("--n-normalized-dims", "-n", type=int, required=False)
    parser.add_argument("--elementwise_affine", "-w", type=bool, default=True)
    parser.add_argument("--bias", "-b", type=bool, default=True)
    parser.add_argument(
        "--dtype", "-d", type=str, choices=["float16", "bfloat16", "float32"]
    )
    parser.add_argument("--splat-value", type=float, required=False)
    args = parser.parse_args()

    sizes = list(map(int, args.sizes.strip().split(",")))
    normalized_shape = (
        args[-args.n_normalized_dims :] if args.n_normalized_dims else sizes[1:]
    )

    from iree.turbine.kernel.boo.layer_norm_exports.layer_norm import LayerNormSignature
    from iree.turbine.kernel.boo.layer_norm_exports.launch import get_launchable

    signature = LayerNormSignature(
        input_shape=sizes,
        normalized_shape=normalized_shape,
        elementwise_affine=args.elementwise_affine,
        bias=args.bias,
        dtype=args.dtype,
    )
    launchable = get_launchable(signature)

    tensors = signature.get_sample_args(
        splat_value=args.splat_value, device=torch.device("cuda:0")
    )
    reference_module = signature.get_nn_module()
    results = launchable(*tensors)
    torch.cuda.synchronize()

    reference_results = reference_module(*tensors)
    for expected, actual in zip(reference_results, results):
        if not torch.allclose(expected, actual, atol=1e-5, rtol=1e-5):
            print(expected - actual)
            raise ValueError("Output mismatch.")


if __name__ == "__main__":
    main()
