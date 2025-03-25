# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import torch
import warnings

from iree.turbine.kernel.boo.conv_exports.conv import ConvSignature, DEFAULT_LAYOUTS

__all__ = [
    "command_to_signature",
    "get_miopen_parser",
    "get_signature",
]


def get_signature(args):
    layouts = {
        "input_layout": args.in_layout,
        "kernel_layout": args.fil_layout,
        "output_layout": args.out_layout,
    }

    n = args.spatial_dim  # default = 2
    # check provided layouts for a different number of spatial dims
    updated_n = False
    for key, value in layouts.items():
        if value is not None:
            same_n = len(value) - 2 == n
            assert (
                same_n or not updated_n
            ), f"provided layouts have inconsistent rank, see {layouts}"
            if not same_n:
                n = len(value) - 2
                updated_n = True

    # now that n is correct, add default layouts
    for key, value in layouts.items():
        if not value:
            layouts[key] = DEFAULT_LAYOUTS[n]

    batch = args.batchsize
    in_channels = args.in_channels
    groups = args.group_count
    out_channels = args.out_channels

    in_dims = {
        "N": batch,
        "C": in_channels,
        "D": args.in_d,
        "H": args.in_h,
        "W": args.in_w,
    }
    w_dims = {
        "N": out_channels,
        "C": int(in_channels) // int(groups),
        "D": args.fil_d,
        "H": args.fil_h,
        "W": args.fil_w,
    }
    conv_config_dicts = {
        "stride": {
            "D": args.conv_stride_d,
            "H": args.conv_stride_h,
            "W": args.conv_stride_w,
        },
        "padding": {
            "D": args.pad_d,
            "H": args.pad_h,
            "W": args.pad_w,
        },
        "dilation": {
            "D": args.dilation_d,
            "H": args.dilation_h,
            "W": args.dilation_w,
        },
        "output_padding": {
            "D": args.trans_output_pad_d,
            "H": args.trans_output_pad_h,
            "W": args.trans_output_pad_w,
        },
    }
    in_shape = [in_dims[char] for char in layouts["input_layout"]]
    ker_shape = [w_dims[char] for char in layouts["kernel_layout"]]
    bias = bool(args.bias)
    spatial_dims = list(set(layouts["input_layout"]).intersection(["D", "H", "W"]))
    # luckily the order is alphabetical regardless of num_spatial_dims
    spatial_dims.sort()

    conv_config = {
        key: [conv_config_dicts[key][dim] for dim in spatial_dims]
        for key in conv_config_dicts.keys()
    }

    if args.forw == 1:
        mode = "fwd"
    elif args.forw == 2:
        mode = "bwd"
    elif args.forw == 4:
        mode = "wrw"
    else:
        mode = "fwd"
        warnings.warn(
            f"Only one of fwd, bwd, wrw conv supported at one time. Got {command}."
        )
    transposed = args.mode == "trans"
    dtype_dict = {
        "convbfp16": torch.bfloat16,
        "conv": torch.float32,
        "convfp16": torch.float16,
    }
    dtype = dtype_dict[args.command]
    return ConvSignature(
        input_shape=in_shape,
        kernel_shape=ker_shape,
        dtype=dtype,
        input_layout=layouts["input_layout"],
        kernel_layout=layouts["kernel_layout"],
        output_layout=layouts["output_layout"],
        bias=bias,
        stride=conv_config["stride"],
        padding=conv_config["padding"],
        dilation=conv_config["dilation"],
        output_padding=conv_config["output_padding"],
        groups=int(groups),
        transposed=transposed,
        mode=mode,
    )


def command_to_signature(command: str):
    parser = get_miopen_parser()
    args, unknown = parser.parse_known_args(command.split())
    return get_signature(args)


def get_miopen_parser():
    parser = argparse.ArgumentParser()
    # TODO: support commented-out args
    parser.add_argument(
        "command", default="convbfp16", choices=["conv", "convbfp16", "convfp16"]
    )
    parser.add_argument(
        "--spatial_dim",
        "-_",
        type=int,
        default=2,
        help="convolution spatial dimension (Default=2)",
    )
    parser.add_argument(
        "--batchsize", "-n", type=int, default=100, help="Mini-batch size (Default=100)"
    )
    parser.add_argument(
        "--in_channels",
        "-c",
        type=int,
        default=3,
        help="Number of Input Channels (Default=3)",
    )
    parser.add_argument(
        "--in_d", "-!", type=int, default=32, help="Input Depth (Default=32)"
    )
    parser.add_argument(
        "--in_h", "-H", type=int, default=32, help="Input Height (Default=32)"
    )
    parser.add_argument(
        "--in_w", "-W", type=int, default=32, help="Input Width (Default=32)"
    )
    parser.add_argument(
        "--in_layout",
        "-I",
        type=str,
        help="Input Layout (Default=NCHW for 2d conv, NCDHW for 3d conv)",
    )
    # parser.add_argument("--in_cast_type",       "-U", type=str, help="Cast type for input tensor, default to not set")
    parser.add_argument(
        "--out_channels",
        "-k",
        type=int,
        default=32,
        help="Number of Output Channels (Default=32)",
    )
    parser.add_argument(
        "--fil_d", "-@", type=int, default=3, help="Filter Depth (Default=3)"
    )
    parser.add_argument(
        "--fil_h", "-y", type=int, default=3, help="Filter Height (Default=3)"
    )
    parser.add_argument(
        "--fil_w", "-x", type=int, default=3, help="Filter Width (Default=3)"
    )
    parser.add_argument(
        "--fil_layout",
        "-f",
        type=str,
        help="Filter Layout (Default=NCHW for 2d conv, NCDHW for 3d conv)",
    )
    # parser.add_argument("--wei_cast_type",      "-R", type=str, help="Cast type for weight tensor, default to not set")
    parser.add_argument(
        "--bias", "-b", type=int, default=0, help="Use Bias (Default=0)"
    )
    parser.add_argument(
        "--out_layout",
        "-O",
        type=str,
        help="Output Layout (Default=NCHW for 2d conv, NCDHW for 3d conv)",
    )
    # parser.add_argument("--out_cast_type",      "-T", type=str, help="Cast type for output tensor, default to not set")
    parser.add_argument(
        "--forw",
        "-F",
        type=int,
        default=1,
        help="Flag enables fwd, bwd, wrw convolutions. Note: default here differs from MiOpen driver.",
    )
    parser.add_argument(
        "--mode",
        "-m",
        type=str,
        default="conv",
        help="Convolution Mode (conv, trans) (Default=conv)",
    )
    parser.add_argument(
        "--conv_stride_d",
        "-#",
        type=int,
        default=1,
        help="Convolution Stride for Depth (Default=1)",
    )
    parser.add_argument(
        "--conv_stride_h",
        "-u",
        type=int,
        default=1,
        help="Convolution Stride for Height (Default=1)",
    )
    parser.add_argument(
        "--conv_stride_w",
        "-v",
        type=int,
        default=1,
        help="Convolution Stride for Width (Default=1)",
    )
    parser.add_argument(
        "--pad_d", "-$", type=int, default=0, help="Zero Padding for Depth (Default=0)"
    )
    parser.add_argument(
        "--pad_h", "-p", type=int, default=0, help="Zero Padding for Height (Default=0)"
    )
    parser.add_argument(
        "--pad_w", "-q", type=int, default=0, help="Zero Padding for Width (Default=0)"
    )
    # parser.add_argument("--pad_val",            "-r", type=int, default=0, help="Padding Value (Default=0)")
    # parser.add_argument("--pad_mode",           "-z", type=str, default="default", help="Padding Mode (same, valid, default) (Default=default)")
    parser.add_argument(
        "--dilation_d",
        "-^",
        type=int,
        default=1,
        help="Dilation of Filter Depth (Default=1)",
    )
    parser.add_argument(
        "--dilation_h",
        "-l",
        type=int,
        default=1,
        help="Dilation of Filter Height (Default=1)",
    )
    parser.add_argument(
        "--dilation_w",
        "-j",
        type=int,
        default=1,
        help="Dilation of Filter Width (Default=1)",
    )
    parser.add_argument(
        "--trans_output_pad_d",
        "-%",
        type=int,
        default=0,
        help="Zero Padding Output for Depth (Default=0)",
    )
    parser.add_argument(
        "--trans_output_pad_h",
        "-Y",
        type=int,
        default=0,
        help="Zero Padding Output for Height (Default=0)",
    )
    parser.add_argument(
        "--trans_output_pad_w",
        "-X",
        type=int,
        default=0,
        help="Zero Padding Output for Width (Default=0)",
    )
    parser.add_argument(
        "--group_count", "-g", type=int, default=1, help="Number of Groups (Default=1)"
    )
    return parser
