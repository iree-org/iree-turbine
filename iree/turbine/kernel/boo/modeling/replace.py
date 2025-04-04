# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch
from ..ops.conv import boo_conv
from ....support.logging import aot_logger as logger

__all__ = [
    "BooConv2d",
    "replace_conv2d_with_boo_conv",
]


class BooConv2d(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
    ):
        super(BooConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self._bias = bias

        self.weight = torch.nn.Parameter(
            torch.randn(out_channels, in_channels // groups, *kernel_size)
        )
        if bias:
            self.bias = torch.nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter("bias", None)

    def forward(self, x):
        if self._bias:
            args = (x, self.weight, self.bias)
        else:
            args = (x, self.weight)
        return boo_conv(
            *args,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            shared_layout="NCHW"
        )


def replace_conv2d_with_boo_conv(model):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            logger.debug("Found Conv2d to replace with BooConv2d : %s", str(module))
            custom_conv = BooConv2d(
                module.in_channels,
                module.out_channels,
                module.kernel_size,
                module.stride,
                module.padding,
                module.dilation,
                module.groups,
                module.bias is not None,
            )

            # Copy weights and bias if applicable
            custom_conv.weight = torch.nn.Parameter(module.weight.data.clone())
            if module.bias is not None:
                custom_conv.bias = torch.nn.Parameter(module.bias.data.clone())

            # Replace the module
            parts = name.split(".")
            if len(parts) > 1:
                parent_name = ".".join(parts[:-1])
                child_name = parts[-1]
                parent_module = model.get_submodule(parent_name)
                setattr(parent_module, child_name, custom_conv)
            else:
                setattr(model, name, custom_conv)
    return model
