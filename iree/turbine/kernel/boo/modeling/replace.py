# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Dict, Sequence
import torch
from ..ops.conv import boo_convolution, make_tuple
from ....support.logging import aot_logger as logger

__all__ = [
    "BooConv1d",
    "BooConv2d",
    "BooConv3d",
    "replace_convs_with_boo",
    "replace_conv2d_with_boo_conv",
]


class BooConv1d(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Sequence[int],
        stride: int | Sequence[int] = 1,
        padding: int | Sequence[int] = 0,
        dilation: int | Sequence[int] = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = make_tuple(kernel_size, 1)
        self.stride = make_tuple(stride, 1)
        self.padding = make_tuple(padding, 1)
        self.dilation = make_tuple(dilation, 1)
        self.groups = groups

        self.weight = torch.nn.Parameter(
            torch.randn(out_channels, in_channels // groups, *self.kernel_size)
        )
        if bias:
            self.bias = torch.nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter("bias", None)

    def forward(self, x):
        no_batch = len(x.shape) == 2
        if no_batch:
            x = x.unsqueeze(0)
        # There is no 1-D channels_last format.
        w = self.weight
        bias = self.bias
        device_type = x.device.type
        if torch.is_autocast_enabled(device_type):
            dtype = torch.get_autocast_dtype(device_type)
            x = x.to(dtype=dtype)
            w = w.to(dtype=dtype)
            bias = bias if bias is None else bias.to(dtype=dtype)
        result = boo_convolution(
            x,
            w,
            bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        return result.squeeze(0) if no_batch else result


class BooConv2d(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int | Sequence[int] = 1,
        padding: int | Sequence[int] = 0,
        dilation: int | Sequence[int] = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = make_tuple(kernel_size, 2)
        self.stride = make_tuple(stride, 2)
        self.padding = make_tuple(padding, 2)
        self.dilation = make_tuple(dilation, 2)
        self.groups = groups

        self.weight = torch.nn.Parameter(
            torch.randn(out_channels, in_channels // groups, *self.kernel_size)
        )
        if bias:
            self.bias = torch.nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter("bias", None)

    def forward(self, x):
        no_batch = len(x.shape) == 3
        if no_batch:
            x = x.unsqueeze(0)
        w = self.weight
        bias = self.bias
        device_type = x.device.type
        if torch.is_autocast_enabled(device_type):
            dtype = torch.get_autocast_dtype(device_type)
            x = x.to(dtype=dtype)
            w = w.to(dtype=dtype)
            bias = bias if bias is None else bias.to(dtype=dtype)
        result = boo_convolution(
            x,
            w,
            bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        return result.squeeze(0) if no_batch else result


class BooConv3d(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int | Sequence[int] = 1,
        padding: int | Sequence[int] = 0,
        dilation: int | Sequence[int] = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = make_tuple(kernel_size, 3)
        self.stride = make_tuple(stride, 3)
        self.padding = make_tuple(padding, 3)
        self.dilation = make_tuple(dilation, 3)
        self.groups = groups

        self.weight = torch.nn.Parameter(
            torch.randn(out_channels, in_channels // groups, *self.kernel_size)
        )
        if bias:
            self.bias = torch.nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter("bias", None)

    def forward(self, x):
        no_batch = len(x.shape) == 4
        if no_batch:
            x = x.unsqueeze(0)
        w = self.weight
        bias = self.bias
        device_type = x.device.type
        if torch.is_autocast_enabled(device_type):
            dtype = torch.get_autocast_dtype(device_type)
            x = x.to(dtype=dtype)
            w = w.to(dtype=dtype)
            bias = bias if bias is None else bias.to(dtype=dtype)
        result = boo_convolution(
            x,
            w,
            bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        return result.squeeze(0) if no_batch else result


NUM_DIMS_TO_CLS_MAPPING = {
    1: (torch.nn.Conv1d, BooConv1d),
    2: (torch.nn.Conv2d, BooConv2d),
    3: (torch.nn.Conv3d, BooConv3d),
}


def replace_convs_with_boo(
    model,
    *,
    allowed_spatial_dims: Sequence[int] = (1, 2, 3),
    kwarg_mapping: Dict[int, Dict] = {},
):
    """Returns a torch.nn.Module which is the same as `model` but with some submodules replaced with boo convs.

    - `allowed_spatial_dims` is a list of spatial dims for which the replacement should apply.
    - `kwarg_mapping` is a restriction on which modules get replaced for each allowed spatial dim.

    Usage examples:

    1. Replace all convs (1d, 2d, and 3d): `model = replace_convs_with_boo(model)`.
    2. Replace all convs, but for 2d, only replace stride=(1,1) convs: `model = replace_convs_with_boo(model, kwarg_mapping={2 : {"stride" : (1,1)}})`.
    3. Replace only 1D convs: `model = replace_convs_with_boo(model, allowed_spatial_dims=(1,))`.
    """

    for name, module in model.named_modules():
        for spatial_dim in allowed_spatial_dims:
            assert (
                spatial_dim in NUM_DIMS_TO_CLS_MAPPING.keys()
            ), f"provided spatial dim invalid. Got {spatial_dim}, but only allow 1d, 2d, or 3d replacements."
            src_cls, dst_cls = NUM_DIMS_TO_CLS_MAPPING[spatial_dim]
            if isinstance(module, src_cls):
                logger.debug("Found %s: %s", str(src_cls.__name__), str(module))

                skip = False
                replacement_kwargs = kwarg_mapping.get(spatial_dim, {})
                for k, v in replacement_kwargs.items():
                    attr = module.__getattribute__(k)
                    logger.debug(
                        "For key (%s), got value (%s) and requested %s.",
                        str(k),
                        str(attr),
                        str(v),
                    )
                    if v != attr:
                        skip = True
                        break

                if skip:
                    continue

                logger.debug(
                    "Replacing %s with %s : %s",
                    str(src_cls.__name__),
                    str(dst_cls.__name__),
                    str(module),
                )
                custom_conv = dst_cls(
                    module.in_channels,
                    module.out_channels,
                    module.kernel_size,
                    module.stride,
                    module.padding,
                    module.dilation,
                    module.groups,
                    module.bias is not None,
                )

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


def replace_conv2d_with_boo_conv(model, **kwargs):
    """Replace Conv2d modules in a model, with matching kwargs, with BooConv2d.

    For example:

    ```python
    model = replace_conv2d_with_boo_conv(model, stride=(1,1))
    ```

    This will replace all Conv2d with stride=(1,1) in `model` with an equivalent BooConv2d.
    Not passing any kwargs will replace all Conv2d modules with BooConv2d.
    """

    return replace_convs_with_boo(
        model, allowed_spatial_dims=(2,), kwarg_mapping={2: kwargs}
    )
