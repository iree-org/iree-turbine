# Copyright 2025 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import (
    Dict,
    Literal,
    Tuple,
    Union,
)
from pathlib import Path
import torch
from iree.turbine.dynamo.backends.basic import backend_generator


class ConvNdBackwards(torch.nn.Module):
    def __init__(
        self,
        num_spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple],
        *,
        stride: Union[int, Tuple] = 1,
        padding: Union[int, Tuple] = 0,
        dilation: Union[int, Tuple] = 1,
        groups: int = 1,
        bias: bool = True
    ):
        if num_spatial_dims > 3 or num_spatial_dims < 1:
            raise ValueError("Number of spatial dimensions should be between 1 and 3")
        if groups != 1:
            raise NotImplementedError("Unimplemented for groups != 1")
        super().__init__()
        self.ND = num_spatial_dims
        self.Cin = in_channels
        self.Cout = out_channels

        def to_tuple(input: Union[int, Tuple]):
            if isinstance(input, Tuple):
                if len(input) != self.ND:
                    raise ValueError(
                        "Provided convolution parameter tuple size does not match number of spatial dimensions."
                    )
                return input
            return self.ND * (input,)

        self.K = to_tuple(kernel_size)
        self.s = to_tuple(stride)
        self.p = to_tuple(padding)
        self.d = to_tuple(dilation)
        self.g = groups

        fwd_map = {1: torch.nn.Conv1d, 2: torch.nn.Conv2d, 3: torch.nn.Conv3d}

        self.fwd_conv = fwd_map[num_spatial_dims](
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

        self.register_parameter("weight", self.fwd_conv.weight)
        if bias:
            self.register_parameter("bias", self.fwd_conv.bias)

    def input_grad(self, loss_gradient, fwd_input_shape: Tuple):
        pad_correction = []
        for i in range(self.ND):
            pad_correction.append(
                (fwd_input_shape[i] + 2 * self.p[i] - self.d[i] * (self.K[i] - 1) - 1)
                % self.s[i]
            )
        pad_correction = tuple(pad_correction)
        return torch.convolution(
            loss_gradient,
            self.weight,
            bias=None,
            stride=self.s,
            padding=self.p,
            dilation=self.d,
            transposed=True,
            output_padding=pad_correction,
            groups=self.g,
        )

    def weight_grad(self, loss_gradient, fwd_input):
        if len(fwd_input.shape) != len(loss_gradient.shape):
            raise ValueError(
                "expected fwd_input and loss_gradient to have the same rank"
            )
        non_spatial_rank = len(fwd_input.shape) - self.ND
        if non_spatial_rank not in [1, 2]:
            raise ValueError("expected one or two non-spatial dims")
        batched = non_spatial_rank == 2
        if batched:
            fwd_input = fwd_input.transpose(0, 1)
            loss_gradient = loss_gradient.transpose(0, 1)
        else:
            fwd_input = fwd_input.unsqueeze(1)
            loss_gradient = loss_gradient.unsqueeze(1)

        # opting to slice out unneccessary values instead of pre-padding the input by (stride - pad_correction) and loss_gradient by 1 at the end of each spatial dim
        conv = torch.convolution(
            fwd_input,
            loss_gradient,
            bias=None,
            stride=self.d,
            padding=self.p,
            dilation=self.s,
            transposed=False,
            output_padding=self.ND * (0,),
            groups=self.g,
        )

        if self.ND == 1:
            sliced = conv[..., : self.K[0]]
        if self.ND == 2:
            sliced = conv[..., : self.K[0], : self.K[1]]
        if self.ND == 3:
            sliced = conv[..., : self.K[0], : self.K[1], : self.K[2]]

        sliced = sliced.transpose(0, 1) if batched else sliced.unsqueeze(1)
        return sliced

    def forward(self, loss_gradient, fwd_input):
        input_grad = self.input_grad(loss_gradient, fwd_input.shape[-self.ND :])
        weight_grad = self.weight_grad(loss_gradient, fwd_input)
        return (input_grad, weight_grad)


def run_comparison(batch_size, input_spatial_dims, config):
    x = torch.randn(
        batch_size, config["in_channels"], *input_spatial_dims, requires_grad=True
    )
    my_bw_conv = ConvNdBackwards(**config)
    my_fw_conv = my_bw_conv.fwd_conv

    # calculate fw output and retain grad for dLdy
    y = my_fw_conv(x)
    y.retain_grad()
    target = torch.randn(y.shape)

    # run a loss function on y and prop backward
    loss_fn = torch.nn.MSELoss()
    loss = loss_fn(y, target)
    loss.backward(retain_graph=True)

    # loss_gradient of fwd output
    dLdy = y.grad
    # pytorch computed gradients
    dLdx = x.grad
    dLdw = my_fw_conv.weight.grad

    # custom gradient computations
    my_dLdx, my_dLdw = my_bw_conv(dLdy, x)
    same_dLdx = torch.allclose(my_dLdx, dLdx)
    same_dLdw = torch.allclose(my_dLdw, dLdw)
    return same_dLdx, same_dLdw


def dump_mlir(
    batch_size: int,
    input_spatial_dims: Tuple,
    config: Dict[str, Union[int, Tuple]],
    save_dir: Union[str, Path] = Path(__file__).parent,
    weight_grad: bool = True,
    input_grad: bool = True,
):
    x = torch.randn(batch_size, config["in_channels"], *input_spatial_dims)
    my_bw_conv = ConvNdBackwards(**config)
    my_bw_conv.eval()
    y = my_bw_conv.fwd_conv(x)
    t = torch.randn(*y.shape)
    options = lambda name: {
        "target_backends": "llvm-cpu",
        "flags": ["--iree-llvmcpu-target-cpu=host"],
        "driver": "local-task",
        "save_mlir": str(save_dir / name),
    }
    if weight_grad:
        wg = torch.compile(
            my_bw_conv.weight_grad,
            backend=backend_generator(**options("weight_grad.mlir")),
        )
        wg(t, x)
    if input_grad:
        ig = torch.compile(
            my_bw_conv.input_grad,
            backend=backend_generator(**options("input_grad.mlir")),
        )
        ig(t, x.shape[2:])
    return


if __name__ == "__main__":
    config = {
        "num_spatial_dims": 2,
        "in_channels": 2,
        "out_channels": 3,
        "kernel_size": (2, 2),
        "stride": (1, 1),
        "dilation": (1, 1),
        "padding": (1, 1),
        "groups": 1,
    }
    batch_size = 2
    input_spatial_dims = (10, 20)
    dump_mlir(batch_size, input_spatial_dims, config)
