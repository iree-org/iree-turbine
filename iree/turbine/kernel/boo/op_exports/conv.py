# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
from typing import Any
from enum import IntEnum
from functools import lru_cache
import math

import torch

from .utils import Permutation
from ..exports.signature import OpSignature, ModeBase
from ..exports.parser import OpCLIParser
from ....ops.conv_fwd import conv_2d_nhwc_fhwc, generic_conv
from ....ops.insert_slice import insert_slice

__all__ = [
    "Mode",
    "ConvParser",
    "ConvSignature",
    "ConvForward",
    "ConvBackward",
    "DEFAULT_LAYOUTS",
    "get_conv_func_name",
]


class Mode(ModeBase, IntEnum):
    FORWARD = 0  # Special value in that `FOWARD | <other-mode> = <other-mode>`
    INPUT_BACKWARD = 1
    WEIGHT_BACKWARD = 2
    INPUT_WEIGHT_BACKWARD = 3
    BIAS_BACKWARD = 4
    INPUT_BIAS_BACKWARD = 5
    WEIGHT_BIAS_BACKWARD = 6
    ALL_BACKWARD = 7

    # alias values
    FWD = FORWARD
    BWD = INPUT_BACKWARD
    WRW = WEIGHT_BACKWARD

    def __str__(self) -> str:
        return self.name

    def __or__(self, other) -> "Mode":
        if not isinstance(other, (int, Mode)):
            raise TypeError(f"Invalid add operation: Mode + ({type(other) = }).")
        return Mode(self.value + int(other))

    @staticmethod
    def from_backward_mask(mask: list[bool]) -> "Mode":
        mode = Mode.FORWARD
        for m, bwd_mode in zip(
            mask,
            [Mode.INPUT_BACKWARD, Mode.WEIGHT_BACKWARD, Mode.BIAS_BACKWARD],
            strict=True,
        ):
            if m:
                mode = mode | bwd_mode
        return mode

    @property
    def backward_mask(self) -> list[bool]:
        v = self.value
        return [
            bool(v % 2),
            bool((v >> 1) % 2),
            bool((v >> 2) % 2),
        ]


DEFAULT_LAYOUTS = {1: "NCH", 2: "NCHW", 3: "NCDHW"}


@lru_cache(maxsize=None)
def get_conv_func_name(
    input_shape: tuple,
    kernel_shape: tuple,
    dtype: str,
    mode: str,
    bias: bool,
    stride: tuple,
    padding: tuple,
    dilation: tuple,
    groups: int,
    input_layout: str,
    kernel_layout: str,
    output_layout: str,
) -> str:
    """Returns the function name to use for a convolution with the specified configuration.

    If a signature object is available, use `OpSignature.func_name` instead.
    """
    num_spatial_dims = len(input_shape) - 2
    name_items = [
        "conv",
        f"{num_spatial_dims}d",
        str(dtype).removeprefix("torch."),
        str(mode).lower(),
    ]
    if bias and mode == "FORWARD":
        name_items.append("b")
    to_shape_string = lambda l: "x".join([str(i) for i in l])
    name_items.extend(
        [
            to_shape_string(input_shape),
            input_layout.lower(),
            to_shape_string(kernel_shape),
            kernel_layout.lower().replace("n", "f"),
            output_layout.lower().replace("c", "f"),
            to_shape_string(stride) + "s",
            to_shape_string(padding) + "p",
            to_shape_string(dilation) + "d",
            f"{groups}g",
        ]
    )
    return "_".join(name_items)


class ConvSignature(OpSignature):
    """
    Convolution signature that provides information for launching specific kernels.
    """

    input_shape: list[int]
    kernel_shape: list[int]
    num_spatial_dims: int
    dtype: torch.dtype
    input_layout: str
    kernel_layout: str
    output_layout: str
    bias: bool
    stride: list[int]
    padding: list[int]
    dilation: list[int]
    transposed: bool
    output_padding: list[int]
    groups: int
    mode: Mode

    def __init__(
        self,
        *,
        input_shape: list[int],
        kernel_shape: list[int],
        shared_layout: str | None = None,
        input_layout: str | None = None,
        kernel_layout: str | None = None,
        output_layout: str | None = None,
        bias: bool = False,
        dtype: torch.dtype = torch.bfloat16,
        stride: int | list[int] = 1,
        padding: int | list[int] = 0,
        dilation: int | list[int] = 1,
        transposed: bool = False,
        output_padding: int | list[int] = 0,
        groups: int = 1,
        mode: str | Mode = Mode.FORWARD,
        backward_mask: list[bool] | None = None,
    ):
        if len(input_shape) != len(kernel_shape):
            raise ValueError(
                f"Expected same rank for input and kernel, got {input_shape=}, {kernel_shape=}"
            )
        num_spatial_dims = len(input_shape) - 2
        default_layout = DEFAULT_LAYOUTS[num_spatial_dims]

        def get_layout(provided) -> str:
            if shared_layout:
                return shared_layout
            if provided:
                return provided
            return default_layout

        def listify(value: Any) -> list[int]:
            if isinstance(value, list):
                assert len(value) == num_spatial_dims
                return value
            if isinstance(value, int):
                return [value] * num_spatial_dims
            try:
                return list(value)
            except TypeError as e:
                raise TypeError(
                    f"ConvSignature kwarg has value {value} with type {type(value).__name__}, but expected int or iterable."
                ) from e

        if backward_mask is not None:
            mode = Mode.from_backward_mask(backward_mask)
        elif isinstance(mode, str):
            mode = Mode.parse(mode)

        self.input_shape = input_shape
        self.kernel_shape = kernel_shape
        self.num_spatial_dims = num_spatial_dims
        self.dtype = dtype
        self.input_layout = get_layout(input_layout)
        self.kernel_layout = get_layout(kernel_layout)
        self.output_layout = get_layout(output_layout)
        self.bias = bias
        self.stride = listify(stride)
        self.padding = listify(padding)
        self.dilation = listify(dilation)
        self.output_padding = listify(output_padding)
        self.transposed = transposed
        self.groups = groups
        self.mode = mode

    @property
    def input_perms(self) -> Permutation:
        """Converts input layout to default"""
        default = DEFAULT_LAYOUTS[self.num_spatial_dims]
        return Permutation.get(self.input_layout, default)

    @property
    def kernel_perms(self) -> Permutation:
        """Converts weight (conflated with kernel/filter) layout to default"""
        default = DEFAULT_LAYOUTS[self.num_spatial_dims]
        return Permutation.get(self.kernel_layout, default)

    @property
    def output_perms(self) -> Permutation:
        """Converts default output layout back to specified output layout"""
        default = DEFAULT_LAYOUTS[self.num_spatial_dims]
        return Permutation.get(default, self.output_layout)

    @property
    def output_shape(self) -> list:
        """Gets the output shape of the forward conv."""
        # pytorch conv shapes:
        in_shape_p = self.input_perms(self.input_shape)
        ker_shape_p = self.kernel_perms(self.kernel_shape)
        out_shape_p = [in_shape_p[0], ker_shape_p[0]]  # [N,C]
        for i in range(self.num_spatial_dims):
            out_shape_p.append(
                (
                    (in_shape_p[i + 2] - 1)
                    + 2 * self.padding[i]
                    - self.dilation[i] * (ker_shape_p[i + 2] - 1)
                )
                // self.stride[i]
                + 1,
            )
        # out_shape_p is NCHW (pytorch) so permute back to specified layout.
        return self.output_perms(out_shape_p)

    @property
    def explicit_padding(self) -> list[int]:
        """Padding of input tensor compatible with torch.constant_pad_nd."""
        torch_pads_NCHW = [[0, 0], [0, 0]] + [[p, p] for p in self.padding]
        # permute back to input ordering
        permuted_pads = self.input_perms.inv()(torch_pads_NCHW)
        # to make compatible with torch.nn.functional.pad reverse ordering
        permuted_pads.reverse()
        # flatten the list
        return [p for dim_pads in permuted_pads for p in dim_pads]

    @property
    def input_grouped_dim(self) -> int:
        layout = self.input_layout
        grouped_dim_char = "C"
        return layout.find(grouped_dim_char)

    @property
    def kernel_grouped_dim(self) -> int:
        layout = self.kernel_layout
        grouped_dim_char = "N"
        return layout.find(grouped_dim_char)

    @property
    def output_grouped_dim(self) -> int:
        layout = self.output_layout
        grouped_dim_char = "C"
        return layout.find(grouped_dim_char)

    @property
    def is_forward(self) -> bool:
        return self.mode == Mode.FORWARD

    @staticmethod
    def get(
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
        **kwargs,
    ) -> "ConvSignature":
        """gets a signature from provided input, weight, bias tensors and additional kwargs"""
        return ConvSignature(
            input_shape=list(input.shape),
            kernel_shape=list(weight.shape),
            dtype=input.dtype,
            bias=(bias is not None),
            **kwargs,
        )

    def as_init_kwargs(self) -> dict[str, Any]:
        # "num_spatial_dims" is a derived value and is not needed for
        # construction, therefore it is intentionally excluded from the list
        # below.
        return {
            key: getattr(self, key)
            for key in (
                "input_shape",
                "kernel_shape",
                "dtype",
                "input_layout",
                "kernel_layout",
                "output_layout",
                "bias",
                "stride",
                "padding",
                "dilation",
                "transposed",
                "output_padding",
                "groups",
                "mode",
            )
        }

    def arrange_backward_launch_args(
        self,
        forward_args: tuple[torch.Tensor, ...],
        forward_results: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, ...]:
        assert not self.is_forward
        x, w, *_ = forward_args
        return (x, w)

    def get_conv_kwargs(self) -> dict[str, Any]:
        """Gets `torch.convolution` (forward-only) kwargs from the signature."""
        conv_extra_args = [
            "stride",
            "padding",
            "dilation",
            "transposed",
            "output_padding",
            "groups",
        ]
        kwargs = self.as_init_kwargs()
        return {name: kwargs[name] for name in conv_extra_args}

    def get_sample_args(
        self,
        *,
        device: str | torch.device | None = None,
        splat_value: int | float | None = None,
        seed: int | None = None,
    ) -> tuple[torch.Tensor, ...]:
        """Gets example args for the convolution (mode-dependent)"""
        out_channels = self.kernel_shape[self.kernel_perms[0]]
        gen = torch.Generator(device=device)
        gen = gen if not seed else gen.manual_seed(seed)

        def get(shape):
            if splat_value is not None:
                return torch.ones(shape, dtype=self.dtype, device=device) * splat_value
            return torch.randn(shape, generator=gen, dtype=self.dtype, device=device)

        if self.mode == Mode.FORWARD:
            # (x, w, b) or (x, w)
            return (
                (get(self.input_shape), get(self.kernel_shape), get(out_channels))
                if self.bias
                else (get(self.input_shape), get(self.kernel_shape))
            )
        # All backward modes take (dLdy, x, w) as inputs.
        # (dLdy, x, w)
        return (
            get(self.output_shape),
            get(self.input_shape),
            get(self.kernel_shape),
        )

    @property
    def func_name(self) -> str:
        return get_conv_func_name(
            tuple(self.input_shape),
            tuple(self.kernel_shape),
            self.dtype,
            str(self.mode),
            self.bias,
            tuple(self.stride),
            tuple(self.padding),
            tuple(self.dilation),
            self.groups,
            self.input_layout,
            self.kernel_layout,
            self.output_layout,
        )

    @property
    def backward_mask(self) -> list[bool]:
        return self.mode.backward_mask

    def get_nn_module(self, *, use_custom: bool = False) -> torch.nn.Module:
        """For a given ConvSignature, returns a torch.nn.Module implementation."""
        if self.mode == Mode.FORWARD:
            return (
                ConvForwardCustomGeneric(self)
                if use_custom and not self.transposed
                else ConvForward(self)
            )
        mask = self.backward_mask
        num_grads = sum(int(m) for m in mask)
        assert (
            num_grads > 0 and num_grads <= 3
        ), f"Expected between one and three backward computations for mode: {self.mode}."
        if use_custom:
            return ConvCustomBackward(self)
        return ConvBackward(self)

    def get_output_size(self) -> int:
        numel = 0
        dtype_bytes = int(self.dtype.itemsize)
        mask = self.backward_mask
        if not any(mask):
            numel = math.prod(self.output_shape)
            return numel * dtype_bytes
        for m, shape in zip(
            mask,
            [
                self.input_shape,
                self.kernel_shape,
                [self.kernel_shape[self.kernel_layout.find("N")]],
            ],
            strict=True,
        ):
            if not m:
                continue
            numel += math.prod(shape)
        return numel * dtype_bytes


class ConvForward(torch.nn.Module):
    def __init__(self, sig: ConvSignature):
        super().__init__()
        self.perms = [
            sig.input_perms,
            sig.kernel_perms,
            sig.output_perms,
        ]
        self.kwargs = sig.get_conv_kwargs()
        if not sig.bias:
            self.kwargs["bias"] = None

    def forward(self, *args: torch.Tensor) -> torch.Tensor:
        mod_args = [
            self.perms[0](args[0]),
            self.perms[1](args[1]),
        ]
        if "bias" not in self.kwargs.keys():
            mod_args.append(args[2])
        output = torch.convolution(*mod_args, **self.kwargs)
        return self.perms[2](output)


class ConvForwardCustomNHWC(torch.nn.Module):
    def __init__(self, sig: ConvSignature):
        super().__init__()
        assert sig.groups == 1, "Grouped custom nhwc is currently unsupported."
        assert not sig.transposed, "Transposed custom nhwc is currently unsupported."
        target_layout = "NHWC"
        nchw_to_nhwc = Permutation.get("NCHW", target_layout)
        self.perms = [
            nchw_to_nhwc * sig.input_perms,
            nchw_to_nhwc * sig.kernel_perms,
            sig.output_perms * (nchw_to_nhwc.inv()),
        ]
        self.stride = sig.stride
        self.dilation = sig.dilation
        self.has_bias = sig.bias
        self.explicit_padding = sig.explicit_padding

    def forward(self, *args: torch.Tensor) -> torch.Tensor:
        x_pad = torch.constant_pad_nd(args[0], self.explicit_padding, value=0)
        x_pad = self.perms[0](x_pad)
        w = self.perms[1](args[1])
        output = conv_2d_nhwc_fhwc(x_pad, w, self.stride, self.dilation)
        output = output.to(dtype=x_pad.dtype)
        if self.has_bias:
            # Note bias has shape [out_channels], which is currently last
            output = output + args[2]
        return self.perms[2](output)


class ConvForwardCustomGeneric(torch.nn.Module):
    def __init__(self, sig: ConvSignature):
        super().__init__()
        if sig.transposed:
            raise NotImplementedError("Generic conv tranpose fwd NYI.")
        self.groups = sig.groups
        self.xl = str(sig.input_layout).lower()
        self.wl = str(sig.kernel_layout).lower().replace("n", "f")
        self.ol = str(sig.output_layout).lower().replace("c", "f")
        self.explicit_padding = sig.explicit_padding
        self.explicit_shape = list(sig.output_shape)
        self.x_pos = sig.input_grouped_dim
        self.w_pos = sig.kernel_grouped_dim
        self.o_pos = sig.output_grouped_dim
        if self.groups != 1:
            self.xl = self.xl[: self.x_pos] + "g" + self.xl[self.x_pos :]
            self.wl = self.wl[: self.w_pos] + "g" + self.wl[self.w_pos :]
            self.ol = self.ol[: self.o_pos] + "g" + self.ol[self.o_pos :]
            self.explicit_shape = (
                self.explicit_shape[: self.o_pos]
                + [self.groups, self.explicit_shape[self.o_pos] // self.groups]
                + self.explicit_shape[self.o_pos + 1 :]
            )
            pad_g_idx = len(self.explicit_padding) - 1 - 2 * self.x_pos
            self.explicit_padding = (
                self.explicit_padding[:pad_g_idx]
                + [0, 0]
                + self.explicit_padding[pad_g_idx:]
            )
        self.stride = sig.stride
        self.dilation = sig.dilation
        self.has_bias = sig.bias

    def forward(self, *args: torch.Tensor) -> torch.Tensor:
        x = args[0]
        w = args[1]
        if self.groups != 1:
            x = x.unflatten(self.x_pos, [self.groups, -1])
            w = w.unflatten(self.w_pos, [self.groups, -1])
        x_pad = torch.constant_pad_nd(x, self.explicit_padding, value=0)
        output = generic_conv(
            x_pad,
            w,
            self.stride,
            self.dilation,
            self.xl,
            self.wl,
            self.ol,
            self.explicit_shape,
        ).to(dtype=x_pad.dtype)
        if self.groups != 1:
            output = output.flatten(self.o_pos, self.o_pos + 1)
        if self.has_bias:
            # Note bias has shape [f], but f in output shape need not be at the back.

            sizes = [-1]
            sizes.extend([1] * (len(self.ol.replace("g", "")) - self.o_pos - 1))
            output = output + args[2].unflatten(0, sizes)
        return output


class ConvBackwardInputCustomGeneric(torch.nn.Module):
    def __init__(self, sig: ConvSignature):
        super().__init__()
        if sig.transposed:
            raise NotImplementedError(
                "unimplemented weight grad decomposition: transposed conv"
            )
        self.ND = sig.num_spatial_dims
        # after preprocessing dLdy, we perform the convolution with strides == 1
        self.stride = [1] * len(sig.stride)
        self.dtype = sig.dtype
        self.dilation = sig.dilation
        self.groups = sig.groups
        self.xl = str(sig.output_layout).lower()
        # Note: reduction dim is filter channel (which we call c). E.g. NCHW -> FCHW -> cfhw
        self.wl = str(sig.kernel_layout).replace("N", "c").replace("C", "f").lower()
        # output layout:
        self.ol = str(sig.input_layout).replace("C", "f").lower()
        self.explicit_shape = list(sig.input_shape)

        self.x_pos = sig.output_grouped_dim
        self.w_pos = sig.kernel_grouped_dim
        self.o_pos = sig.input_grouped_dim
        if self.groups != 1:
            self.xl = self.xl[: self.x_pos] + "g" + self.xl[self.x_pos :]
            self.wl = self.wl[: self.w_pos] + "g" + self.wl[self.w_pos :]
            self.ol = self.ol[: self.o_pos] + "g" + self.ol[self.o_pos :]
            self.explicit_shape[self.o_pos] = (
                self.explicit_shape[self.o_pos] // self.groups
            )
            self.explicit_shape.insert(self.o_pos, self.groups)

        self.input_padding = sig.padding

        # compute the dims which need a flip for the weight tensor
        self.flip_dims = []
        for i, (char, size) in enumerate(zip(sig.kernel_layout, sig.kernel_shape)):
            if char in {"N", "C"} or size == 1:
                continue
            self.flip_dims.append(i if self.groups == 1 or i < self.w_pos else i + 1)

        K_spatial = sig.kernel_perms(sig.kernel_shape)[2:]
        H_spatial = sig.input_perms(sig.input_shape)[2:]

        # When computing dLdx, we sum over all elements of dLdy and ker s.t.:
        #   s*dLdy_idx + d*ker_idx = dLdx_idx + p,
        # Assume for now that s=1 (we will resolve this later)
        # When computing for dLdx_idx = 0, we need to access
        # all possible values for dLdy_idx and ker_idx satisfying:
        #   dLdy_idx + d*ker_idx = p
        # This would be fine, but the last element of ker_idx (K - 1) would give
        #   dLdy_idx = p - d*(K-1)
        # If this expression is negative, we need to add zero padding to dLdy:
        #   padded_idx = dLdy_idx + d*(K - 1) - p
        padding = list(
            [
                sig.dilation[i] * (K_spatial[i] - 1) - sig.padding[i]
                for i in range(self.ND)
            ]
        )
        # TODO: if p > d*(K-1), we could write in an extract slice op or introduce an offset in the conv generic.
        # This isn't a common situation, as p > d*(K-1) means the fwd conv was excessively padded.
        for p in padding:
            if p < 0:
                raise NotImplementedError(
                    "Negative padding not currently supported in conv transpose -> conv decomposition."
                )

        # Determine if we need to resolve strides
        self._stride = sig.stride
        self._do_insert_slice = False
        for s in self._stride:
            if s > 1:
                self._do_insert_slice = True
                break

        # If strides are > 1, we scatter dLdy into a zero init tensor
        self._slice_offset = []
        self._slice_stride = []
        self._strided_sizes = []
        self.explicit_padding = []
        if self._do_insert_slice:
            shape_pt_layout = sig.output_perms.inv()(sig.output_shape)
            for i, size in enumerate(shape_pt_layout):
                if i < 2 or sig.stride[i - 2] == 1:
                    self._slice_offset.append(0)
                    self._slice_stride.append(1)
                    self._strided_sizes.append(size)
                    continue
                stride = sig.stride[i - 2]
                self._slice_offset.append(padding[i - 2])
                self._slice_stride.append(stride)
                # We need the strided dLdy tensor large enough to see all possible h + d*k values
                self._strided_sizes.append(
                    (H_spatial[i - 2] - 1)
                    + sig.dilation[i - 2] * (K_spatial[i - 2] - 1)
                    + 1
                )
            # Permute lists back to original layout
            self._slice_offset = sig.output_perms(self._slice_offset)
            self._slice_stride = sig.output_perms(self._slice_stride)
            self._strided_sizes = sig.output_perms(self._strided_sizes)
            if self.groups != 1:
                self._slice_offset.insert(self.x_pos, 0)
                self._slice_stride.insert(self.x_pos, 1)
                self._strided_sizes[self.x_pos] = (
                    self._strided_sizes[self.x_pos] // self.groups
                )
                self._strided_sizes.insert(self.x_pos, self.groups)
        else:
            # if we have all strides == 1, we can just pad the dLdy tensor
            torch_pads_NCHW = [[0, 0], [0, 0]] + [[p, p] for p in padding]
            # permute back to input ordering
            permuted_pads = sig.output_perms(torch_pads_NCHW)
            if self.groups != 1:
                # put the group dim padding after channel dim (since we reverse)
                permuted_pads.insert(self.x_pos + 1, [0, 0])
            # to make compatible with torch.nn.functional.pad reverse ordering
            permuted_pads.reverse()
            # flatten the list
            self.explicit_padding = list(
                [p for dim_pads in permuted_pads for p in dim_pads]
            )

    def forward(
        self, dLdy: torch.Tensor, x: torch.Tensor, w: torch.Tensor
    ) -> torch.Tensor:
        if self.groups != 1:
            dLdy = dLdy.unflatten(self.x_pos, [self.groups, -1])
            w = w.unflatten(self.w_pos, [self.groups, -1])

        if len(self.flip_dims) != 0:
            w = torch.flip(w, self.flip_dims)

        if self._do_insert_slice:
            zero_init = torch.zeros(
                self._strided_sizes, dtype=dLdy.dtype, device=dLdy.device
            )
            dLdy = insert_slice(dLdy, zero_init, self._slice_offset, self._slice_stride)
        else:
            dLdy = torch.constant_pad_nd(dLdy, self.explicit_padding, 0)

        dLdx = generic_conv(
            dLdy,
            w,
            self.stride,
            self.dilation,
            self.xl,
            self.wl,
            self.ol,
            self.explicit_shape,
        ).to(dtype=self.dtype)

        if self.groups != 1:
            dLdx = dLdx.flatten(self.o_pos, self.o_pos + 1)

        return dLdx


class ConvBackwardWeightCustomGeneric(torch.nn.Module):
    def __init__(self, sig: ConvSignature):
        super().__init__()
        if sig.transposed:
            raise NotImplementedError(
                "unimplemented weight grad decomposition: transposed conv"
            )
        self.stride = sig.dilation
        self.dilation = sig.stride
        self.groups = sig.groups
        self.dtype = sig.dtype
        # Note: need to swap reduction dim to N
        self.xl = str(sig.input_layout).replace("N", "c").replace("C", "n").lower()
        self.wl = str(sig.output_layout).replace("N", "c").replace("C", "f").lower()
        # output layout:
        self.ol = str(sig.kernel_layout).replace("N", "f").replace("C", "n").lower()
        self.explicit_padding = sig.explicit_padding
        self.explicit_shape = list(sig.kernel_shape)
        self.x_pos = sig.input_grouped_dim
        self.w_pos = sig.output_grouped_dim
        self.o_pos = sig.kernel_grouped_dim
        if self.groups != 1:
            self.xl = self.xl[: self.x_pos] + "g" + self.xl[self.x_pos :]
            self.wl = self.wl[: self.w_pos] + "g" + self.wl[self.w_pos :]
            self.ol = self.ol[: self.o_pos] + "g" + self.ol[self.o_pos :]
            self.explicit_shape[self.o_pos] = (
                self.explicit_shape[self.o_pos] // self.groups
            )
            self.explicit_shape.insert(self.o_pos, self.groups)
            pad_g_idx = len(self.explicit_padding) - 1 - 2 * self.x_pos
            self.explicit_padding = (
                self.explicit_padding[:pad_g_idx]
                + [0, 0]
                + self.explicit_padding[pad_g_idx:]
            )

    def forward(
        self, dLdy: torch.Tensor, x: torch.Tensor, w: torch.Tensor
    ) -> torch.Tensor:
        if self.groups != 1:
            x = x.unflatten(self.x_pos, [self.groups, -1])
            dLdy = dLdy.unflatten(self.w_pos, [self.groups, -1])

        x_pad = torch.constant_pad_nd(x, self.explicit_padding, 0)

        dLdw = generic_conv(
            x_pad,
            dLdy,
            self.stride,
            self.dilation,
            self.xl,
            self.wl,
            self.ol,
            self.explicit_shape,
        ).to(dtype=self.dtype)

        if self.groups != 1:
            dLdw = dLdw.flatten(self.o_pos, self.o_pos + 1)
        return dLdw


class ConvBackwardBiasCustomGeneric(torch.nn.Module):
    def __init__(self, sig: ConvSignature):
        super().__init__()
        self.bias_sizes = sig.kernel_shape[sig.kernel_layout.find("N")]
        self.reduction_dims = [
            idx for idx, char in enumerate(sig.output_layout) if char != "C"
        ]

    def forward(
        self, dLdy: torch.Tensor, x: torch.Tensor, w: torch.Tensor
    ) -> torch.Tensor:
        return torch.sum(dLdy, self.reduction_dims)


class ConvBackward(torch.nn.Module):
    def __init__(self, sig: ConvSignature):
        super().__init__()
        self.perms = {
            "dLdy": sig.output_perms.inv(),
            "x": sig.input_perms,
            "w": sig.kernel_perms,
            "dLdx": sig.input_perms.inv(),
            "dLdw": sig.kernel_perms.inv(),
        }
        # remainder from forward output_shape calculation needs to be accounted for
        ker_shape_p = sig.kernel_perms(sig.kernel_shape)
        # get arguments for substitute conv
        self.bias_sizes = [ker_shape_p[0]]
        self.stride = sig.stride
        self.padding = sig.padding
        self.dilation = sig.dilation
        self.transposed = sig.transposed
        self.output_padding = sig.output_padding
        self.groups = sig.groups
        self.mask = sig.backward_mask

    def forward(
        self, dLdy: torch.Tensor, x: torch.Tensor, w: torch.Tensor
    ) -> tuple[torch.Tensor, ...] | torch.Tensor:
        dLdy = self.perms["dLdy"](dLdy)
        x = self.perms["x"](x)
        w = self.perms["w"](w)
        dLdx, dLdw, dLdb = torch.ops.aten.convolution_backward(
            dLdy,
            x,
            w,
            self.bias_sizes,
            self.stride,
            self.padding,
            self.dilation,
            self.transposed,
            self.output_padding,
            self.groups,
            self.mask,
        )
        grads = (
            None if dLdx is None else self.perms["dLdx"](dLdx),
            None if dLdw is None else self.perms["dLdw"](dLdw),
            dLdb,
        )

        rets = tuple(g for g, m in zip(grads, self.mask) if m)
        if len(rets) == 1:
            return rets[0]
        return rets


class ConvCustomBackward(torch.nn.Module):
    def __init__(self, sig: ConvSignature):
        super().__init__()
        self.grad_modules = [
            ConvBackwardInputCustomGeneric(sig),
            ConvBackwardWeightCustomGeneric(sig),
            ConvBackwardBiasCustomGeneric(sig),
        ]
        self.mask = sig.backward_mask

    def forward(
        self, dLdy: torch.Tensor, x: torch.Tensor, w: torch.Tensor
    ) -> tuple[torch.Tensor, ...] | torch.Tensor:
        grads = (
            self.grad_modules[0].forward(dLdy, x, w) if self.mask[0] else None,
            self.grad_modules[1].forward(dLdy, x, w) if self.mask[1] else None,
            self.grad_modules[2].forward(dLdy, x, w) if self.mask[2] else None,
        )
        rets = tuple(g for g, m in zip(grads, self.mask) if m)
        if len(rets) == 1:
            return rets[0]
        return rets


class ConvParser(OpCLIParser):
    @classmethod
    def get_op_name(cls) -> str:
        return "conv"

    @staticmethod
    def get_signature(args) -> ConvSignature:
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

        match args.forw:
            case 1:
                mode = Mode.FORWARD
            case 2:
                mode = Mode.INPUT_BACKWARD
            case 4:
                mode = Mode.WEIGHT_BACKWARD
            case 6:
                mode = Mode.INPUT_WEIGHT_BACKWARD
            case 7:
                mode = Mode.BIAS_BACKWARD
            case 8:
                mode = Mode.INPUT_BIAS_BACKWARD
            case 9:
                mode = Mode.WEIGHT_BIAS_BACKWARD
            case 10:
                mode = Mode.ALL_BACKWARD
            case _:
                raise NotImplementedError(
                    f"Mixed forward and backward kernels unsupported. Got {args.forw = }. Unsupported values = [3: fwd+bwd, 5: fwd+wrw]."
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

    def get_miopen_parser() -> argparse.ArgumentParser:
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
            "--batchsize",
            "-n",
            type=int,
            default=100,
            help="Mini-batch size (Default=100)",
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
            "--pad_d",
            "-$",
            type=int,
            default=0,
            help="Zero Padding for Depth (Default=0)",
        )
        parser.add_argument(
            "--pad_h",
            "-p",
            type=int,
            default=0,
            help="Zero Padding for Height (Default=0)",
        )
        parser.add_argument(
            "--pad_w",
            "-q",
            type=int,
            default=0,
            help="Zero Padding for Width (Default=0)",
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
            "--group_count",
            "-g",
            type=int,
            default=1,
            help="Number of Groups (Default=1)",
        )
        return parser
