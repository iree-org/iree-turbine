# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
from typing import (
    Any,
    Sequence,
    TypeVar,
)
from collections.abc import Collection
from enum import IntEnum
from functools import lru_cache
import math
import warnings

import torch

from ..exports.signature import OpSignature, ModeBase
from ..exports.parser import OpCLIParser
from ....ops.conv_fwd import conv_2d_nhwc_fhwc, generic_conv
from ....ops.insert_slice import insert_slice

__all__ = [
    "Mode",
    "ConvParser",
    "ConvSignature",
    "ConvForward",
    "ConvBackwardWeight",
    "ConvBackwardInput",
    "DEFAULT_LAYOUTS",
    "get_conv_func_name",
]

_T = TypeVar("_T")


class Permutation:
    """Composable and invertible lists which represent the second argument of `torch.permute`."""

    def __init__(self, ordering: Sequence[int]):
        assert list(sorted(ordering)) == list(
            range(len(ordering))
        ), "ordering must be rearragement of [0,1,2,...,n-1]"
        self._items = tuple(ordering)

    @property
    def size(self) -> int:
        return len(self._items)

    @property
    def items(self) -> tuple[int, ...]:
        return self._items

    def __getitem__(self, n: int) -> int:
        return self.items[n]

    def __repr__(self) -> str:
        return f"Permutation of {self.size} : {self.items}"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Permutation):
            return False
        return self.items == other.items

    def __len__(self) -> int:
        return self.size

    def __mul__(self, other: "Permutation") -> "Permutation":
        """mimics composition `torch.permute(torch.permute(a, p1), p0) = torch.permute(a, p0*p1)"""
        assert self.size == other.size, "permutations must be the same size"
        return Permutation([other.items[element] for element in self.items])

    def __call__(self, other: torch.Tensor | Collection[_T]) -> torch.Tensor | list[_T]:
        """apply the permutation to a tensor or iterable (e.g., a shape)"""
        if isinstance(other, torch.Tensor):
            assert (
                len(other.shape) == self.size
            ), f"permutation must match the rank of the tensor being permuted, got permutation size {self.size} for tensor of shape {other.shape}"
            return torch.permute(other, self.items)
        if isinstance(other, Collection):
            assert len(other) == self.size
            return [other[item] for item in self.items]
        raise TypeError(f"Unexpected argument type: {type(other)}.")

    def __truediv__(self, other: "Permutation") -> "Permutation":
        return self * other.inv()

    def inv(self) -> "Permutation":
        """inverts the permutation x*inv(x) = inv(x)*x = Permutation.identity(x.size)"""
        inverse = list(range(self.size))
        for i in range(self.size):
            index = self.items[i]
            inverse[index] = i
        return Permutation(inverse)

    @staticmethod
    def identity(size: int) -> "Permutation":
        """creates an identity permutation"""
        assert size > 0, "size must be positive integer"
        return Permutation(list(range(size)))

    @staticmethod
    def get(src: Collection[_T], target: Collection[_T]) -> "Permutation":
        """Gets a permutation p such that `torch.permute(a, p) = b` where `a.shape = src` and `b.shape = target`"""
        n = len(src)
        assert n > 0 and n == len(
            target
        ), "source and target iterables must share the same positive length"
        d = {t: i for i, t in enumerate(target)}
        inverse = []
        try:
            for item in src:
                value = d.pop(item)
                inverse.append(value)
        except KeyError as e:
            raise ValueError(
                f"src and target should be permutations of a common set of unique items, got {src=}, {target=}"
            )
        return Permutation(inverse).inv()


class Mode(ModeBase, IntEnum):
    FORWARD = 0
    INPUT_BACKWARD = 1
    WEIGHT_BACKWARD = 2

    # alias values
    FWD = FORWARD
    BWD = INPUT_BACKWARD
    WRW = WEIGHT_BACKWARD

    def __str__(self) -> str:
        return self.name


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

        if isinstance(mode, str):
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

    def make_signature_copy_for_forward(self) -> "ConvSignature":
        kwargs = self.as_init_kwargs()
        kwargs["mode"] = "fwd"
        return ConvSignature(**kwargs)

    def get_arg_index_for_backward(self) -> int | None:
        assert not self.is_forward
        # TODO: should we just use the index of the argument we are computing
        # the derivative of as mode and -1 as forward instead?
        match self.mode:
            case Mode.INPUT_BACKWARD:
                return 0
            case Mode.WEIGHT_BACKWARD:
                return 1
            case _:
                return None

    def arrange_backward_launch_args(
        self,
        forward_args: tuple[torch.Tensor, ...],
        forward_results: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, ...]:
        assert not self.is_forward
        x, w, *_ = forward_args
        if self.mode == Mode.INPUT_BACKWARD:
            return (w,)
        if self.mode == Mode.WEIGHT_BACKWARD:
            return (x,)
        raise ValueError(f"Unsupported mode {self.mode}")

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
        if self.mode == Mode.WEIGHT_BACKWARD:
            # (dLdy, x)
            return (get(self.output_shape), get(self.input_shape))
        if self.mode == Mode.INPUT_BACKWARD:
            # (dLdy, w)
            return (get(self.output_shape), get(self.kernel_shape))
        raise ValueError(f"Unknown mode: {self.mode}")

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

    def get_nn_module(self, *, use_custom: bool = False) -> torch.nn.Module:
        """For a given ConvSignature, returns a torch.nn.Module implementation."""
        if self.mode == Mode.WEIGHT_BACKWARD:
            return (
                ConvBackwardWeightCustomGeneric(self)
                if use_custom
                else ConvBackwardWeight(self)
            )
        if self.mode == Mode.INPUT_BACKWARD:
            return (
                ConvBackwardInputCustomGeneric(self)
                if use_custom
                else ConvBackwardInput(self)
            )
        if self.mode == Mode.FORWARD:
            return (
                ConvForwardCustomGeneric(self)
                if use_custom and not self.transposed
                else ConvForward(self)
            )
        raise ValueError(f"signature has unexpected mode: {self.mode}")

    def get_output_size(self) -> int:
        numel = 0
        if int(self.mode) == 0:
            numel = math.prod(self.output_shape)
        elif int(self.mode) == 1:
            numel = math.prod(self.input_shape)
        elif int(self.mode) == 2:
            numel = math.prod(self.kernel_shape)
        dtype_bytes = int(self.dtype.itemsize)
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
        self.explicit_padding = sig.explicit_padding
        self.kwargs["padding"] = [0] * sig.num_spatial_dims

    def forward(self, *args: torch.Tensor) -> torch.Tensor:
        mod_args = [
            self.perms[0](
                torch.constant_pad_nd(args[0], self.explicit_padding, value=0)
            ),
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


class ConvBackwardInput(torch.nn.Module):
    def __init__(self, sig: ConvSignature):
        super().__init__()
        # TODO: Unblock when torch-mlir fix for grouped tranpose convolution lands
        if sig.groups != 1:
            raise NotImplementedError(
                "unimplemented input grad decompostion: groups != 1"
            )
        if sig.transposed:
            raise NotImplementedError(
                "unimplemented input grad decomposition: transposed conv"
            )
        self.perms = [
            sig.output_perms.inv(),
            sig.kernel_perms,
            sig.input_perms.inv(),
        ]
        # remainder from forward output_shape calculation needs to be accounted for
        pad_correction = []
        in_shape_p = sig.input_perms(sig.input_shape)
        ker_shape_p = sig.kernel_perms(sig.kernel_shape)
        for i in range(sig.num_spatial_dims):
            pad_correction.append(
                (
                    (in_shape_p[i + 2] - 1)
                    + 2 * sig.padding[i]
                    - sig.dilation[i] * (ker_shape_p[i + 2] - 1)
                )
                % sig.stride[i]
            )
        # get arguments for substitute conv
        self.kwargs = sig.get_conv_kwargs()
        self.kwargs["transposed"] = True
        self.kwargs["output_padding"] = pad_correction

    def forward(self, dLdy: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        dLdy = self.perms[0](dLdy)
        w = self.perms[1](w)
        dLdx = torch.convolution(
            dLdy,
            w,
            bias=None,
            **self.kwargs,
        )
        return self.perms[2](dLdx)


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

    def forward(self, dLdy: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
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


class ConvBackwardWeight(torch.nn.Module):
    def __init__(self, sig: ConvSignature):
        super().__init__()
        # TODO: support grouped weight_grad
        # Note: expanding the weight shape to g x Cout//g x Cin//g x K
        # dLdw[gidx, co, ci, k] = sum_n sum_hout x[n, gidx, ci, dil*k + s*hout]* dLdy[n, gidx, co, hout]
        # The sum is over N, so this convolution-like op should have group=1, and the "batch-dim"
        # should be Cin, since it is shared by `x` and `dLdw`; however, dLdy only gets used
        # at the same gidx for Cout, which adds some redundancy if we perform this as a conv.
        # i.e., Z = conv{g=1}(x.T, dLdy.T).T has shape CoutxCinxK, but needs to be Coutx(Cin//g)xK.
        # dLdw[gidx, co,ci,k] = Z[gidx*(Cout//g) + co, gidx*(Cin//g) + ci,k].
        # Reshaping Z to (Cout//g) x g x g x (Cin//g) x K, this is essentially taking a diagonal slice
        # over the (g x g) dims.
        if sig.groups != 1:
            raise NotImplementedError(
                "unimplemented weight grad decompostion: groups != 1"
            )
        if sig.transposed:
            raise NotImplementedError(
                "unimplemented weight grad decomposition: transposed conv"
            )
        self.ND = sig.num_spatial_dims
        self.K = sig.kernel_perms(sig.kernel_shape)[-self.ND :]
        self.kwargs = {
            "stride": sig.dilation,
            "padding": sig.padding,
            "dilation": sig.stride,
            "transposed": False,
            "output_padding": sig.num_spatial_dims * (0,),
            # can't group N
            "groups": 1,
        }
        # need to permute layouts further for weight grad:
        # 1. x (NCHW) to (CNHW)
        #    already have in_perm : (in_layout) -> (NCHW)
        #    so compose (1,0,2,3) o in_perm : (in_layout) -> (CNHW)
        # 2. dLdy (NCoutHW) to (CoutNHW)
        #    already have out_perm : (NCoutHW) -> (out_layout)
        #    so compose (1,0,2,3) o out_perm^{-1} : (out_layout) -> (CoutNHW)
        # 3. dLdw (CinCoutHW) back to (CoutCinHW)
        #    already have kernel_perm : (fil_layout) -> (CoutCinHW)
        #    so pre-compose kernel_perm^{-1} o (1,0,2,3) : (CinCoutHW) -> (fil_layout)
        #    note: (1,0,2,3) is it's own inverse.
        NC_perm = Permutation([1, 0] + list(range(2, sig.num_spatial_dims + 2)))
        self.perms = [
            NC_perm * sig.input_perms,
            NC_perm / sig.output_perms,
            sig.kernel_perms.inv() * NC_perm,
        ]
        self.explicit_padding = sig.explicit_padding
        self.kwargs["padding"] = sig.num_spatial_dims * [0]

    def forward(self, dLdy: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        x = torch.constant_pad_nd(x, self.explicit_padding, 0)
        conv = torch.convolution(
            self.perms[0](x),
            self.perms[1](dLdy),
            bias=None,
            **self.kwargs,
        )

        # The forward conv's output_shape calculation is subtractive w.r.t. kernel_shape.
        # Therefore, we need to remove unneccessary values from the backward conv.
        # We choose to slice them out after the conv in this impl.
        # One could instead pre-pad spatial dims:
        #  1. x by stride - pad_correction (see ConvBackwardInput)
        #  2. dLdy by 1
        if self.ND == 1:
            sliced = conv[..., : self.K[0]]
        if self.ND == 2:
            sliced = conv[..., : self.K[0], : self.K[1]]
        if self.ND == 3:
            sliced = conv[..., : self.K[0], : self.K[1], : self.K[2]]

        return self.perms[2](sliced)


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

    def forward(self, dLdy: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
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


class ConvParser(OpCLIParser):
    @classmethod
    def get_op_name(self) -> str:
        return "conv"

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
