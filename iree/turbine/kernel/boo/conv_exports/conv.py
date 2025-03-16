# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import (
    List,
    NamedTuple,
    Optional,
    Union,
)

from enum import IntEnum

import torch

from .utils import Permutation

__all__ = [
    "Mode",
    "ConvSignature",
    "ConvForward",
    "ConvBackwardWeight",
    "ConvBackwardInput",
]


class Mode(IntEnum):
    FORWARD = 0
    INPUT_BACKWARD = 1
    WEIGHT_BACKWARD = 2

    # alias values
    FWD = FORWARD
    BWD = INPUT_BACKWARD
    WRW = WEIGHT_BACKWARD

    @staticmethod
    def parse(spec: Union[str, None, "Mode"]) -> "Mode":
        if spec is None:
            return Mode.FORWARD
        if isinstance(spec, Mode):
            return spec
        spec = spec.upper().replace("-", "_")
        if spec not in Mode.__members__:
            raise ValueError(
                f"For mode= argument, expected one of: "
                f"{', '.join(Mode.__members__.keys())}"
            )
        return Mode[spec]

    def __str__(self):
        return self.name


DEFAULT_LAYOUTS = {1: "NCH", 2: "NCHW", 3: "NCDHW"}


class ConvSignatureStorage(NamedTuple):
    """A named tuple specifying some convolution configuration."""

    input_shape: List[int]
    kernel_shape: List[int]
    num_spatial_dims: int
    dtype: torch.dtype
    input_layout: str
    kernel_layout: str
    output_layout: str
    bias: bool
    stride: List[int]
    padding: List[int]
    dilation: List[int]
    transposed: bool
    output_padding: List[int]
    groups: int
    mode: Mode


class ConvSignature:
    """
    Wraps ConvSignatureStorage with some additional helper methods and easier instantiation.

    """

    def __init__(
        self,
        *,
        input_shape: List[int],
        kernel_shape: List[int],
        shared_layout: Optional[str] = None,
        input_layout: Optional[str] = None,
        kernel_layout: Optional[str] = None,
        output_layout: Optional[str] = None,
        bias: bool = False,
        dtype: torch.dtype = torch.bfloat16,
        stride: Union[int, List[int]] = 1,
        padding: Union[int, List[int]] = 0,
        dilation: Union[int, List[int]] = 1,
        transposed: bool = False,
        output_padding: Union[int, List[int]] = 0,
        groups: int = 1,
        mode: Union[str, Mode] = Mode.FORWARD,
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

        def listify(value: int | List[int]) -> List[int]:
            if isinstance(value, int):
                return [value] * num_spatial_dims
            if isinstance(value, list):
                assert len(value) == num_spatial_dims
                return value

        if isinstance(mode, str):
            mode = Mode.parse(mode)
        self._signature = ConvSignatureStorage(
            input_shape=input_shape,
            kernel_shape=kernel_shape,
            num_spatial_dims=num_spatial_dims,
            dtype=dtype,
            input_layout=get_layout(input_layout),
            kernel_layout=get_layout(kernel_layout),
            output_layout=get_layout(output_layout),
            bias=bias,
            stride=listify(stride),
            padding=listify(padding),
            dilation=listify(dilation),
            output_padding=listify(output_padding),
            transposed=transposed,
            groups=groups,
            mode=mode,
        )

    def __getattr__(self, name):
        # forward stored signature attributes
        return self._signature.__getattribute__(name)

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
    def output_shape(self) -> List:
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
    def explicit_padding(self) -> List[int]:
        """Padding of input tensor compatible with torch.constant_pad_nd."""
        torch_pads_NCHW = [[0, 0], [0, 0]] + [[p, p] for p in self.padding]
        # permute back to input ordering
        permuted_pads = self.input_perms.inv()(torch_pads_NCHW)
        # to make compatible with torch.nn.functional.pad reverse ordering
        permuted_pads.reverse()
        # flatten the list
        return [p for dim_pads in permuted_pads for p in dim_pads]

    @staticmethod
    def get(
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> "ConvSignature":
        """gets a signature from provided input, weight, bias tensors and additional kwargs"""
        return ConvSignature(
            input_shape=list(input.shape),
            kernel_shape=list(weight.shape),
            dtype=input.dtype,
            bias=bool(bias),
            **kwargs,
        )

    def get_conv_kwargs(self):
        """Gets `torch.convolution` (forward-only) kwargs from the signature."""
        conv_extra_args = [
            "stride",
            "padding",
            "dilation",
            "transposed",
            "output_padding",
            "groups",
        ]
        return {name: self._asdict()[name] for name in conv_extra_args}

    def get_sample_conv_args(self, *, splat_value=None, seed: Optional[int] = None):
        """Gets example args for the convolution (mode-dependent)"""
        out_channels = self.kernel_shape[self.kernel_perms[0]]
        if splat_value:
            x = torch.ones(self.input_shape, dtype=self.dtype) * splat_value
            w = torch.ones(self.kernel_shape, dtype=self.dtype) * splat_value
            b = torch.ones(out_channels, dtype=self.dtype) * splat_value
        else:
            gen = None if not seed else torch.random.manual_seed(seed)
            x = torch.randn(self.input_shape, generator=gen, dtype=self.dtype)
            w = torch.randn(self.kernel_shape, generator=gen, dtype=self.dtype)
            b = torch.randn(out_channels, generator=gen, dtype=self.dtype)
        if self.mode == Mode.FORWARD:
            return (x, w, b) if self.bias else (x, w)
        dLdy = torch.randn(self.output_shape, generator=gen, dtype=self.dtype)
        if self.mode == Mode.WEIGHT_BACKWARD:
            return (dLdy, x)
        if self.mode == Mode.INPUT_BACKWARD:
            return (dLdy, w)

    def get_func_name(self):
        name_items = [
            "conv",
            f"{self.num_spatial_dims}d",
            str(self.dtype).removeprefix("torch."),
            str(self.mode).lower(),
        ]
        l2s = lambda l: "x".join([str(i) for i in l])
        name_items.extend(
            [
                l2s(self.input_shape),
                l2s(self.kernel_shape),
                l2s(self.stride) + "s",
                l2s(self.padding) + "p",
                l2s(self.dilation) + "d",
                f"{self.groups}g",
            ]
        )
        return "_".join(name_items)

    def get_nn_module(self) -> torch.nn.Module:
        """For a given ConvSignature, returns a torch.nn.Module implementation."""
        if self.mode == Mode.FORWARD:
            return ConvForward(self)
        if self.mode == Mode.WEIGHT_BACKWARD:
            return ConvBackwardWeight(self)
        if self.mode == Mode.INPUT_BACKWARD:
            return ConvBackwardInput(self)
        raise ValueError(f"signature has unexpected mode: {self.mode}")


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

    def forward(self, *args):
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

    def forward(self, dLdy, w):
        dLdy = self.perms[0](dLdy)
        w = self.perms[1](w)
        dLdx = torch.convolution(
            dLdy,
            w,
            bias=None,
            **self.kwargs,
        )
        return self.perms[2](dLdx)


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

    def forward(self, dLdy, x):
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
