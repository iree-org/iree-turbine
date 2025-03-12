from typing import (
    List,
    NamedTuple,
    Optional,
    Union,
)

from enum import IntEnum

import torch

from iree.turbine.boo.conv_exports.utils import Permutation

__all__ = [
    "Mode",
    "ConvSignature",
    "ConvForward",
    "ConvBackwardWeight",
    "ConvBackwardInput",
    "get_nn_module",
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
                f"For import_phase= argument, expected one of: "
                f"{', '.join(Mode.__members__.keys())}"
            )
        return Mode[spec]

    def __str__(self):
        return self.name


DEFAULT_LAYOUTS = {1: "NCH", 2: "NCHW", 3: "NCDHW"}


class ConvSignature(NamedTuple):
    """A named tuple specifying some convolution configuration. Includes helper methods for getting useful information."""

    input_shape: List[int]
    kernel_shape: List[int]
    num_spatial_dims: int = 2
    dtype: torch.dtype = torch.bfloat16
    input_layout: str = DEFAULT_LAYOUTS[num_spatial_dims]
    kernel_layout: str = DEFAULT_LAYOUTS[num_spatial_dims]
    output_layout: str = DEFAULT_LAYOUTS[num_spatial_dims]
    bias: bool = False
    stride: List[int] = num_spatial_dims * [1]
    padding: List[int] = num_spatial_dims * [0]
    dilation: List[int] = num_spatial_dims * [1]
    transposed: bool = False
    output_padding: List[int] = num_spatial_dims * [0]
    groups: int = 1
    mode: Mode = Mode.FORWARD

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

    @staticmethod
    def get(
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> "ConvSignature":
        """gets a signature from provided input, weight, bias tensors and additional kwargs"""
        return ConvSignature(
            input_shape=input.shape,
            kernel_shape=weight.shape,
            num_spatial_dims=len(input.shape) - 2,
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

    def get_sample_conv_args(self):
        """Gets example args for the convolution (mode-dependent)"""
        out_channels = self.kernel_shape[self.kernel_perms[0]]
        x = torch.randn(self.input_shape, dtype=self.dtype)
        w = torch.randn(self.kernel_shape, dtype=self.dtype)
        b = torch.randn(out_channels, dtype=self.dtype)
        if self.mode == Mode.FORWARD:
            return (x, w, b) if self.bias else (x, w)
        dLdy = torch.randn(self.output_shape, dtype=self.dtype)
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


class ConvForward(torch.nn.Module):
    def __init__(self, sig: ConvSignature):
        super().__init__()
        self.perms = [
            sig.input_perms.items,
            sig.kernel_perms.items,
            sig.output_perms.items,
        ]
        self.kwargs = sig.get_conv_kwargs()
        if not sig.bias:
            self.kwargs["bias"] = None

    def forward(self, *args):
        mod_args = [
            torch.permute(args[0], self.perms[0]),
            torch.permute(args[1], self.perms[1]),
        ]
        if "bias" not in self.kwargs.keys():
            mod_args.append(args[2])
        output = torch.convolution(*mod_args, **self.kwargs)
        return torch.permute(output, self.perms[2])


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
            sig.output_perms.inv().items,
            sig.kernel_perms.items,
            sig.input_perms.inv().items,
        ]
        pad_correction = []
        for i in range(sig.num_spatial_dims):
            in_dim = sig.input_shape[sig.input_perms[i + 2]]
            ker_dim = sig.kernel_shape[sig.kernel_perms[i + 2]]
            pad_correction.append(
                ((in_dim - 1) + 2 * sig.padding[i] - sig.dilation[i] * (ker_dim - 1))
                % sig.stride[i]
            )
        # get arguments for substitute conv
        self.kwargs = sig.get_conv_kwargs()
        self.kwargs["transposed"] = True
        self.kwargs["output_padding"] = pad_correction

    def forward(self, dLdy, w):
        dLdy = torch.permute(dLdy, self.perms[0])
        w = torch.permute(w, self.perms[1])
        dLdx = torch.convolution(
            dLdy,
            w,
            bias=None,
            **self.kwargs,
        )
        return torch.permute(dLdx, self.perms[2])


class ConvBackwardWeight(torch.nn.Module):
    def __init__(self, sig: ConvSignature):
        super().__init__()
        # TODO: support grouped input_grad
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
                "unimplemented input grad decompostion: groups != 1"
            )
        if sig.transposed:
            raise NotImplementedError(
                "unimplemented input grad decomposition: transposed conv"
            )
        self.ND = sig.num_spatial_dims
        self.K = sig.kernel_shape[-self.ND :]
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
            (NC_perm * sig.input_perms).items,
            (NC_perm / sig.output_perms).items,
            (sig.kernel_perms.inv() * NC_perm).items,
        ]

    def forward(self, dLdy, x):
        # Slice out unneccessary values after convolution.
        # One can instead pre-pad spatial dims:
        #  1. x by (stride - pad_correction)
        #  2. dLdy by 1
        conv = torch.convolution(
            torch.permute(x, self.perms[0]),
            torch.permute(dLdy, self.perms[1]),
            bias=None,
            **self.kwargs,
        )

        if self.ND == 1:
            sliced = conv[..., : self.K[0]]
        if self.ND == 2:
            sliced = conv[..., : self.K[0], : self.K[1]]
        if self.ND == 3:
            sliced = conv[..., : self.K[0], : self.K[1], : self.K[2]]

        return torch.permute(sliced, self.perms[2])


def get_nn_module(signature: ConvSignature):
    """For a given ConvSignature, returns a torch.nn.Module implementation."""
    if signature.mode == Mode.FORWARD:
        return ConvForward(signature)
    if signature.mode == Mode.WEIGHT_BACKWARD:
        return ConvBackwardWeight(signature)
    if signature.mode == Mode.INPUT_BACKWARD:
        return ConvBackwardInput(signature)
    raise ValueError(f"Signature has unexpected mode: {signature.mode}")
