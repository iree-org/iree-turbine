# generate sample kernels corresponding to MiOpen conv signatures

from pathlib import Path
import re
import gc

from typing import (
    Dict,
    NamedTuple,
    List,
    Union,
    Optional,
)

import torch

from iree.compiler.extras.fx_importer import FxImporter
from iree.compiler.passmanager import PassManager

from iree.turbine.boo.conv_exports.alias import get_aliases_and_defaults

ALIAS_MAP, DEFAULT_MAP = get_aliases_and_defaults()


class InputConvSignature(NamedTuple):
    num_spatial_dims: int
    dtype: torch.dtype
    input_perms: List[int]
    kernel_perms: List[int]
    output_perms: List[int]
    input_shape: List[int]
    kernel_shape: List[int]
    bias: bool
    stride: List[int]
    padding: List[int]
    dilation: List[int]
    transposed: bool
    output_padding: List[int]
    groups: int
    forward: bool
    input_backward: bool
    weight_backward: bool


class ConvForward(torch.nn.Module):
    def __init__(self, sig: InputConvSignature):
        super().__init__()
        self.perms = [sig.input_perms, sig.kernel_perms, sig.output_perms]
        self.kwargs = get_conv_kwargs(sig)
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
    def __init__(self, sig: InputConvSignature):
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
        self.perms = [inv(sig.output_perms), sig.kernel_perms, inv(sig.input_perms)]
        pad_correction = []
        for i in range(sig.num_spatial_dims):
            in_dim = sig.input_shape[sig.input_perms[i + 2]]
            ker_dim = sig.kernel_shape[sig.kernel_perms[i + 2]]
            pad_correction.append(
                ((in_dim - 1) + 2 * sig.padding[i] - sig.dilation[i] * (ker_dim - 1))
                % sig.stride[i]
            )
        # get arguments for substitute conv
        self.kwargs = get_conv_kwargs(sig)
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
    def __init__(self, sig: InputConvSignature):
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
        NC_perm = [1, 0]
        for i in range(sig.num_spatial_dims):
            NC_perm.append(i + 2)
        self.perms = [
            compose(NC_perm, sig.input_perms),
            compose(NC_perm, inv(sig.output_perms)),
            compose(inv(sig.kernel_perms), NC_perm),
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


def compose(p0, p1):
    """mimics composition `torch.permute(torch.permute(a, p1), p0) = torch.permute(a, compose(p0,p1))"""
    assert len(p0) == len(p1), "permutations must be the same length"
    # note: p0[i] is the source dim for p0(T).shape[i]
    # i.e., T.shape[p0[i]] = p0(T).shape[i]
    # Therefore, T.shape[p1[p0[i]]] = p1(T).shape[p0[i]] = p0(p1(T)).shape[i]
    # All to say: the ordering here is correct.
    return [p1[p0[i]] for i in range(len(p0))]


def get_permutation(old, new):
    """Solves for `p` in `torch.permute(a, p) = b`, where `a.shape = old` and `b.shape = new`."""
    n = len(old)
    perms = []
    for val in new:
        for j in range(n):
            if old[j] == val:
                perms.append(j)
    if len(perms) != n:
        raise ValueError(
            f"Invalid provided iterables: {old} and {new} are not permuatations."
        )
    return perms


def inv(perm):
    identity = [i for i in range(len(perm))]
    return get_permutation(perm, identity)


def get_conv_kwargs(signature: InputConvSignature):
    conv_extra_args = [
        "stride",
        "padding",
        "dilation",
        "transposed",
        "output_padding",
        "groups",
    ]
    return {name: signature._asdict()[name] for name in conv_extra_args}


def get_sample_conv_args(sig: InputConvSignature):
    x = torch.randn(sig.input_shape, dtype=sig.dtype)
    w = torch.randn(sig.kernel_shape, dtype=sig.dtype)
    b = torch.randn(sig.kernel_shape[0], dtype=sig.dtype)
    if sig.forward:
        return (x, w, b) if sig.bias else (x, w)
    output_shape = []
    for i in range(len(sig.input_shape)):
        if i == 0:
            output_shape.append(sig.input_shape[sig.input_perms[0]])
        elif i == 1:
            output_shape.append(sig.kernel_shape[sig.kernel_perms[0]])
        else:
            in_dim = sig.input_shape[sig.input_perms[i]]
            ker_dim = sig.kernel_shape[sig.kernel_perms[i]]
            output_shape.append(
                (
                    (in_dim - 1)
                    + 2 * sig.padding[i - 2]
                    - sig.dilation[i - 2] * (ker_dim - 1)
                )
                // sig.stride[i - 2]
                + 1,
            )
    permuted_shape = compose(sig.output_perms, output_shape)
    dLdy = torch.randn(permuted_shape, dtype=sig.dtype)
    if sig.weight_backward:
        return (dLdy, x)
    if sig.input_backward:
        return (dLdy, w)


def get_nn_module(signature: InputConvSignature):
    if not (signature.forward ^ signature.input_backward ^ signature.weight_backward):
        raise NotImplementedError(
            "Currently only support generating IR for exactly one specification (fwd, wrw, bwd)."
        )
    if signature.forward:
        m = ConvForward(signature)
    elif signature.weight_backward:
        m = ConvBackwardWeight(signature)
    elif signature.input_backward:
        m = ConvBackwardInput(signature)
    else:
        assert False, "unreachable configuration"
    return m


def get_func_name(signature: InputConvSignature):
    name_items = [
        "conv",
        f"{signature.num_spatial_dims}d",
        str(signature.dtype).removeprefix("torch."),
    ]
    if signature.forward:
        name_items.append("fwd")
    if signature.weight_backward:
        name_items.append("wrw")
    if signature.input_backward:
        name_items.append("bwd")
    if signature.transposed:
        name_items.append("tr")
    l2s = lambda s0, l: s0.join([str(i) for i in l])
    name_items.extend(
        [
            l2s("x", signature.input_shape),
            l2s("x", signature.kernel_shape),
            l2s("x", signature.stride) + "s",
            l2s("x", signature.padding) + "p",
            l2s("x", signature.dilation) + "d",
            f"{signature.groups}g",
        ]
    )
    return "_".join(name_items)


def generate_mlir(
    signature: InputConvSignature, output_path: Optional[Union[str, Path]] = None
):
    args = get_sample_conv_args(signature)
    # func_name = get_func_name(signature)
    m = get_nn_module(signature)
    e = torch.export.export(m, args=args)
    importer = FxImporter()
    importer.import_program(e, func_name=get_func_name(signature))
    pm = PassManager.parse(
        "builtin.module(torch-backend-to-linalg-on-tensors-backend-pipeline)",
        context=importer.module.context,
    )
    pm.run(importer.module.operation)
    if output_path:
        Path(output_path).write_text(str(importer.module))
        return
    print(importer.module)


def batch_generate_mlir(signatures: Dict[str, InputConvSignature], save_dir: Path):
    save_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    err = 0
    for name, s in signatures.items():
        print(f"processing {name}...")
        path = save_dir / f"{name}.mlir"
        total += 1
        try:
            generate_mlir(s, path)
        except Exception as e:
            err += 1
            path = save_dir / f"ERR_{name}.log"
            path.write_text(f"signature = {s}\n{str(e)}")
        gc.collect()
    return err, total


def filter_signatures(signatures: Dict[str, InputConvSignature], **kwargs):
    filtered = {}
    for name, sig in signatures.items():
        match = True
        for k, v in kwargs.items():
            if sig._asdict()[k] != v:
                match = False
        if match:
            filtered[name] = sig
    return filtered


def command_to_signature(command: str, ignore_layouts: bool = False):
    comm_list = command.split(" ")

    def find(flag, *, default=None):
        for (i, item) in enumerate(comm_list):
            if flag == item or ALIAS_MAP[flag] == item:
                try:
                    return comm_list[i + 1]
                except IndexError:
                    pass
        return default

    in_layout = find("-I")
    fil_layout = find("-f")
    out_layout = find("-O")

    assert in_layout is not None
    assert fil_layout is not None
    assert out_layout is not None

    n = len(in_layout) - 2

    pytorch_layout = [
        "NCH",
        "NCHW",
        "NCDHW",
    ][n - 1]

    if ignore_layouts:
        in_layout = pytorch_layout
        fil_layout = pytorch_layout
        out_layout = pytorch_layout

    in_perms = get_permutation(in_layout, pytorch_layout)
    kernel_perms = get_permutation(fil_layout, pytorch_layout)
    out_perms = get_permutation(pytorch_layout, out_layout)

    batch = find("-n")
    assert batch is not None
    in_channels = find("-c")
    assert in_channels is not None
    groups = find("-g")
    assert groups is not None
    out_channels = find("-k")
    assert out_channels is not None

    in_dims = {
        "N": batch,
        "C": find("-c"),
        "D": find("-!"),
        "H": find("-H"),
        "W": find("-W"),
    }
    w_dims = {
        "N": out_channels,
        "C": int(in_channels) // int(groups),
        "D": find("-@"),
        "H": find("-y"),
        "W": find("-x"),
    }
    conv_config_dicts = {
        "stride": {
            "D": find("-#"),
            "H": find("-u"),
            "W": find("-v"),
        },
        "padding": {
            "D": find("-$"),
            "H": find("-p"),
            "W": find("-q"),
        },
        "dilation": {
            "D": find("-^"),
            "H": find("-l"),
            "W": find("-j"),
        },
        "output_padding": {
            "D": find("-%", default=0),
            "H": find("-Y", default=0),
            "W": find("-X", default=0),
        },
    }
    in_shape = [int(in_dims[char]) for char in in_layout]
    ker_shape = [int(w_dims[char]) for char in fil_layout]
    bias = find("-b") == "1"
    order = list(set(in_layout).intersection(["D", "H", "W"]))
    order.sort()

    conv_config = {
        "stride": [],
        "padding": [],
        "dilation": [],
        "output_padding": [],
    }
    for dim in order:
        for key in conv_config.keys():
            item = conv_config_dicts[key][dim]
            if item is not None:
                conv_config[key].append(int(item))
    for value in conv_config.values():
        assert len(value) == n

    conv_config["groups"] = int(groups)
    fwd = find("-F")
    conv_config["forward"] = fwd in ["0", "1", "3", "5"]
    conv_config["input_backward"] = fwd in ["0", "2", "3", "6"]
    conv_config["weight_backward"] = fwd in ["0", "4", "5", "6"]
    conv_config["transposed"] = find("-m") == "trans"
    return InputConvSignature(
        num_spatial_dims=n,
        dtype=torch.bfloat16,
        input_perms=in_perms,
        kernel_perms=kernel_perms,
        output_perms=out_perms,
        input_shape=in_shape,
        kernel_shape=ker_shape,
        bias=bias,
        **conv_config,
    )


def load_commands():
    path = Path(__file__).parent / "conv_configs.txt"
    commands = path.read_text().splitlines()
    return commands


def get_safe_name(command: str) -> str:
    name = "".join(command.split())
    return re.sub("-", "_", name)


if __name__ == "__main__":
    # TODO: argparse for filtering
    commands = load_commands()
    signatures = {
        get_safe_name(c): command_to_signature(c, ignore_layouts=True) for c in commands
    }
    filtered = filter_signatures(signatures, forward=True)
    base_dir = Path(__file__).parent / "conv_ir"
    batch_generate_mlir(filtered, base_dir)

# MiOpen args

# --batchsize          -n        Mini-batch size (Default=100)
# --in_channels        -c        Number of Input Channels (Default=3)
# --in_d               -!        Input Depth (Default=32)
# --in_h               -H        Input Height (Default=32)
# --in_w               -W        Input Width (Default=32)
# --in_layout          -I        Input Layout (Default=NCHW for 2d conv, NCDHW for 3d conv)
# --in_cast_type       -U        Cast type for input tensor, default to not set

# --out_channels       -k        Number of Output Channels (Default=32)
# --fil_d              -@        Filter Depth (Default=3)
# --fil_h              -y        Filter Height (Default=3)
# --fil_w              -x        Filter Width (Default=3)
# --fil_layout         -f        Filter Layout (Default=NCHW for 2d conv, NCDHW for 3d conv)
# --wei_cast_type      -R        Cast type for weight tensor, default to not set

# --bias               -b        Use Bias (Default=0)

# --out_layout         -O        Output Layout (Default=NCHW for 2d conv, NCDHW for 3d conv)
# --out_cast_type      -T        Cast type for output tensor, default to not set

# --forw               -F        Flag enables fwd, bwd, wrw convolutions
#                                 0 fwd+bwd+wrw (default)
#                                 1 fwd only
#                                 2 bwd only
#                                 4 wrw only
#                                 3 fwd+bwd
#                                 5 fwd+wrw
#                                 6 bwd+wrw
# --mode               -m        Convolution Mode (conv, trans) (Default=conv)

# --conv_stride_d      -#        Convolution Stride for Depth (Default=1)
# --conv_stride_h      -u        Convolution Stride for Height (Default=1)
# --conv_stride_w      -v        Convolution Stride for Width (Default=1)

# --pad_d              -$        Zero Padding for Depth (Default=0)
# --pad_h              -p        Zero Padding for Height (Default=0)
# --pad_w              -q        Zero Padding for Width (Default=0)
# --pad_val            -r        Padding Value (Default=0)
# --pad_mode           -z        Padding Mode (same, valid, default) (Default=default)

# --dilation_d         -^        Dilation of Filter Depth (Default=1)
# --dilation_h         -l        Dilation of Filter Height (Default=1)
# --dilation_w         -j        Dilation of Filter Width (Default=1)

# --trans_output_pad_d -%        Zero Padding Output for Depth (Default=0)
# --trans_output_pad_h -Y        Zero Padding Output for Height (Default=0)
# --trans_output_pad_w -X        Zero Padding Output for Width (Default=0)

# --group_count        -g        Number of Groups (Default=1)
