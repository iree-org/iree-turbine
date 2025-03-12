# generates sample kernels corresponding to MiOpen conv signatures

from pathlib import Path
import re
import gc

from typing import (
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

import warnings

import torch

from iree.compiler.extras.fx_importer import FxImporter
from iree.compiler.passmanager import PassManager

from iree.turbine.boo.conv_exports.utils import get_aliases_and_defaults
from iree.turbine.boo.conv_exports.conv import get_nn_module, ConvSignature, Mode

ALIAS_MAP, DEFAULT_MAP = get_aliases_and_defaults()

DEFAULT_TORCH_TO_LINALG_PIPELINE = [
    "torch-backend-to-linalg-on-tensors-backend-pipeline"
]


def generate_mlir(
    signature: ConvSignature,
    output_path: Optional[Union[str, Path]] = None,
    *,
    import_pipeline: List[str] = DEFAULT_TORCH_TO_LINALG_PIPELINE,
) -> None:
    """For a given ConvSignature, imports the conv to mlir"""
    args = signature.get_sample_conv_args()
    m = get_nn_module(signature)
    e = torch.export.export(m, args=args)
    importer = FxImporter()
    importer.import_program(e, func_name=signature.get_func_name())
    pipeline_str = "builtin.module(" + ",".join(import_pipeline) + ")"
    pm = PassManager.parse(
        pipeline_str,
        context=importer.module.context,
    )
    pm.run(importer.module.operation)
    if output_path:
        Path(output_path).write_text(str(importer.module))
        return
    print(importer.module)


def batch_generate_mlir(
    signatures: Dict[str, ConvSignature],
    save_dir: Path,
    *,
    import_pipeline: List[str] = DEFAULT_TORCH_TO_LINALG_PIPELINE,
) -> Tuple[int, int]:
    save_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    err = 0
    for name, s in signatures.items():
        print(f"processing {name}...")
        path = save_dir / f"{name}.mlir"
        total += 1
        try:
            generate_mlir(s, path, import_pipeline=import_pipeline)
        except Exception as e:
            err += 1
            path = save_dir / f"ERR_{name}.log"
            path.write_text(f"signature = {s}\n{str(e)}")
        gc.collect()
    return err, total


def filter_signatures(signatures: Dict[str, ConvSignature], **kwargs):
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
    if fwd == "1":
        conv_config["mode"] = Mode.FORWARD
    elif fwd == "2":
        conv_config["mode"] = Mode.INPUT_BACKWARD
    elif fwd == "4":
        conv_config["mode"] = Mode.WEIGHT_BACKWARD
    else:
        warnings.warn(
            f"Only one of fwd, bwd, wrw conv supported at one time. Got {command}."
        )
    conv_config["transposed"] = find("-m") == "trans"
    dtype_dict = {
        "convbfp16": torch.bfloat16,
        "conv": torch.float32,
        "convfp16": torch.float16,
    }
    dtype = dtype_dict[comm_list[0]]
    return ConvSignature(
        num_spatial_dims=n,
        dtype=dtype,
        input_layout=in_layout,
        kernel_layout=fil_layout,
        output_layout=out_layout,
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


import argparse


def _get_argparse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tool for generating MLIR for convolutions matching the signature of the MiOpen driver <https://github.com/ROCm/MIOpen/blob/develop/driver/README.md> for conv."
    )
    parser.add_argument(
        "--all",
        "-a",
        action="store_true",
        default=False,
        help="Imports all configs in `conv_configs.txt`",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        help="Specify relative path from cwd to store output mlir.",
    )
    parser.add_argument(
        "--forw",
        "-F",
        choices=["fwd", "bwd", "wrw"],
        help="Filter configs to fwd (forward), bwd (input_backward), or wrw (weight_backward) conv modes.",
    )
    parser.add_argument(
        "--num-spatial-dims",
        "-N",
        help="Filter configs to a specific number of spatial dims.",
    )
    parser.add_argument(
        "--command",
        "-c",
        help="Run a specific config provided as a string. Ignores other filters.",
    )
    parser.add_argument(
        "--pipeline",
        "-p",
        help="Provide an explicit pipeline to lower from torch-ir to iree-input. Defaults to `builtin.module(torch-backend-to-linalg-on-tensors-backend-pipeline)`.",
    )
    return parser.parse_args()


def main(args: argparse.Namespace):
    path = None
    if args.output_dir:
        path = Path.cwd() / args.output_dir
        path.mkdir(exist_ok=True, parents=True)
    pipeline = args.pipeline if args.pipeline else DEFAULT_TORCH_TO_LINALG_PIPELINE
    if args.command:
        sig = command_to_signature(args.command)
        return generate_mlir(sig, path, import_pipeline=pipeline)
    commands = load_commands()
    signatures = {get_safe_name(c): command_to_signature(c) for c in commands}
    filters = {}
    if args.forw:
        filters["mode"] = Mode.parse(str(args.forw))
    if args.num_spatial_dims:
        filters["num_spatial_dims"] = int(args.num_spatial_dims)
    if args.all:
        return batch_generate_mlir(signatures, path, import_pipeline=pipeline)
    if len(filters.keys()) != 0:
        signatures = filter_signatures(signatures, **filters)
        return batch_generate_mlir(signatures, path, import_pipeline=pipeline)
    raise ValueError(
        "At least one of `--command`, `--forw`, `--num-spatial-dims`, or `--all` should be set"
    )


if __name__ == "__main__":
    results = main(_get_argparse())
    if results is not None:
        err, total = results
        print(f"Failed to generate IR for {err} configs of {total} total.")
