# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# generates sample kernels corresponding to MiOpen conv signatures

import argparse
import gc
import re

from pathlib import Path

from typing import (
    Dict,
    List,
    Tuple,
)

import torch

from iree.compiler.extras.fx_importer import FxImporter
from iree.compiler.passmanager import PassManager

from iree.turbine.kernel.boo.conv_exports.miopen_parser import (
    command_to_signature,
)

from iree.turbine.kernel.boo.conv_exports.conv import ConvSignature

__all__ = [
    "DEFAULT_TORCH_TO_LINALG_PIPELINE",
    "generate_mlir",
    "filter_signatures",
]

DEFAULT_TORCH_TO_LINALG_PIPELINE = [
    "torch-backend-to-linalg-on-tensors-backend-pipeline"
]


def generate_mlir(
    signature: ConvSignature,
    output_path: str | Path | None = None,
    *,
    import_pipeline: str | List[str] | None = DEFAULT_TORCH_TO_LINALG_PIPELINE,
    print_ir: bool = False,
    print_ir_after_all: bool = False,
):
    """For a given ConvSignature, imports the conv to mlir"""
    args = signature.get_sample_conv_args()
    m = signature.get_nn_module()
    e = torch.export.export(m, args=args)
    importer = FxImporter()
    importer.import_program(e, func_name=signature.get_func_name())
    if import_pipeline:
        pipeline_str = (
            import_pipeline
            if isinstance(import_pipeline, str)
            else "builtin.module(" + ",".join(import_pipeline) + ")"
        )
        pm = PassManager.parse(
            pipeline_str,
            context=importer.module.context,
        )
        if print_ir_after_all:
            pm.enable_ir_printing()
        pm.run(importer.module.operation)
    if output_path:
        Path(output_path).write_text(str(importer.module))
    if print_ir:
        print(str(importer.module))
    return importer.module


def _batch_generate_mlir(
    signatures: Dict[str, ConvSignature],
    save_dir: str | Path | None,
    *,
    import_pipeline: List[str] = DEFAULT_TORCH_TO_LINALG_PIPELINE,
    print_ir: bool = False,
    print_ir_after_all: bool = False,
) -> Tuple[int, int]:
    """prints or saves mlir for each signature provided. Returns tuple: (#failed, #total)"""
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
    total = 0
    err = 0
    for name, s in signatures.items():
        print(f"processing {name}...")
        path = None if not save_dir else Path(save_dir) / f"{name}.mlir"
        total += 1
        try:
            generate_mlir(
                s,
                path,
                import_pipeline=import_pipeline,
                print_ir=print_ir,
                print_ir_after_all=print_ir_after_all,
            )
        except Exception as e:
            err += 1
            err_str = f"signature = {s}\n{str(e)}"
            if save_dir:
                path = Path(save_dir) / f"ERR_{name}.log"
                path.write_text(err_str)
            else:
                print(err_str)
        gc.collect()
    return err, total


def filter_signatures(signatures: Dict[str, ConvSignature], **kwargs):
    """Filters a dictonary of named conv signatures by kwargs."""
    if len(kwargs.keys()) == 0:
        return signatures
    filtered = {}
    for name, sig in signatures.items():
        match = True
        for k, v in kwargs.items():
            if sig._asdict()[k] != v:
                match = False
        if match:
            filtered[name] = sig
    return filtered


def _load_commands(commands_file):
    """loads commands from a text file."""
    # try an absolute path
    path = Path(commands_file)
    # if the path doesn't point anywhere, try relative to cwd and this file.
    if not path.is_file():
        path = Path.cwd() / commands_file
    if not path.is_file():
        path = Path(__file__) / commands_file
    if not path.is_file():
        raise ValueError(
            f"'commands-file' specification, '{commands_file}', cannot be found."
        )
    commands = [c for c in path.read_text().splitlines() if c.startswith("conv")]
    return commands


def _get_safe_name(command: str) -> str:
    name = "".join(command.split())
    return re.sub("-", "_", name)


def _get_argparse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tool for generating MLIR for convolutions matching the signature of the MiOpen driver <https://github.com/ROCm/MIOpen/blob/develop/driver/README.md> for conv."
    )
    parser.add_argument(
        "--commands-file",
        "-f",
        type=str,
        help="Allows running all Miopen driver commands from a text file.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        help="Specify relative path from cwd to store output mlir files.",
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
    parser.add_argument(
        "--mlir-print-ir-after-all",
        action="store_true",
        default=False,
        help="Enables ir printing for the pass manager. This will dump IR after each pass applied.",
    )
    return parser.parse_args()


def main(args: argparse.Namespace):
    if not args.command and not args.commands_file:
        raise ValueError(
            "At least one of `--command` or `--commands-file` should be set"
        )
    path = None
    if args.output_dir:
        path = Path(args.output_dir)
        if not path.is_absolute():
            path = Path.cwd() / args.output_dir
        path.mkdir(exist_ok=True, parents=True)
    print_ir = not bool(path)
    pipeline = args.pipeline if args.pipeline else DEFAULT_TORCH_TO_LINALG_PIPELINE
    if args.command:
        sig = command_to_signature(args.command)
        generate_mlir(
            sig,
            path,
            import_pipeline=pipeline,
            print_ir=print_ir,
            print_ir_after_all=args.mlir_print_ir_after_all,
        )
        return None
    # user must have specified a commands-file
    commands = _load_commands(args.commands_file)
    signatures = {_get_safe_name(c): command_to_signature(c) for c in commands}
    # check for filters
    filters = {}
    if args.forw:
        filters["mode"] = args.forw
    if args.num_spatial_dims:
        filters["num_spatial_dims"] = int(args.num_spatial_dims)
    signatures = filter_signatures(signatures, **filters)
    return _batch_generate_mlir(
        signatures,
        path,
        import_pipeline=pipeline,
        print_ir=print_ir,
        print_ir_after_all=args.mlir_print_ir_after_all,
    )


if __name__ == "__main__":
    results = main(_get_argparse())
    if results is not None:
        err, total = results
        print(f"Failed to generate IR for {err} configs of {total} total.")
