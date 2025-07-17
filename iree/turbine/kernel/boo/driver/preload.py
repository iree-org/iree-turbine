# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from multiprocessing import Pool
from pathlib import Path
from typing import Any, Sequence, Union
import argparse

import torch

from iree.turbine.kernel.boo.driver.registry import BooOpRegistry
from iree.turbine.kernel.boo.driver.utils import load_commands
from iree.turbine.kernel.boo.driver.launch import get_module_asm
from iree.turbine.kernel.boo.runtime import BOO_TUNING_SPEC_PATH
from iree.turbine.kernel.boo.runtime import out_of_process_compile, set_cache_dir
from iree.turbine.runtime.device import get_device_from_torch
from iree.turbine.support.ir_imports import MLIRError
from iree.turbine.support.logging import runtime_logger as logger
from iree.turbine.kernel.boo.exports.signature import OpSignature
from iree.turbine.kernel.boo.exports.parser import OpCLIParser

__all__ = [
    "CachePopulator",
]


def _get_unique_torch_device_list() -> list[torch.device]:
    """Returns a list of Torch devices with unique names."""

    torch_devices = [torch.device("cpu")]
    first_unique_device: dict[str, int] = {}
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            device_name = torch.cuda.get_device_properties(i).name
            if device_name not in first_unique_device.keys():
                first_unique_device[device_name] = i
    for i in first_unique_device.values():
        torch_devices.append(torch.device(f"cuda:{i}"))
    return torch_devices


class CachePopulator:
    def __init__(
        self,
        *,
        cache_dir: Union[str, Path, None] = None,
        devices: Sequence[Union[torch.device, str]] | None = None,
        signatures: Sequence[OpSignature] = (),
        commands: Sequence[str] = (),
        commands_file: Union[str, Path, None] = None,
        allow_download: bool = False,
        extra_compile_flags: Sequence[str] = (),
    ):
        self.cache_dir = set_cache_dir(cache_dir)
        self.torch_devices = devices or _get_unique_torch_device_list()
        self.torch_devices = [torch.device(d) for d in self.torch_devices if d != ""]
        self.signatures = list(signatures)
        self.commands = list(commands)
        self.commands_file = commands_file
        self.download = allow_download
        if self.download:
            raise NotImplementedError(
                "Downloading optimized kernels is not yet supported."
            )
        self.extra_compile_flags = extra_compile_flags
        self.mlir_cache_status = {}
        self.vmfb_cache_status = {}

    def _assemble_signatures(self):
        if self.commands_file:
            self.commands += load_commands(self.commands_file)
            self.commands_file = None
        if len(self.commands) > 0:
            new_signatures = [BooOpRegistry.parse_command(c) for c in self.commands]
            self.signatures = list(self.signatures) + list(new_signatures)
            self.commands = None
        # de-duplicate signatures
        self.signatures = list(set(self.signatures))

    def run(
        self, max_processes: int | None = None, *, use_multiprocess_import: bool = False
    ):
        """
        Runs the pre-populator. Will first convert all MiOpen commands to ConvSignatures.

        The compilation phase is always done in multiple processes, but the import phase
        is not done in parallel (by default) because of an issue occurring in dynamo when
        `torch.cuda.is_available()`.
        """
        self._assemble_signatures()
        logger.debug(
            "Pre-populating signatures: %s",
            str(list[self.signatures]),
        )

        key_hashes_and_flags = set()
        for d in self.torch_devices:
            turbine_device = get_device_from_torch(d)
            compile_flags = tuple(turbine_device.compile_target_flags) + tuple(
                self.extra_compile_flags
            )
            key_hashes_and_flags.add(
                (
                    turbine_device.get_type_key_hash(),
                    compile_flags,
                )
            )

        pool = Pool(max_processes)
        if use_multiprocess_import:
            # workaround since OpSignature isn't pickle-able
            # TODO: figure out how to get this to work on CUDA.
            # Currently, the user needs to set an environment variable:
            # 'CUDA_VISIBLE_DEVICES="-1"' before running to generate mlir.
            # Then the user needs to re-run without the environment variable
            # to generate vmfbs for GPU.
            if torch.cuda.is_available():
                warning_msg = (
                    "CUDA must be disabled during import for dynamo to work properly. "
                    'Try running your script with CUDA_VISIBLE_DEVICES="-1" once to populate mlir. '
                    "Then run again without the flag to generate binaries for cuda devices."
                )
                logger.warning(warning_msg)
            sig_dicts_and_cls = [
                (sig.as_init_kwargs(), type(sig)) for sig in self.signatures
            ]
            name_and_status = pool.map(_mlir_import_picklable, sig_dicts_and_cls)
        else:
            name_and_status = [mlir_import(s) for s in self.signatures]

        items = [
            (name, key_hashes_and_flags)
            for name, succeeded in name_and_status
            if succeeded
        ]

        name_and_compile_status = pool.starmap(out_of_process_compile, items)

        pool.close()

        self.mlir_cache_status = dict(name_and_status)
        self.vmfb_cache_status = dict(name_and_compile_status)

    def get_cache_status(
        self,
        func_name: str,
    ):

        mlir_status = self.mlir_cache_status.get(func_name, False)
        status = {"mlir_import", mlir_status}
        for i, d in enumerate(self.torch_devices):
            status[d] = mlir_status and self.vmfb_cache_status[func_name][i]

        return status

    def get_failures(self) -> dict[str, str]:
        # assume we already have assembled signatures
        failed_funcs: dict[str, str] = {}
        for fn in self.mlir_cache_status.keys():
            if not self.mlir_cache_status[fn]:
                failed_funcs[fn] = "mlir import failure"
                continue
            failed_devices = set()
            for i, d in enumerate(self.torch_devices):
                if self.vmfb_cache_status[fn][i]:
                    continue
                failed_devices.add(str(d))
            if len(failed_devices) == 0:
                continue
            failed_funcs[fn] = f"failed compilation for devices: {failed_devices}"
        return failed_funcs


def mlir_import(sig: OpSignature) -> tuple[str, bool]:
    func_name = sig.func_name
    success = False
    try:
        get_module_asm(sig, func_name)
        success = True
    except MLIRError as e:
        logger.debug(
            "Signature failed lowering to iree-input: %s. raised exception: %s",
            func_name,
            str(e),
        )
    except NotImplementedError as e:
        logger.debug(
            "Found an unimplemented signature: %s. raised exception: %s",
            func_name,
            str(e),
        )
    except Exception as e:
        logger.debug(
            "Unknown exception encountered for signature %s, see : %s",
            func_name,
            str(e),
        )
    return func_name, success


def _mlir_import_picklable(
    args: tuple[dict[str, Any], type[OpSignature]],
) -> tuple[str, bool]:
    signature_dict, signature_cls = args
    sig = signature_cls(**signature_dict)
    return mlir_import(sig)


def _cl_main(
    args: argparse.Namespace,
    extra_flags: Sequence[str],
):
    if args.cache_dir:
        set_cache_dir(args.cache_dir)
    devices = (
        [args.device] if args.device is not None else _get_unique_torch_device_list()
    )
    populator = CachePopulator(
        devices=devices,
        commands_file=args.commands_file,
        extra_compile_flags=extra_flags,
    )
    populator.run(
        max_processes=args.max_processes,
        use_multiprocess_import=args.import_multiprocessing,
    )
    failed_funcs = populator.get_failures()
    print("Failures:")
    for fn, failure_str in failed_funcs.items():
        print(f"{fn}  :  {failure_str}")


def _get_preload_args() -> tuple[str, str]:
    parser = argparse.ArgumentParser(
        description="Tool for pre-populating the boo cache from the command line for all MIOpen driver commands in a specific file."
    )
    parser.add_argument(
        "commands_file",
        type=str,
        help="Allows running all MIOpen driver commands from a text file.",
    )
    parser.add_argument(
        "--cache-dir",
        "-o",
        required=False,
        type=str,
        help=(
            "Specify absolute or relative path from cwd to store output mlir files."
            "Uses `BOO_CACHE_DIR` or default if not specified."
        ),
    )
    parser.add_argument(
        "--device",
        "-d",
        required=False,
        type=str,
        help=(
            "specify a string identifier for a torch.device to compile for. "
            "E.g. 'cpu' or 'cuda:0'. Default is to compile for each device type."
        ),
    )
    parser.add_argument(
        "--import-multiprocessing",
        "-m",
        action="store_true",
        default=False,
        help=(
            "Set this flag to enable multiprocessing for mlir imports. "
            "Currently requires passing 'CUDA_VISIBLE_DEVICES=-1' on the first run. "
            "Unset the environment variable and run again to populate GPU executables."
        ),
    )
    parser.add_argument(
        "--max-processes",
        "-j",
        type=int,
        help="Specify a maximum number of concurrent processes.",
    )
    return parser.parse_known_args()


def preload():
    args, extra_compile_flags = _get_preload_args()
    if BOO_TUNING_SPEC_PATH != "":
        extra_compile_flags.extend(
            [f"--iree-codegen-tuning-spec-path={BOO_TUNING_SPEC_PATH}"]
        )
    _cl_main(args, extra_compile_flags)


if __name__ == "__main__":
    preload()
