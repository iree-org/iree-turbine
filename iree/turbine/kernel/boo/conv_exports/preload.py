import hashlib
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, Sequence, Union

import torch

from iree.turbine.kernel.boo.conv_exports.conv import (
    ConvSignature,
    ConvSignatureStorage,
)
from iree.turbine.kernel.boo.conv_exports.generate import _load_commands
from iree.turbine.kernel.boo.conv_exports.launch import (
    _get_module_asm,
    set_boo_cache,
    _out_of_process_compile,
    get_launchable,
)
from iree.turbine.kernel.boo.conv_exports.miopen_parser import command_to_signature
from iree.turbine.runtime.device import get_device_from_torch
from iree.turbine.support.ir_imports import MLIRError
from iree.turbine.support.logging import runtime_logger as logger

__all__ = [
    "CachePopulator",
]


def _get_unique_torch_device_list():
    import torch

    torch_devices = [torch.device("cpu")]
    first_unique_device: Dict[str, int] = {}
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
        signatures: Sequence[ConvSignature] = (),
        commands: Sequence[str] = (),
        commands_file: Union[str, Path, None] = None,
        allow_download: bool = False,
    ):
        self.cache_dir = set_boo_cache(cache_dir)
        self.torch_devices = devices or _get_unique_torch_device_list()
        self.torch_devices = [torch.device(d) for d in self.torch_devices]
        self.signatures = list(signatures)
        self.commands = list(commands)
        self.commands_file = commands_file
        self.download = allow_download
        if self.download:
            raise NotImplementedError(
                "Downloading optimized kernels is not yet supported."
            )

    def _assemble_signatures(self):
        if self.commands_file:
            self.commands += _load_commands(self.commands_file)
            self.commands_file = None
        if len(self.commands) > 0:
            new_signatures = [command_to_signature(c) for c in self.commands]
            self.signatures = list(self.signatures) + list(new_signatures)
            self.commands = None
        # de-duplicate signatures
        self.signatures = list(set(self.signatures))

    def run(
        self, max_processes: int | None = None, *, use_multiprocess_import: bool = False
    ):
        """
        Runs the prepopulator. Will first convert all MiOpen commands to ConvSignatures.

        The compilation phase is always done in multiple processes, but the import phase
        is not done in parallel (by default) because of an issue occuring in dynamo when
        `torch.cuda.is_available()`.
        """
        self._assemble_signatures()
        logger.debug(
            "Prepopulating signatures: %s",
            str(list([s._signature for s in self.signatures])),
        )

        key_hashes_and_flags = set()
        for d in self.torch_devices:
            turbine_device = get_device_from_torch(d)
            key_hashes_and_flags.add(
                (
                    turbine_device.get_type_key_hash(),
                    turbine_device.compile_target_flags,
                )
            )

        pool = Pool(max_processes)
        if use_multiprocess_import:
            # workaround since ConvSignature isn't pickle-able
            # TODO: figure out how to get this to work on CUDA.
            # Currently, the user needs to set an environment variable:
            # 'CUDA_VISIBLE_DEVICES="-1"' before running to generate mlir.
            # Then the user needs to re-run without the environement variable to generate vmfbs for GPU.
            if torch.cuda.is_available():
                warning_msg = (
                    "CUDA must be disabled during import for dynamo to work properly. "
                    'Try running your script with CUDA_VISIBLE_DEVICES="-1" once to populate mlir. '
                    "Then run again without the flag to generate binaries for cuda devices."
                )
                logger.warning(warning_msg)
            sig_storages = [sig._signature for sig in self.signatures]
            names = pool.map(_mlir_import, sig_storages)
            items = [(n, key_hashes_and_flags) for n in names]
        else:
            items = [(mlir_import(s), key_hashes_and_flags) for s in self.signatures]

        pool.starmap(_out_of_process_compile, items)
        pool.close()

    def get_cache_status(
        self, func_name: str, cache_dir: Union[str, Path, None] = None
    ):

        if not cache_dir:
            from iree.turbine.kernel.boo.conv_exports.launch import CACHE_BASE_DIR

            cache_dir = CACHE_BASE_DIR

        cache_dir = Path(cache_dir) / func_name

        mlir_import_status = (
            cache_dir.is_dir() and (cache_dir / f"{func_name}.mlir").is_file()
        )
        status = {"mlir_import": mlir_import_status}
        for d in self.torch_devices:
            vmfb_path = (
                cache_dir / f"{get_device_from_torch(d).get_type_key_hash()}.vmfb"
            )
            status[d] = mlir_import_status and vmfb_path.is_file()

        return status


def mlir_import(sig: ConvSignature) -> str:
    func_name = sig.get_func_name()
    try:
        _get_module_asm(sig, func_name),
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
    return func_name


def _mlir_import(sig_storage: ConvSignatureStorage) -> str:
    """
    Runs mlir_import from an underlying ConvSignatureStorage.
    ConvSignature is not pickle-able, so this function can be used instead.
    """
    kwargs = sig_storage._asdict()
    kwargs.pop("num_spatial_dims")
    sig = ConvSignature(**kwargs)
    return mlir_import(sig)


def cl_main(args):
    if args.cache_dir:
        set_boo_cache(args.cache_dir)
    devices = [args.device] if args.device else _get_unique_torch_device_list()
    populator = CachePopulator(devices=devices, commands_file=args.commands_file)
    populator.run(
        max_processes=args.max_processes,
        use_multiprocess_import=args.import_multiprocessing,
    )


def _get_preload_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Tool for prepopulating the boo cache from the command line for all miopen driver commands in a specific file."
    )
    parser.add_argument(
        "commands_file",
        type=str,
        help="Allows running all Miopen driver commands from a text file.",
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
    return parser.parse_args()


if __name__ == "__main__":
    cl_main(_get_preload_args())
