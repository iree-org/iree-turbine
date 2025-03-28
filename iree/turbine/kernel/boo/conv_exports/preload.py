import hashlib
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, Sequence, Union

import torch

from iree.compiler.tools.core import compile_file, CompilerToolError

from .conv import ConvSignature, ConvSignatureStorage
from .generate import _load_commands
from .launch import CACHE_BASE_DIR, _get_module_asm
from .miopen_parser import command_to_signature
from ....runtime.device import get_device_from_torch
from ....support.ir_imports import MLIRError
from ....support.logging import runtime_logger as logger

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
        devices: Sequence[Union[torch.device, str]] | None = None,
        signatures: Sequence[ConvSignature] = (),
        commands: Sequence[str] = (),
        commands_file: Union[str, Path, None] = None,
        allow_download: bool = False,
    ):
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

        key_hashes_and_flags = []
        for d in self.torch_devices:
            turbine_device = get_device_from_torch(d)
            key_hashes_and_flags.append(
                (
                    hashlib.sha1(
                        turbine_device.type_cache_key.encode(), usedforsecurity=False
                    ).hexdigest(),
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

        pool.starmap(_compile, items)
        pool.close()

    def get_cache_status(self):
        key_to_hash = {}
        for d in self.torch_devices:
            key = get_device_from_torch(d).type_cache_key
            key_to_hash[key] = hashlib.sha1(
                key.encode(), usedforsecurity=False
            ).hexdigest()
        status = {}
        for dir in CACHE_BASE_DIR.glob("*/"):
            name = dir.name
            status[name] = {}
            status[name]["MLIR"] = (dir / f"{name}.mlir").is_file()
            for key, hash in key_to_hash.items():
                status[name][key] = (dir / f"{hash}.vmfb").is_file()
        return status


def mlir_import(sig: ConvSignature) -> str:
    func_name = sig.get_func_name()
    # make an empty cache dir here, in case we fail import
    cache_dir = CACHE_BASE_DIR / func_name
    cache_dir.mkdir(exist_ok=True, parents=True)
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


def _compile(func_name, key_hashes_and_flags):
    mlir_path = CACHE_BASE_DIR / func_name / f"{func_name}.mlir"
    if not mlir_path.is_file():
        return

    for key_hash, flags in key_hashes_and_flags:
        try:
            vmfb_path: Path = CACHE_BASE_DIR / func_name / f"{key_hash}.vmfb"
            if vmfb_path.is_file():
                logger.debug("found vmfb in cache: %s", str(vmfb_path))
                continue
            logger.debug("Compiling vmfb to cache: %s", str(vmfb_path))
            options = {
                "output_file": str(vmfb_path),
                "extra_args": flags,
            }
            compile_file(str(mlir_path), **options)
        except CompilerToolError as e:
            logger.debug("failed compilation with diagnostics: %s", str(e))
