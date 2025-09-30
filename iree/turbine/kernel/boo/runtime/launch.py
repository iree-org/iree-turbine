# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import shlex
import subprocess
import warnings

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable, Iterable, Tuple, Sequence, TypeAlias

import torch

from iree.runtime import VmModule

from .cache import *
from ....aot import export
from ....importers.ir import Attribute, MLIRError
from ....runtime import Launchable, Device
from ....support.logging import runtime_logger as logger

__all__ = [
    "BOO_TUNING_SPEC_PATH",
    "get_module_asm",
    "get_launchable",
    "out_of_process_compile",
    "user_flags_jit_callback",
]

## environment variables

BOO_TUNING_SPEC_PATH = os.environ.get(
    "BOO_TUNING_SPEC_PATH", str(Path(__file__).parent / "tuning_specs.mlir")
)


def get_module_asm(
    module_factory: Callable[[], torch.nn.Module],
    arg_factory: Callable[[], Iterable],
    func_name: str = "main",
    force_single_dispatch: bool = False,
) -> str:
    """Create MLIR assembly str from input torch.nn.Module.

    Roughly speaking, for an input torch.nn.Module with a CustomOp,
    `get_module_asm` goes through the following transformation steps:
        1. Export torch.nn.Module to an fx graph
            class GraphModule(torch.nn.Module):
                def forward
                    my_op: "f32[128, 24, 48, 384]" = torch.ops.turbine.something ...
                    return (my_op,)
        2. Convert fx graph to MLIR
            @something(%arg0: !torch.vtensor<[...],bf16>) -> !torch.vtensor<[...],bf16> {
                %0 = torch.operator "torch.turbine.something"(...
                return %0 : !torch.vtensor<[...],bf16>
            }
        3. Custom op expansion
            @something(%arg0: !torch.vtensor<[...],bf16>) -> !torch.vtensor<[...],bf16> {
                %0 = torch_c.to_builtin_tensor %arg0 ...
                %1 = util.call @something_impl(%0) ...
                %2 = torch_c.from_builtin_tensor %1 ...
                return %2 : !torch.vtensor<[...],bf16>
            }
            util.func private @something_impl(%arg0: tensor<...>) -> tensor<...> {
                ...
                util.return ...
            }
        4. Lower to IREE
            - `torch-to-iree` pass converts torch dialect to IREE input dialects.

    Args:
        module_factory: torch.nn.Module to lower
        arg_factory: Either an iterable of arguments or a callable that returns
            an iterable of arguments. These are the example inputs used for
            exporting the module.
        func_name: Name of the exported function.
        force_single_dispatch: If True, wraps output in an attribute instructing
            IREE to attempt to produce a single dispatch.

    Returns:
        module_asm: MLIR asm representation of the input torch.nn.Module
    """
    cache_dir = set_cache_dir() / func_name
    mlir_path = cache_dir / f"{func_name}.mlir"

    if is_cache_enabled() and mlir_path.is_file():
        logger.debug("Loading cached mlir file at %s", str(mlir_path))
        return mlir_path.read_text()

    e = export(
        module_factory(),
        args=tuple(arg_factory()),
        function_name=func_name,
    )

    e.import_to("full")

    mod = e.mlir_module

    if force_single_dispatch:
        ctx = mod.context
        func_op = mod.regions[0].blocks[0].operations[0]
        try:
            with ctx:
                pipeline_attr = Attribute.parse(
                    '#util.preprocessing_pipeline<"iree-preprocessing-make-single-dispatch">'
                )
                func_op.attributes["preprocessing_pipeline"] = pipeline_attr
        except MLIRError:
            warnings.warn(
                f"Failed to attach #util.preprocessing_pipeline attr to func op. Please try using a newer version of IREE."
            )

    module_asm = str(mod)

    if is_cache_enabled():
        logger.debug("Saving newly generated mlir file to %s", str(mlir_path))
        cache_dir = set_cache_dir() / func_name
        cache_dir.mkdir(parents=True, exist_ok=True)
        mlir_path = cache_dir / f"{func_name}.mlir"
        mlir_path.write_text(module_asm)

    return module_asm


CompilerFlagsCallback: TypeAlias = Callable[[Device, Path], list[str]]


def out_of_process_compile(
    func_name: str, key_hashes_and_flags: Sequence[Tuple[str, Sequence[str]]]
) -> Tuple[str, Tuple[bool, ...]]:
    """Runs compilation via command line tool. Does not raise an exception if compilation fails."""
    boo_cache = set_cache_dir()
    mlir_path = boo_cache / func_name / f"{func_name}.mlir"
    if not mlir_path.is_file():
        logger.debug("no mlir file found at %s", str(mlir_path))
        return func_name, tuple([False] * len(key_hashes_and_flags))
    success = []
    for key_hash, flags in key_hashes_and_flags:
        vmfb_path: Path = boo_cache / func_name / f"{key_hash}.vmfb"
        if vmfb_path.is_file():
            logger.debug("found vmfb in cache: %s", str(vmfb_path))
            success.append(True)
            continue
        logger.debug("Compiling vmfb to cache: %s", str(vmfb_path))
        cl_list = (
            ["iree-compile"] + list(flags) + [str(mlir_path), "-o", str(vmfb_path)]
        )
        command = shlex.join(cl_list)
        (boo_cache / func_name / f"compile_command_{key_hash}.txt").write_text(command)
        logger.debug("compile command:\n%s", command)
        try:
            ret = subprocess.run(command, capture_output=True, shell=True, timeout=10)
        except subprocess.TimeoutExpired as e:
            logger.warning("Process timed out. See message:\n%s", str(e))
            vmfb_path.unlink(missing_ok=True)
            success.append(False)
            continue
        if ret.returncode != 0:
            logger.warning("failed executing compile command: %s", command)
            # clean-up any empty vmfb files created
            vmfb_path.unlink(missing_ok=True)
        print(ret.stdout.decode())
        print(ret.stderr.decode())
        success.append(ret.returncode == 0)
    return func_name, tuple(success)


def user_flags_jit_callback(
    func_name: str, compiler_flags_callback: CompilerFlagsCallback, source: str
):
    """VmModule callback for out-of-process compilation with extra flags provided.
    If boo cache is disabled, this will create temporary files for compilation."""

    boo_cache = set_cache_dir()

    def _compile(flags, mlir_path, vmfb_path):
        cl_list = ["iree-compile"] + flags + [str(mlir_path), "-o", str(vmfb_path)]
        command = shlex.join(cl_list)
        (vmfb_path.parent / f"compile_command_{vmfb_path.stem}.txt").write_text(command)
        ret = subprocess.run(command, capture_output=True, shell=True, timeout=10)
        if ret.returncode != 0:
            raise RuntimeError(
                f"Failed compilation for kernel {mlir_path.stem}. "
                "If this is a supported BOO op, it's possible compilation failed due to a bad fusion.\n"
                "Please file an issue at https://github.com/iree-org/iree/issues with the following info.\n"
                f"SOURCE IR:\n{Path(mlir_path).read_text()}\n."
                f"COMPILE COMMAND:\n{command}\n."
                f"STDERR: {ret.stderr.decode()}."
            )
        return vmfb_path.read_bytes()

    def callback(device):
        key_hash = device.get_type_key_hash()
        vmfb_path: Path = boo_cache / func_name / f"{key_hash}.vmfb"
        vm_instance = device.vm_instance

        if is_cache_enabled() and vmfb_path.is_file():
            logger.debug("Loading vmfb from cache: %s", str(vmfb_path))
            vmfb = vmfb_path.read_bytes()
            return VmModule.copy_buffer(vm_instance, vmfb)

        flags = list(device.compile_target_flags) + compiler_flags_callback(
            device, vmfb_path.parent
        )

        if is_cache_enabled():
            mlir_path = boo_cache / func_name / f"{func_name}.mlir"
            logger.debug("Compiling vmfb to cache: %s", str(vmfb_path))
            vmfb = _compile(flags, mlir_path, vmfb_path)
            return VmModule.copy_buffer(vm_instance, vmfb)

        with TemporaryDirectory() as td:
            mlir_path = Path(td) / "source.mlir"
            mlir_path.write_text(source)
            vmfb_path = Path(td) / "target.vmfb"
            vmfb = _compile(flags, mlir_path, vmfb_path)
            return VmModule.copy_buffer(vm_instance, vmfb)

    return callback


def default_compiler_flags_callback(device: Device, cache_dir: Path) -> list[str]:
    flags: list[str] = []
    if device.driver_id == "hip":
        flags.append("--iree-opt-level=O3")
        flags.append(
            "--iree-dispatch-creation-enable-fuse-padding-into-linalg-consumer-ops"
        )
        flags.append(
            "--iree-preprocessing-pass-pipeline=builtin.module(iree-preprocessing-convert-conv-filter-to-channels-last)"
        )
        if BOO_TUNING_SPEC_PATH != "":
            assert Path(
                BOO_TUNING_SPEC_PATH
            ).is_file(), "Provided path for tuning specs is invalid."
            flags.append(f"--iree-codegen-tuning-spec-path={BOO_TUNING_SPEC_PATH}")
    else:
        # This is the highest level with full backend support.
        flags.append("--iree-opt-level=O1")
    return flags


def get_launchable(
    module_factory: torch.nn.Module | Callable[[], torch.nn.Module],
    arg_factory: Iterable | Callable[[], Iterable],
    func_name: str = "main",
    *,
    cache_only: bool = False,
    force_single_dispatch: bool = False,
    compiler_flags_callback: CompilerFlagsCallback = default_compiler_flags_callback,
) -> Launchable:
    session_cache_key = func_name + cache_only * "_no_jit"
    launch = LaunchableRuntimeCache.get(session_cache_key)
    if launch:
        return launch
    module = module_factory
    if isinstance(module, torch.nn.Module):
        module_factory = lambda: module
    args = arg_factory
    if isinstance(args, Iterable):
        arg_factory = lambda: args
    cache_dir = set_cache_dir() / func_name if is_cache_enabled() else None
    if cache_only:
        assert (
            cache_dir is not None
        ), "Cache-only was requested, but the cache is disabled."
        launch = Launchable.from_file_cache_only(
            cache_dir,
            parameter_providers=(),
            entry_point=f"{func_name}$async",
        )
    else:
        module_asm = get_module_asm(
            module_factory, arg_factory, func_name, force_single_dispatch
        )
        launch = Launchable.from_vm_module(
            user_flags_jit_callback(
                func_name,
                compiler_flags_callback,
                module_asm,
            ),
            entry_point=f"{func_name}$async",
        )
    LaunchableRuntimeCache.add(session_cache_key, launch)
    return launch
