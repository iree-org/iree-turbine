import os
import shlex
import shutil
import subprocess
import warnings

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Tuple, Union, OrderedDict, Sequence

from iree.compiler.tools.core import compile_file, CompilerToolError
from iree.runtime import VmModule

from .conv import ConvSignature
from ....aot import export
from ....importers.ir import Attribute, MLIRError
from ....runtime import Launchable
from ....support.logging import runtime_logger as logger
from ..runtime import (
    LaunchableRuntimeCache,
    get_module_asm,
    is_cache_enabled,
    clear_cache,
    BOO_TUNING_SPEC_PATH,
)
from ..runtime import get_launchable as generic_get_launchable

__all__ = [
    "is_cache_enabled",
    "clear_cache_dir",
    "ConvLaunchableRuntimeCache",
    "get_launchable",
]

clear_cache_dir = clear_cache
ConvLaunchableRuntimeCache = LaunchableRuntimeCache


def _get_module_asm(
    signature: ConvSignature, func_name: str | None = None, use_custom: bool = True
) -> str:
    func_name = func_name or signature.get_func_name()
    module_factory = lambda: signature.get_nn_module(use_custom=use_custom)
    arg_factory = lambda: signature.get_sample_conv_args(splat_value=0)
    return get_module_asm(
        module_factory, arg_factory, func_name, force_single_dispatch=True
    )


def get_launchable(
    signature: ConvSignature, *, use_custom=True, cache_only=False
) -> Launchable:
    func_name = signature.get_func_name()
    module_factory = lambda: signature.get_nn_module(use_custom=use_custom)
    arg_factory = lambda: signature.get_sample_conv_args(splat_value=0)
    return generic_get_launchable(
        module_factory=module_factory,
        arg_factory=arg_factory,
        func_name=func_name,
        cache_only=cache_only,
        force_single_dispatch=True,
    )
