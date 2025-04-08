import os
import shutil
import warnings

from pathlib import Path
from typing import Union, OrderedDict

from iree.compiler.tools.core import compile_file, CompilerToolError

from .conv import ConvSignature
from ....aot import export
from ....importers.ir import Attribute, MLIRError
from ....runtime import Launchable
from ....support.logging import runtime_logger as logger

__all__ = [
    "is_cache_enabled",
    "clear_cache_dir",
    "get_launchable",
    "set_boo_cache",
]

_default_cache_base_dir = Path.home() / ".cache" / "turbine_kernels" / "boo"


def set_boo_cache(cache_dir: Union[Path, str, None] = None) -> Path:
    global CACHE_BASE_DIR
    if cache_dir:
        CACHE_BASE_DIR = Path(cache_dir)
        return CACHE_BASE_DIR
    if not "CACHE_BASE_DIR" in globals():
        CACHE_BASE_DIR = Path(os.environ.get("BOO_CACHE_DIR", _default_cache_base_dir))
        return CACHE_BASE_DIR


set_boo_cache()
BOO_CACHE_ON = int(os.environ.get("BOO_CACHE_ON", 1))


def is_cache_enabled() -> bool:
    return bool(BOO_CACHE_ON)


def clear_cache_dir():
    if not CACHE_BASE_DIR.is_dir():
        return
    shutil.rmtree(CACHE_BASE_DIR)


def _out_of_process_compile(func_name, key_hashes_and_flags):
    mlir_path = CACHE_BASE_DIR / func_name / f"{func_name}.mlir"
    if not mlir_path.is_file():
        logger.debug("no mlir file found at %s", str(mlir_path))
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


def _get_module_asm(
    signature: ConvSignature, func_name: str | None = None, use_custom: bool = True
) -> str:
    func_name = func_name or signature.get_func_name()
    cache_dir = CACHE_BASE_DIR / func_name
    mlir_path = cache_dir / f"{func_name}.mlir"

    if is_cache_enabled() and mlir_path.is_file():
        logger.debug("Loading cached mlir file at %s", str(mlir_path))
        return mlir_path.read_text()

    e = export(
        signature.get_nn_module(use_custom=use_custom),
        args=signature.get_sample_conv_args(splat_value=0),
        function_name=func_name,
    )

    e.import_to("full")

    mod = e.mlir_module

    ctx = mod.context
    func_op = mod.regions[0].blocks[0].operations[0]
    try:
        with ctx:
            pipeline_attr = Attribute.parse(
                '#util.preprocessing_pipeline<"iree-preprocessing-make-single-dispatch">'
            )
            func_op.attributes["preprocessing_pipeline"] = pipeline_attr
    except MLIRError as e:
        warnings.warn(
            f"Failed to attach #util.preprocessing_pipeline attr to func op. Please try using a newer version of IREE."
        )

    module_asm = str(e.mlir_module)

    if is_cache_enabled():
        logger.debug("Saving newly generated mlir file to %s", str(mlir_path))
        cache_dir = CACHE_BASE_DIR / func_name
        cache_dir.mkdir(parents=True, exist_ok=True)
        mlir_path = cache_dir / f"{func_name}.mlir"
        mlir_path.write_text(module_asm)

    return module_asm


class ConvLaunchableRuntimeCache:
    def __init__(self, cache_limit: int | None = None):
        self.cache_limit = cache_limit
        self.session_cache: OrderedDict[str, Launchable] = OrderedDict()

    def set_cache_limit(self, new_cache_limit: int | None):
        self.cache_limit = new_cache_limit

    def add_to_session_cache(self, key: str, launchable: Launchable):
        self.session_cache[key] = launchable
        self.session_cache.move_to_end(key)
        if (
            self.cache_limit is not None
            and len(self.session_cache.keys()) > self.cache_limit
        ):
            self.session_cache.popitem(last=False)

    def get(self, key: str) -> Launchable | None:
        return self.session_cache.get(key, None)

    @staticmethod
    def get_launchable_cache(cache_limit: int | None = None):
        global _launchable_cache
        if "_launchable_cache" in globals():
            _launchable_cache.set_cache_limit(cache_limit)
            return _launchable_cache
        return ConvLaunchableRuntimeCache(cache_limit)


def get_launchable(
    signature: ConvSignature, *, use_custom=True, cache_only=False
) -> Launchable:
    func_name = signature.get_func_name()
    session_cache_key = func_name + cache_only * "_no_jit"
    launch_cache = ConvLaunchableRuntimeCache.get_launchable_cache()
    launch = launch_cache.get(session_cache_key)
    if launch:
        return launch
    cache_dir = CACHE_BASE_DIR / func_name if is_cache_enabled() else None
    if cache_only:
        launch = Launchable.from_file_cache_only(
            cache_dir,
            parameter_providers=(),
            entry_point=func_name,
        )
    else:
        module_asm = _get_module_asm(signature, func_name, use_custom=use_custom)
        launch = Launchable.jit_compile(
            module_asm,
            parameter_providers=(),
            entry_point=func_name,
            file_cache_dir=cache_dir,
        )
    launch_cache.add_to_session_cache(session_cache_key, launch)
    return launch
