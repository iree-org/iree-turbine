import os
import shutil
import warnings

from pathlib import Path

from .conv import ConvSignature
from ....aot import export
from ....importers.ir import Attribute, MLIRError
from ....runtime import Launchable
from ....support.logging import runtime_logger as logger

__all__ = [
    "is_cache_enabled",
    "clear_cache_dir",
    "get_launchable",
]

_default_cache_base_dir = Path.home() / ".cache" / "turbine_kernels" / "boo"
CACHE_BASE_DIR = Path(os.environ.get("BOO_CACHE_DIR", _default_cache_base_dir))
BOO_CACHE_ON = int(os.environ.get("BOO_CACHE_ON", 1))


def is_cache_enabled() -> bool:
    return bool(BOO_CACHE_ON)


def clear_cache_dir():
    if not CACHE_BASE_DIR.is_dir():
        return
    shutil.rmtree(CACHE_BASE_DIR)


def _get_module_asm(signature: ConvSignature, func_name: str | None = None) -> str:
    func_name = func_name or signature.get_func_name()
    cache_dir = CACHE_BASE_DIR / func_name
    mlir_path = cache_dir / f"{func_name}.mlir"

    if is_cache_enabled() and mlir_path.is_file():
        logger.debug("Loading cached mlir file at %s", str(mlir_path))
        return mlir_path.read_text()

    e = export(
        signature.get_nn_module(use_custom=True),
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


def get_launchable(signature: ConvSignature) -> Launchable:
    func_name = signature.get_func_name()
    module_asm = _get_module_asm(signature, func_name)
    cache_dir = CACHE_BASE_DIR / func_name if is_cache_enabled() else None
    return Launchable.jit_compile(
        module_asm,
        parameter_providers=(),
        entry_point=func_name,
        file_cache_dir=cache_dir,
    )
