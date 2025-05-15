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

__all__ = [
    "is_cache_enabled",
    "clear_cache_dir",
    "ConvLaunchableRuntimeCache",
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
BOO_TUNING_SPEC_PATH = os.environ.get(
    "BOO_TUNING_SPEC_PATH", str(Path(__file__).parent / "tuning_specs.mlir")
)


def is_cache_enabled() -> bool:
    return bool(BOO_CACHE_ON)


def clear_cache_dir():
    if not CACHE_BASE_DIR.is_dir():
        return
    shutil.rmtree(CACHE_BASE_DIR)


def _out_of_process_compile(
    func_name: str, key_hashes_and_flags: Sequence[Tuple[str, Sequence[str]]]
) -> Tuple[str, Tuple[bool]]:
    """Runs compilation via command line tool. Does not raise an exception if compilation fails."""
    mlir_path = CACHE_BASE_DIR / func_name / f"{func_name}.mlir"
    if not mlir_path.is_file():
        logger.debug("no mlir file found at %s", str(mlir_path))
        return func_name, tuple([False] * len(key_hashes_and_flags))
    success = []
    for key_hash, flags in key_hashes_and_flags:
        vmfb_path: Path = CACHE_BASE_DIR / func_name / f"{key_hash}.vmfb"
        if vmfb_path.is_file():
            logger.debug("found vmfb in cache: %s", str(vmfb_path))
            success.append(True)
            continue
        logger.debug("Compiling vmfb to cache: %s", str(vmfb_path))
        cl_list = (
            ["iree-compile"] + list(flags) + [str(mlir_path), "-o", str(vmfb_path)]
        )
        command = shlex.join(cl_list)
        (CACHE_BASE_DIR / func_name / f"compile_command_{key_hash}.txt").write_text(
            command
        )
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


def _user_flags_jit_callback(func_name: str, extra_flags, source: str):
    """VmModule callback for out-of-process compilation with extra flags provided.
    If boo cache is disabled, this will create temporary files for compilation."""

    def _compile(flags, mlir_path, vmfb_path):
        cl_list = ["iree-compile"] + flags + [str(mlir_path), "-o", str(vmfb_path)]
        command = shlex.join(cl_list)
        (vmfb_path.parent / f"compile_command_{vmfb_path.stem}.txt").write_text(command)
        ret = subprocess.run(command, capture_output=True, shell=True, timeout=10)
        if ret.returncode != 0:
            raise RuntimeError(
                f"Failed compilation with diagnostics: {ret.stderr.decode()}."
            )
        return vmfb_path.read_bytes()

    def callback(device):
        key_hash = device.get_type_key_hash()
        vmfb_path: Path = CACHE_BASE_DIR / func_name / f"{key_hash}.vmfb"
        vm_instance = device.vm_instance

        if is_cache_enabled() and vmfb_path.is_file():
            logger.debug("Loading vmfb from cache: %s", str(vmfb_path))
            vmfb = vmfb_path.read_bytes()
            return VmModule.copy_buffer(vm_instance, vmfb)

        flags = list(device.compile_target_flags) + list(extra_flags)

        if is_cache_enabled():
            mlir_path = CACHE_BASE_DIR / func_name / f"{func_name}.mlir"
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
    def get_launchable_cache():
        global _launchable_cache
        if "_launchable_cache" in globals():
            return _launchable_cache
        _launchable_cache = ConvLaunchableRuntimeCache()
        return _launchable_cache

    @staticmethod
    def clear():
        global _launchable_cache
        if not "_launchable_cache" in globals():
            return
        _launchable_cache.session_cache.clear()

    @staticmethod
    def set_cache_limit(new_cache_limit: int | None):
        global _launchable_cache
        if "_launchable_cache" in globals():
            _launchable_cache.cache_limit = new_cache_limit
            return
        _launchable_cache = ConvLaunchableRuntimeCache(new_cache_limit)


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
            entry_point=f"{func_name}$async",
        )
    elif BOO_TUNING_SPEC_PATH != "":
        module_asm = _get_module_asm(signature, func_name, use_custom=use_custom)
        launch = Launchable.from_vm_module(
            _user_flags_jit_callback(
                func_name,
                (
                    f"--iree-codegen-tuning-spec-path={BOO_TUNING_SPEC_PATH}",
                    "--iree-llvmgpu-set-workgroup-distribution-along=x",
                ),
                module_asm,
            ),
            entry_point=f"{func_name}$async",
        )
    else:
        module_asm = _get_module_asm(signature, func_name, use_custom=use_custom)
        launch = Launchable.jit_compile(
            module_asm,
            parameter_providers=(),
            entry_point=f"{func_name}$async",
            file_cache_dir=cache_dir,
        )
    launch_cache.add_to_session_cache(session_cache_key, launch)
    return launch
