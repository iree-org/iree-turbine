import inspect
import hashlib
import warnings

import torch

from ...aot import export, CompiledModule
from ..cache import (
    RUNTIME_CACHE,
    KernelCacheManager,
    KernelNamespace,
    is_cache_enabled,
)
from ..compiler.kernel_codegen import KernelBufferUsage
from ...importers.ir import Attribute, MLIRError
from ...runtime import Device, Launchable
from iree.runtime import VmContext, VmInstance, VmModule, VmFunction
from typing import Any, Callable
from iree.compiler.api import Session, Source, Output


def torch_fusion(eager_function):
    class Mod(torch.nn.Module):
        def forward(self, *args, **kwargs):
            return eager_function(*args, **kwargs)

    loader_meta = CacheEnabledEagerLoader(f"{eager_function.__name__}", Mod())

    def f(*args, device: Device | None = None):
        loader = loader_meta.get_loader(args)
        launchable = Launchable(loader)
        return launchable(*args, device)

    return f


@KernelNamespace.add("boo")
def get_hash(module: torch.nn.Module, sample_args, device_type):
    sample_arg_types = [(a.shape, a.dtype) for a in sample_args]
    key = [
        inspect.getsource(module.__class__),
        module.__getstate__(),
        device_type,
        sample_arg_types,
    ]
    return hashlib.sha1(str(key).encode("utf-8")).hexdigest()


RUNTIME_CACHE: dict[tuple[str, str], tuple[VmContext, VmFunction]] = {}




class CacheEnabledEagerLoader:
    def __init__(self, name: str, nn_module: torch.nn.Module):
        self._name = name
        self._m = nn_module
        self._cache_enabled = is_cache_enabled()

    def get_mlir(self, input_args):
        e = export(self._m, args=input_args, function_name=self._name)
        e.import_to("import")
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
        return mod

    def get_loader(self, input_args):
        def _load(device: Device):
            kernel_hash = None
            runtime_key = None
            main_module = None
            if self._cache_enabled:
                cache_manager = KernelCacheManager.get()
                kernel_hash = cache_manager.get_hash(
                    "boo",
                    self._m,
                    input_args,
                    device.type_cache_key,
                )
                device_key = device.instance_cache_key
                runtime_key = (device_key, kernel_hash)
                if runtime_key in RUNTIME_CACHE.keys():
                    return RUNTIME_CACHE[runtime_key]
                main_module = self._cache_load(device, kernel_hash)

            if not main_module:
                session = Session()
                mlir_module = self.get_mlir(input_args)
                source = Source.wrap_buffer(session, str(mlir_module).encode())
                session.set_flags(*device.compile_target_flags)
                inv = session.invocation()
                output = Output.open_membuffer()
                inv.parse_source(source)
                if not inv.execute():
                    # TODO: Capture diagnostics and report.
                    raise RuntimeError(f"JIT compilation failed. See diagnostics.")
                inv.output_vm_bytecode(output)
                mapped_memory = output.map_memory()
                vm_instance = device.vm_instance
                # TODO: VmModule.wrap_buffer would be better here, but it is still
                # unreliable capturing mapped memory from the compiler.
                # See: https://github.com/iree-org/iree/issues/17403
                main_module = VmModule.copy_buffer(vm_instance, mapped_memory)
            vm_instance = device.vm_instance
            ctx = VmContext(vm_instance, [device.create_hal_module(), main_module])
            func = main_module.lookup_function(self._name)
            if self._cache_enabled and runtime_key not in RUNTIME_CACHE:
                RUNTIME_CACHE[runtime_key] = (ctx, func)
            # TODO: save to file cache. Honestly, just make a new cache_manager?
            # if self._cache_enabled and kernel_hash not in cache_manager.file_cache["boo"]:
            return ctx, func

        return _load

    def _cache_load(self, device: Device, kernel_hash):
        cache_manager = KernelCacheManager.get()
        cached_kernel_entry = cache_manager.load_kernel("boo", self._name, kernel_hash)
        if not cached_kernel_entry:
            return

        vmfb = cached_kernel_entry.vmfb

        vm_instance = device.vm_instance
        mod = VmModule.copy_buffer(vm_instance, vmfb)
        return mod

