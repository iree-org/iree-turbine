import inspect
import hashlib
import warnings

import torch

from ...aot import export, CompiledModule
from .._support.tracing import Launchable
from ..cache import (
    KernelCacheManager,
    KernelNamespace,
    NAMESPACE_REGISTRY,
    is_cache_enabled,
    invoke_vmfb,
    invoke_cached_kernel,
)
from ..compiler.kernel_codegen import KernelBufferUsage
from ...importers.ir import Attribute, MLIRError


def boo_kernel(eager_function):
    class Mod(torch.nn.Module):
        def forward(self, *args, **kwargs):
            return eager_function(*args, **kwargs)

    return BOOLaunchable(f"{eager_function.__name__}", Mod())


@KernelNamespace.add("boo")
def get_hash(module: torch.nn.Module, compile_config, run_config, sample_args):
    sample_arg_types = tuple([(a.shape, a.dtype) for a in sample_args])
    key = [
        inspect.getsource(module.__class__),
        module.__getstate__(),
        compile_config,
        run_config,
        sample_arg_types,
    ]
    return hashlib.sha256(str(key).encode("utf-8")).hexdigest()


class BOOLaunchable(Launchable):
    def __init__(
        self,
        name: str,
        nn_module: torch.nn.Module,
    ):
        super().__init__(nn_module.forward)
        self._name = name
        self._m = nn_module
        self._sig = inspect.signature(nn_module.forward)

    def infer_output_types(self, input_args):
        """fallback for when output_shape is not known"""
        pytorch_results = self._m(*input_args)
        if isinstance(pytorch_results, torch.Tensor):
            return ((pytorch_results.shape, pytorch_results.dtype),)
        return tuple([(r.shape, r.dtype) for r in pytorch_results])

    def eager_execute(self, args, kwargs):
        raise NotImplementedError("Unimplemented: eager execution.")

    def aot_execute(self, args, kwargs):
        raise NotImplementedError("Unimplemented: aot execution.")

    def test_execute(self, input_args, kwargs):
        run = True
        run_bench = False
        func_name = self._name
        create_vmfb_file = kwargs.get("create_vmfb_file", None)
        dynamic_symbols_map = kwargs.get("dynamic_symbols_map", {})
        dynamic_symbols = kwargs.get("dynamic_symbols", [])
        compile_config = kwargs.get("compile_config", None)
        run_config = kwargs.get("run_config", None)
        output_types = kwargs.get("output_types", None)
        if not output_types:
            output_types = self.infer_output_types(input_args)
        device = input_args[0].device
        output_args = tuple(
            [torch.zeros(s, dtype=dty, device=device) for (s, dty) in output_types]
        )
        all_args = input_args + output_args
        # When this is passed in from the user, we will populate it with the kernel hash.
        # It will always be returned with just one entry which is the hash of the kernel.

        # Get cached kernel when available.
        cache_enabled = is_cache_enabled()
        kernel_hash = None
        if cache_enabled:
            cache_manager = KernelCacheManager.get()
            if not kernel_hash:
                kernel_hash = cache_manager.get_hash(
                    "boo",
                    self._m,
                    compile_config,
                    run_config,
                    input_args,
                )
            cached_kernel = cache_manager.load_kernel("boo", func_name, kernel_hash)
            if cached_kernel and (run or run_bench):
                invoke_cached_kernel(
                    cached_kernel,
                    all_args,
                    run_config,
                    dynamic_symbols,
                    dynamic_symbols_map,
                    run,
                    run_bench,
                    inplace=False,
                )
                if len(output_args) == 1:
                    return output_args[0]
                return output_args

        # Recompile kernel from scratch if not found in cache.
        if not compile_config:
            raise ValueError("no compilation config provided")

        cm = export(self._m, args=input_args, function_name=func_name)

        # This will allow custom op expansion.
        cm.import_to("full")

        # Force into a single dispatch.
        try:
            CompiledModule.run_pass_pipeline(
                cm.compiled_module,
                "builtin.module(util.func(iree-preprocessing-make-single-dispatch))",
                enable_ir_printing=True,
            )
        except MLIRError:
            warnings.warn(
                f"Failed to apply `iree-preprocessing-make-single-dispatch`. Please try using a newer version of IREE."
            )

        # Delete pass pipeline application above and uncomment the following once https://github.com/iree-org/iree/pull/20314 lands
        # Try to attach a func.func attribute for forming a single dispatch:
        # func_op = cm.mlir_module.regions[0].blocks[0].operations[0]

        # try:
        #     with cm.mlir_module.context as ctx:
        #         pipeline_attr = Attribute.parse(
        #             '#util.preprocessing_pipeline<"iree-preprocessing-make-single-dispatch">'
        #         )
        #         func_op.attributes["preprocessing_pipeline"] = pipeline_attr
        # except MLIRError as e:
        #     warnings.warn(
        #         f"Failed to attach #util.preprocessing_pipeline attr to func op. Please try using a newer version of IREE."
        #     )

        if run or run_bench or create_vmfb_file:
            if compile_config.get("print_mlir", False):
                cm.print_readable()

            flags = set(compile_config.get("flags", []))

            flags = list(
                flags.union(
                    [
                        "--iree-vm-bytecode-module-strip-source-map=true",
                        "--iree-opt-strip-assertions=true",
                        "--iree-vm-target-truncate-unsupported-floats",
                    ]
                )
            )
            cm.session.set_flags(*flags)

            target_backends = compile_config.get("backends", ("llvm-cpu",))
            compiler_output = cm.compile(save_to=None, target_backends=target_backends)
            compiled_vmfb = compiler_output.map_memory().raw

            kernel_usages = (KernelBufferUsage.INPUT,) * len(input_args) + (
                KernelBufferUsage.OUTPUT,
            ) * len(output_args)

            if cache_enabled:
                cache_manager.store_kernel(
                    compiled_vmfb,
                    kernel_usages,
                    str(cm.mlir_module),
                    "boo",
                    func_name,
                    kernel_hash,
                )

            invoke_vmfb(
                compiled_vmfb,
                func_name,
                run_config,
                input_args,
                output_args,
                [],
                run,
                run_bench,
                inplace=False,
                kernel_hash=kernel_hash,
            )

        if len(output_args) == 1:
            return output_args[0]
        return output_args
