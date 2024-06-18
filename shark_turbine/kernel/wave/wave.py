from typing import Any, Callable, Optional, Type
import inspect
import os

from ..compiler import builder, dispatch_codegen, kernel_codegen
from ..compiler.ir import Context, Operation
from .codegen import WaveEmitter
from ..lang import Grid
from ..ops import wave_ops
from .._support.tracing import (
    CapturedTrace,
    CompiledContext,
    KernelRegionGraph,
    Launchable,
)

__all__ = ["wave", "wave_trace_only"]


def wave():
    def decorator(f: Callable[..., Any]) -> "LaunchableWave":
        return LaunchableWave(f.__name__, f)

    return decorator


def wave_trace_only():
    def decorator(f: Callable[..., Any]) -> "Callable[[], CapturedTrace]":
        wave = LaunchableWave(f.__name__, f)
        return wave._trace  # type: ignore

    return decorator


class LaunchableWave(Launchable):
    def __init__(
        self,
        name: str,
        eager_function: Callable[[Any], Any],
    ):
        super().__init__(eager_function)

        self.grid_type = Grid[None, None]
        self._name = name
        self._f = eager_function
        self._sig = inspect.signature(eager_function)

    def _trace(self) -> CapturedTrace:
        region_graph = KernelRegionGraph()
        with CompiledContext(region_graph, grid_type=self.grid_type) as context:
            # Get all explictly defined custom ops
            custom_ops: dict[str, wave_ops.CustomOp] = {
                cls.tkw_op_name: cls
                for name, cls in inspect.getmembers(wave_ops, inspect.isclass)
                if issubclass(cls, wave_ops.CustomOp) and hasattr(cls, "tkw_op_name")
            }

            # Register custom ops
            for name, op in custom_ops.items():
                context.register_custom_op(name, op)

            with region_graph.subtracer() as subtracer:
                root_name, _ = subtracer.trace(self._f)
                trace = CapturedTrace(region_graph, root_name)

        return trace

    def _trace_and_get_kernel_signature(
        self,
        args,
        kwargs,
        context: Optional[Context] = None,
        module_op: Optional[Operation] = None,
    ) -> CapturedTrace:
        # Trace the function.
        trace = self._trace()

        kernel_sig = kernel_codegen.KernelSignature()
        # Fixed values for now, will be determined through constraints
        self.grid_type.dims = [32, 32]  # Will be determined by constraints
        grid = self.grid_type

        mb = builder.ModuleBuilder(context=context, module_op=module_op)
        entrypoint_name = self._name
        exe = dispatch_codegen.StreamExecutable(mb, name=entrypoint_name)
        dispatch_entrypoint = exe.define_entrypoint(entrypoint_name, kernel_sig, grid)

        emitter = WaveEmitter(dispatch_entrypoint, trace)
        emitter.emit(trace.get_root_graph())

        return trace

    def test_execute(self, args, kwargs):
        # For now only tracing
        self._trace_and_get_kernel_signature(args, kwargs)

    def aot_execute(self, args, kwargs):
        raise NotImplementedError("AOT execution for wave not implemented yet.")

    def eager_execute(self, args, kwargs):
        raise NotImplementedError("Eager execution for wave not implemented yet.")

    def __repr__(self):
        return f"tk.wave @{self._name}[{self.grid_type}]"
