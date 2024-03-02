from typing import (
    Type,
    Callable,
    Optional,
)

import inspect
import math

import torch

from ..lang import (
    KernelBuffer,
    Grid,
    IndexExpr,
)

from .._support.tracing import (
    CapturedTrace,
    CompiledContext,
    EagerContext,
    Launchable,
    KernelRegionGraph,
    LaunchContext,
    AOTLaunchContext,
)

from .._support.indexing import IndexingContext

from ..compiler import (
    kernel_codegen,
    dispatch_codegen,
    builder,
    vector_codegen,
    host_codegen,
)

from ..compiler.ir import (
    Context,
    Operation,
)

__all__ = [
    "thread",
]


def thread(*symbolic_shape: IndexExpr):
    GridType = Grid[symbolic_shape]

    def decorator(f: Callable) -> "LaunchableThread":
        return LaunchableThread(GridType, f.__name__, f)

    return decorator


class LaunchableThread(Launchable):
    def __init__(
        self,
        grid_type: Type[Grid],
        name: str,
        eager_function: Callable,
    ):
        super().__init__(eager_function)
        self.grid_type = grid_type
        self._name = name
        self._f = eager_function
        self._sig = inspect.signature(eager_function)

    def _trace(self) -> CapturedTrace:
        region_graph = KernelRegionGraph()
        with CompiledContext(region_graph, grid_type=self.grid_type) as context:
            with region_graph.subtracer() as subtracer:
                root_name, _ = subtracer.trace(self._f)
                trace = CapturedTrace(region_graph, root_name)
        return trace

    def eager_execute(self, args, kwargs):
        grid = self.grid_type()
        rank = grid.rank
        with EagerContext(rank=rank) as context:
            sig = self._sig
            bound = sig.bind(*args, *kwargs)
            bound.apply_defaults()
            # Transform args to KernelBuffers.
            for arg_name in list(bound.arguments.keys()):
                arg_value = bound.arguments[arg_name]
                param = sig.parameters[arg_name]
                param_type = param.annotation
                if isinstance(param_type, type) and issubclass(
                    param_type, KernelBuffer
                ):
                    kernel_buffer = param_type(arg_value)
                    bound.arguments[arg_name] = kernel_buffer
            volume = math.prod(grid)
            current_thread = context.current_thread
            for it in range(volume):
                for i in range(rank - 1):
                    current_thread[i] = it // grid[i]
                    it = it % grid[i]
                current_thread[-1] = it
                self._eager_function(*bound.args, **bound.kwargs)

    def _trace_and_get_kernel_signature(
        self,
        args,
        kwargs,
        context: Optional[Context] = None,
        module_op: Optional[Operation] = None,
    ):
        # Trace the function.
        trace = self._trace()
        idxc = IndexingContext.current()

        sig = self._sig
        bound = sig.bind(*args, *kwargs)
        bound.apply_defaults()

        for arg_name in list(bound.arguments.keys()):
            arg_value = bound.arguments[arg_name]
            param = sig.parameters[arg_name]
            param_type = param.annotation
            if isinstance(param_type, type) and issubclass(param_type, KernelBuffer):
                assert isinstance(arg_value, torch.Tensor)
                idxc.bind_shaped(arg_name, param_type, list(arg_value.shape))

        idxc.finalize()

        kernel_sig = kernel_codegen.KernelSignature()
        kernel_sig.add_from_graph_placeholders(trace.get_root_graph())
        kernel_sig.add_grid(self.grid_type)

        grid = self.grid_type()

        mb = builder.ModuleBuilder(context=context, module_op=module_op)
        entrypoint_name = self._name
        exe = dispatch_codegen.StreamExecutable(mb, name=entrypoint_name)
        dispatch_entrypoint = exe.define_entrypoint(entrypoint_name, kernel_sig, grid)
        emitter = vector_codegen.ThreadEmitter(dispatch_entrypoint, trace)
        emitter.emit()
        emitter.finish()

        mb.module_op.verify()

        return mb, exe, kernel_sig, entrypoint_name

    def test_execute(self, args, kwargs):
        mb, exe, kernel_sig, entrypoint_name = self._trace_and_get_kernel_signature(
            args, kwargs
        )
        host_codegen.isolated_test_call(mb, exe, kernel_sig, entrypoint_name)

        print(mb.module_op.get_asm())

    def aot_execute(self, args, kwargs):
        launch_context = LaunchContext.current()
        assert isinstance(launch_context, AOTLaunchContext)

        module = launch_context.module

        mb, exe, kernel_sig, entrypoint_name = self._trace_and_get_kernel_signature(
            args, kwargs, context=module.context, module_op=module.operation
        )

    def __repr__(self):
        return f"tk.gen.thread @{self._name}[{self.grid_type}]"
