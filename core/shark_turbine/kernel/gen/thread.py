from typing import (
    Generic,
    Optional,
    Type,
    TypeVar,
    Callable,
    Union,
    cast,
    Any,
)

import inspect
import math

import torch.fx as fx
from torch.export import export

from ..lang import (
    KernelBuffer,
    Grid,
    IndexExpr,
)

from .._support.tracing import (
    CapturedTrace,
    CompiledContext,
    EagerContext,
    KernelTracer,
    Launchable,
    KernelRegionGraph,
)

__all__ = [
    "thread",
    "parameterize",
]

TCallable = TypeVar("TCallable", bound=Callable)


def parameterize(parameters: dict[str, Any]):
    def decorator(f: Optional[TCallable]) -> Optional[TCallable]:
        if f is not None:
            f.parameters = parameters
        return f

    return decorator


def thread(*symbolic_shape: IndexExpr):
    GridType = Grid[symbolic_shape]

    def decorator(f: Optional[TCallable] = None) -> "UnconfiguredThread[TCallable]":
        # Eagerly capture the trace and attach it to the wrapped function.
        region_graph = KernelRegionGraph()
        with CompiledContext(region_graph, grid_type=GridType) as context:
            with region_graph.subtracer() as subtracer:
                root_name, implicit_capture = subtracer.trace(f)
                trace = CapturedTrace(region_graph, root_name)

        return UnconfiguredThread[TCallable](GridType, f.__name__, f, trace)

    return decorator


class UnconfiguredThread(Generic[TCallable]):
    def __init__(
        self,
        grid_type: Type[Grid],
        name: str,
        wrapped_f: TCallable,
        trace: CapturedTrace,
    ):
        self.grid_type = grid_type
        self._name = name
        self._wrapped_f = wrapped_f
        self._trace = trace

    def __getitem__(self, grid: Union[int, tuple[int]]) -> TCallable:
        if not isinstance(grid, tuple):
            grid = (grid,)
        assert isinstance(grid, tuple) and all(isinstance(i, int) for i in grid)
        grid = self.grid_type(*grid)
        return cast(
            TCallable, LaunchableThread(grid, self._name, self._wrapped_f, self._trace)
        )

    def __repr__(self):
        return f"tk.gen.thread @{self._name}[no grid]"


class LaunchableThread(Launchable):
    def __init__(
        self, grid: Grid, name: str, eager_function: Callable, trace: CapturedTrace
    ):
        super().__init__(eager_function)
        self.grid = grid
        self.grid_type = type(grid)
        self._name = name
        self._trace = trace
        self._sig = inspect.signature(eager_function)

    def eager_execute(self, args, kwargs):
        grid = self.grid
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

    def __repr__(self):
        return f"tk.gen.thread @{self._name}[{', '.join(str(i) for i in self.grid)}]"
