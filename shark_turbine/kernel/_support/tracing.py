from abc import ABC, abstractmethod
from typing import (
    Optional,
    TypeVar,
    Callable,
    Type,
    cast,
    Dict,
    Tuple,
)

from ..compiler.ir import Operation

import functools
import warnings

import torch.fx as fx

from .indexing import (
    backed_sym_index_type,
    BoundedRelation,
    IndexExpr,
    IndexSymbol,
    IndexingContext,
)

from ..lang.kernel_buffer import KernelBuffer
from ..lang.grid import Grid

from ..lang.types import (
    Index,
)
from ..lang.wave_types import IndexMapping
from ..ops.wave_ops import CustomOp, Placeholder, Reduction, Unknown

from .regions import RegionGraph, SubgraphTracer

from .. import ops
from ..ops.base import (
    OpDispatcher,
)

from . import context
from .dtype import DataType

try:
    from typing import assert_type
except ImportError:
    # No-op if not supported. Introduced in Python 3.11.
    def assert_type(a, b):
        pass


TCallable = TypeVar("TCallable", bound=Callable)

###############################################################################
# Kernel Region Graph
###############################################################################


class KernelRegionGraph(RegionGraph):
    def new_subtracer(
        self,
        region_graph: "RegionGraph",
        parent: Optional["SubgraphTracer"] = None,
    ) -> "KernelTracer":
        return KernelTracer(region_graph, parent=parent)


###############################################################################
# Tracing machinery
###############################################################################


class KernelBufferProxy(fx.Proxy):
    """Custom proxy for KernelBuffer so that we can override special methods."""

    def __init__(
        self,
        node: fx.Node,
        tracer: "KernelTracer",
        orig_type: Type[KernelBuffer],
    ):
        super().__init__(node, tracer)
        self._orig_type = orig_type
        # The shape and rank are statically available (not proxied).
        self.symbolic_shape = orig_type.symbolic_shape
        self.rank = orig_type.rank

    def __getitem__(self, key):
        return ops.kernel_buffer_getitem(self, key)

    def __setitem__(self, key, item):
        ops.kernel_buffer_setitem(self, key, item)


class KernelTracer(SubgraphTracer):
    """Custom Tracer for generating a trace of a kernel computation."""

    # Register our custom proxies.
    def proxy(self, node: fx.Node) -> fx.Proxy:
        t = node.type
        if t is not None:
            if issubclass(t, KernelBuffer):
                return KernelBufferProxy(node, self, t)
        return super().proxy(node)

    def create_arg(self, a):
        # Let IndexExpr persist as arguments.
        if isinstance(a, IndexExpr):
            return a
        # Let DataType persist as arguments.
        if isinstance(a, DataType):
            return a
        if isinstance(a, IndexMapping):
            return a
        return super().create_arg(a)


class CapturedTrace:
    def __init__(self, region_graph: RegionGraph, root_graph: str):
        self.region_graph = region_graph
        self.root_graph = root_graph

    def get_subgraph(self, name: str) -> fx.Graph:
        return self.region_graph.subgraphs[name]

    def get_root_graph(self) -> fx.Graph:
        return self.get_subgraph(self.root_graph)

    def walk(self, filter: Callable[[fx.Node], bool]) -> list[fx.Node]:
        nodes: list[fx.Node] = []
        for region in self.region_graph.subgraphs.values():
            for node in region.nodes:
                if filter(node):
                    nodes.append(node)
        return nodes


###############################################################################
# Execution context.
# A valid BaseContext derived instance (EagerContext or CompiledContext) must
# be active for any evaluation of a generated/traced function.
###############################################################################


class BaseContext(OpDispatcher):
    __tk_context_idname__ = "ExecutionContext"

    def __init__(self, *, eager: bool):
        self.eager = eager

    @staticmethod
    def current() -> "BaseContext":
        return context.current(BaseContext)

    def __enter__(self) -> "BaseContext":
        context.push(OpDispatcher, self)
        return context.push(BaseContext, self)

    def __exit__(self, exc_type, exc_val, exc_tb):
        context.pop(OpDispatcher, self)
        context.pop(BaseContext, self)


class EagerContext(BaseContext):
    def __init__(self, rank: int = 0):
        super().__init__(eager=True)
        self.rank = rank
        self.current_thread: list[int] = rank * [0]

    def handle_thread_program_id(self, op, axis: int) -> int:
        assert axis >= 0 and axis < self.rank
        return Index(self.current_thread[axis])

    def handle_kernel_buffer_getitem(self, op, kernel_buffer: KernelBuffer, key):
        return kernel_buffer._tensor.__getitem__(key)

    def handle_kernel_buffer_setitem(self, op, kernel_buffer: KernelBuffer, key, item):
        kernel_buffer._tensor.__setitem__(key, item)


class CompiledContext(BaseContext):
    def __init__(self, region_graph: RegionGraph, *, grid_type: Type[Grid]):
        super().__init__(eager=False)
        self.region_graph = region_graph
        self.grid_type = grid_type
        self.current_thread_types = [
            backed_sym_index_type(BoundedRelation(0, n, upper_inclusive=False))
            for n in grid_type.symbolic_shape
        ]

    def register_custom_op(self, name: str, op: CustomOp):
        def handler(*args, **kwargs):
            return op.handle(self.region_graph, *args, **kwargs)

        setattr(self, f"handle_{name}", handler)

    ### ========================================================================
    ### Core Operations
    ### ========================================================================

    def handle_thread_program_id(self, op, axis: int) -> Index:
        grid_types = self.current_thread_types
        if axis < 0 or axis >= len(grid_types):
            raise IndexError(
                f"Illegal index into grid of rank {len(grid_types)}: {axis}"
            )

        proxy = self.region_graph.create_proxy(
            "call_function",
            op,
            args=(axis,),
            kwargs={},
            type_expr=grid_types[axis],
        )
        return proxy

    def handle_to_dtype(self, op, val, dtype):
        return self.region_graph.create_proxy(
            "call_function",
            op,
            args=(val, dtype),
            kwargs={},
        )

    def handle_kernel_buffer_getitem(self, op, kernel_buffer: KernelBuffer, key):
        return self.region_graph.create_proxy(
            "call_function",
            op,
            args=(kernel_buffer, key),
            kwargs={},
        )

    def handle_kernel_buffer_setitem(self, op, kernel_buffer: KernelBuffer, key, item):
        self.region_graph.create_proxy(
            "call_function",
            target=op,
            args=(kernel_buffer, key, item),
            kwargs={},
        )

    ### ========================================================================
    ### Memory Operations
    ### ========================================================================
    def handle_kernel_buffer_load(self, op, kernel_buffer, multi_index, shape):
        return self.region_graph.create_proxy(
            "call_function",
            target=op,
            args=(kernel_buffer, multi_index, shape),
            kwargs={},
        )

    def handle_kernel_buffer_store(self, op, kernel_buffer, multi_index, item):
        self.region_graph.create_proxy(
            "call_function",
            target=op,
            args=(kernel_buffer, multi_index, item),
            kwargs={},
        )

    ### ========================================================================
    ### Control Flow Operations
    ### ========================================================================

    def handle_for_loop(self, op, start, stop=None, step=None, init_args=[]):
        if stop is None:
            stop = start
            start = 0
        if step is None:
            step = 1

        def wrapper(f):
            with self.region_graph.subtracer() as subtracer:
                subgraph_name, implicit_capture = subtracer.trace(f)
            # Create a call to this subgraph
            ret = self.region_graph.create_proxy(
                "call_function",
                target=op,
                name="for_loop",
                args=(start, stop, step, init_args),
                kwargs={
                    "subgraph": subgraph_name,
                    "implicit_capture": implicit_capture,
                },
            )
            return ret

        return wrapper

    ### ========================================================================
    ### Math Operations
    ### ========================================================================
    def handle_exp2(self, op, val):
        return self.region_graph.create_proxy(
            "call_function",
            target=op,
            args=(val,),
            kwargs={},
        )

    def handle_vector_constant(
        self, op, shape: Tuple[int, ...], dtype, value: int | float
    ):
        return self.region_graph.create_proxy(
            "call_function",
            target=op,
            args=(shape, dtype, value),
            kwargs={},
        )

    ### ========================================================================
    ### Reduction Operations
    ### ========================================================================
    def handle_vector_max(self, op, vector, axis=None, acc=None):
        return self.region_graph.create_proxy(
            "call_function",
            target=op,
            args=(vector, axis, acc),
            kwargs={},
        )

    def handle_vector_sum(self, op, vector, axis=None, acc=None):
        return self.region_graph.create_proxy(
            "call_function",
            target=op,
            args=(vector, axis, acc),
            kwargs={},
        )

    def handle_vector_dot(self, op, lhs, rhs, acc=None):
        return self.region_graph.create_proxy(
            "call_function",
            target=op,
            args=(lhs, rhs, acc),
            kwargs={},
        )

    ### ========================================================================
    ### Shape Manipulation Operations
    ### ========================================================================
    def handle_vector_broadcast(self, op, vector, leading_sizes):
        return self.region_graph.create_proxy(
            "call_function",
            target=op,
            args=(vector, leading_sizes),
            kwargs={},
        )

    def handle_vector_broadcast_in_dim(self, op, vector, shape, broadcast_dimensions):
        # Currently, we do not have a corressponding op in MLIR, so
        # we trace this to broadcast + transpose.
        # TODO: Add a vector dialect op for this in MLIR.

        # Remove broadcast_dimensions from shape.
        shape_with_leading = tuple(
            dim for i, dim in enumerate(shape) if i not in broadcast_dimensions
        )

        # Broadcast
        broadcasted_vector = self.region_graph.create_proxy(
            "call_function",
            target=ops.vector_broadcast,
            args=(vector, shape_with_leading),
            kwargs={},
        )

        # Get the permutation for the transpose.
        permutation = tuple(
            i for i in range(len(shape)) if i not in broadcast_dimensions
        )
        permutation = permutation + tuple(broadcast_dimensions)

        # Transpose
        return self.region_graph.create_proxy(
            "call_function",
            target=ops.vector_transpose,
            args=(broadcasted_vector, permutation),
            kwargs={},
        )

    def handle_vector_transpose(self, op, vector, permutation):
        return self.region_graph.create_proxy(
            "call_function",
            target=op,
            args=(vector, permutation),
            kwargs={},
        )


###############################################################################
# Launch context
# The launch context controls how the call into a kernel is dispatched.
# This can either be to run it eagerly for debugging or some higher order
# integration.
###############################################################################


class Launchable(ABC):
    """Base class for objects which behave like a kernel launch when called."""

    def __init__(self, eager_function: Callable):
        self._eager_function = eager_function

    def __call__(self, *args, **kwargs):
        launch_context = LaunchContext.current()
        return launch_context.launch(self, args, kwargs)

    @abstractmethod
    def eager_execute(self, args, kwargs):
        ...

    def aot_execute(self, args, kwargs):
        ...

    def test_execute(self, args, kwargs):
        ...


class LaunchContext(ABC):
    __tk_context_idname__ = "ExecutionContext"

    def __init__(self, constant_bindings: Dict[IndexSymbol, int] = {}):
        self.constant_bindings = constant_bindings

    @staticmethod
    def current() -> "LaunchContext":
        try:
            return context.current(LaunchContext)
        except IndexError:
            warnings.warn(
                "defaulting to debug/eager execution of tk kernel launch "
                "because no launch context has been established"
            )
            return DebugLaunchContext()

    def __enter__(self) -> "LaunchContext":
        # Push an indexing context with the constand bindings for this launch
        # context in it.
        # TODO: Is creating a IndexingContext as part of LaunchContext the
        # correct layering?
        idxc = IndexingContext()
        context.push(IndexingContext, idxc)
        for s, val in self.constant_bindings.items():
            idxc.bind_constant(s, val)
        return context.push(LaunchContext, self)

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Pop the indexing context created as part of this launch.
        # TODO: Is creating a IndexingContext as part of LaunchContext the
        # correct layering?
        context.pop(IndexingContext, IndexingContext().current())
        context.pop(LaunchContext, self)

    @abstractmethod
    def launch(self, launchable: Launchable, args, kwargs):
        ...


class DebugLaunchContext(LaunchContext):
    def launch(self, launchable: Launchable, args, kwargs):
        return launchable.eager_execute(args, kwargs)


class TestLaunchContext(LaunchContext):
    def launch(self, launchable: Launchable, args, kwargs):
        return launchable.test_execute(args, kwargs)


class AOTLaunchContext(LaunchContext):
    module: Operation

    def __init__(
        self, module: Operation, constant_bindings: Dict[IndexSymbol, int] = {}
    ):
        self.module = module
        super().__init__(constant_bindings)

    def launch(self, launchable: Launchable, args, kwargs):
        return launchable.aot_execute(args, kwargs)


###############################################################################
# Helpers
###############################################################################


def eager_context() -> EagerContext:
    context = BaseContext.current()
    assert context.eager, "Expected to be executed against an EagerContext"
    assert_type(context, EagerContext)
    return context


def custom_primitive_fn(
    f: Optional[TCallable] = None, *, compiled: Callable
) -> TCallable:
    """Decorator for a primitive function with a custom callback for tracing.

    The wrapped function will be invoked as-is when executing eagerly. When
    tracing, the `compiled` callback will be invoked with the same signature
    but with the `CompiledContext` added as a first postional argument.
    """
    if f is None:
        return functools.partial(custom_primitive_fn, compiled=compiled)

    @functools.wraps(f)
    def wrapper(*args, **kwargs):  # type: ignore
        context = BaseContext.current()
        if context.eager:
            return f(*args, **kwargs)
        else:
            assert_type(context, CompiledContext)
            return compiled(context, *args, **kwargs)

    return cast(TCallable, wrapper)
