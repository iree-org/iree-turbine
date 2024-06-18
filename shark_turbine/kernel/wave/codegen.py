from typing import Any, Callable, ClassVar, Optional
from dataclasses import dataclass
import torch.fx as fx

from ..ops.wave_ops import write, register, mma, read, reduction
from ..compiler.base import CodegenError
from ..compiler.ir import InsertionPoint, Location
from ..compiler.kernel_codegen import BoundKernelSignature
from .._support.tracing import CapturedTrace


@dataclass
class WaveEmitter:
    """Emits a warp function as a `func` with a signature derived from the gm."""

    root_sig: BoundKernelSignature
    trace: CapturedTrace
    ip: InsertionPoint = None
    OP_HANDLERS: ClassVar[dict[str, Callable[["WaveEmitter", fx.Node], None]]] = {}

    def __post_init__(self):
        self.ip = InsertionPoint(self.root_sig.entry_block)

    def emit(self, graph: Optional[fx.Graph] = None):
        with self.ip, Location.unknown():
            self._emit_graph(
                graph if graph is not None else self.trace.get_root_graph()
            )

    def _emit_graph(self, graph: fx.Graph):
        """Emits the given graph at the current insertion point."""
        for node in graph.nodes:
            if node.op == "call_function" or node.op == "call_method":
                self._emit_function_call_node(node)

    def _emit_function_call_node(self, node: fx.Node):
        target_op = node.tkw_op_name
        try:
            handler = self.OP_HANDLERS[target_op]
        except KeyError:
            raise CodegenError(f"No handler registered for op {target_op}")

        handler(self, node)


def handle_op(op: Callable[..., Any]):
    def decorator(
        f: Callable[[WaveEmitter, fx.Node], None]
    ) -> Callable[[WaveEmitter, fx.Node], None]:
        WaveEmitter.OP_HANDLERS[op.__name__] = f
        return f

    return decorator


###############################################################################
# Memory Ops
###############################################################################


@handle_op(register)
def handle_register(emitter: WaveEmitter, node: fx.Node):
    raise NotImplementedError("Register: Currently only stub implementation")


@handle_op(read)
def handle_read(emitter: WaveEmitter, node: fx.Node):
    raise NotImplementedError("Read: Currently only stub implementation")


@handle_op(write)
def handle_write(emitter: WaveEmitter, node: fx.Node):
    raise NotImplementedError("Write: Currently only stub implementation")


###############################################################################
# Math Ops
###############################################################################


@handle_op(mma)
def handle_mma(emitter: WaveEmitter, node: fx.Node):
    raise NotImplementedError("MMA: Currently only stub implementation")


###############################################################################
# Control Flow ops
###############################################################################


@handle_op(reduction)
def handle_reduction(emitter: WaveEmitter, node: fx.Node):
    raise NotImplementedError("Reduction: Currently only stub implementation")
