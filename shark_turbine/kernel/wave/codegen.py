import operator
import sympy
from typing import Any, Callable, ClassVar, Optional, List, Type
from dataclasses import dataclass
import torch.fx as fx

from ..compiler.ir import (
    InsertionPoint,
    Location,
    OpResult,
    IrType,
    Value,
    IndexType,
    MemRefType,
    ShapedType,
    VectorType,
    IntegerAttr,
    arith_d,
    func_d,
    gpu_d,
    stream_d,
    vector_d,
)
from shark_turbine.aot.support.ir_utils import _is_float_type, _is_integer_like_type

# TK infrastructure imports.
from shark_turbine.kernel.lang.global_symbols import *
from ..ops.wave_ops import write, register, mma, read, reduction
from ..compiler.base import CodegenError, ValidationError, NDEBUG
from ..compiler.kernel_codegen import BoundKernelSignature
from .._support.tracing import CapturedTrace
from ..compiler.builder import IRProxyValue
from ..compiler.vector_codegen import (
    cast_kernel_buffer,
    cast_py_literal,
    cast_py_value,
    cast_vector,
)

# Indexing imports.
from .._support.indexing import IndexingContext


@dataclass
class WaveEmitter:
    """Emits a warp function as a `func` with a signature derived from the gm."""

    root_sig: BoundKernelSignature
    trace: CapturedTrace
    ip: InsertionPoint = None
    OP_HANDLERS: ClassVar[dict[str, Callable[["WaveEmitter", fx.Node], None]]] = {}
    _node_values: ClassVar[dict[fx.Node, List[IRProxyValue]]] = {}

    def __post_init__(self):
        self.ip = InsertionPoint(self.root_sig.entry_block)

    def emit_program_invariants(self):
        self.workgroup_ids = [
            stream_d.dispatch_workgroup_id(IntegerAttr.get(IndexType.get(), 0)),
            stream_d.dispatch_workgroup_id(IntegerAttr.get(IndexType.get(), 1)),
            stream_d.dispatch_workgroup_id(IntegerAttr.get(IndexType.get(), 2)),
        ]
        self.thread_ids = [
            gpu_d.thread_id(gpu_d.Dimension.x),
            gpu_d.thread_id(gpu_d.Dimension.y),
            gpu_d.thread_id(gpu_d.Dimension.z),
        ]

    def emit(self, graph: Optional[fx.Graph] = None):
        with self.ip, Location.unknown():
            self.emit_program_invariants()
            self._emit_graph(
                graph if graph is not None else self.trace.get_root_graph()
            )

    def finish(self):
        with self.ip, Location.unknown():
            func_d.ReturnOp([])

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

    def lookup_node_values(self, node: fx.Node) -> List[Value]:
        assert NDEBUG or isinstance(node, fx.Node)
        values = self._node_values.get(node)
        if values is None:
            values = [self.root_sig.resolve_by_reference(("node", node))]
            self._node_values[node] = values
        return values

    def bind_node_proxy(self, node: fx.Node, proxy: IRProxyValue):
        """Binds a node's result to a Python/IR proxy object."""
        assert NDEBUG or (isinstance(node, fx.Node) and isinstance(proxy, IRProxyValue))
        self._node_values[node] = [proxy]


def get_type_or_element_type(operand_type: IrType):
    assert isinstance(operand_type, IrType)
    if isinstance(operand_type, ShapedType):
        return operand_type.element_type
    else:
        return operand_type


def gen_sympy_index(emitter: WaveEmitter, expr: sympy.Expr) -> OpResult:
    stack: list[OpResult] = []

    # TODO: factor this out
    all_symbols = emitter.thread_ids + emitter.workgroup_ids
    dynamics = dict(
        zip(
            [THREAD_0, THREAD_1, THREAD_2, WORKGROUP_0, WORKGROUP_1, WORKGROUP_2],
            all_symbols,
        )
    )

    idxc = IndexingContext.current()
    # Substitute in frozen vars to simplify expression.
    if not isinstance(expr, sympy.Expr):
        expr = sympy.sympify(expr)
    expr = expr.subs(idxc.subs)
    # Why affine, for now simply create indexing expressions.
    # This can easily be adapted to affine expressions later.
    for term in sympy.postorder_traversal(expr):
        match term:
            case sympy.Symbol():
                if term in idxc.subs.keys():
                    cst = arith_d.constant(IndexType.get(), idxc.subs[term])
                    stack.append(cst)
                elif term in dynamics.keys():
                    stack.append(dynamics[term])
                else:
                    raise CodegenError(f"Unknown symbol {term}")
            case sympy.Integer():
                stack.append(arith_d.constant(IndexType.get(), int(term)))
            case sympy.Mul():
                args = []
                for _ in range(len(term.args)):
                    args.append(stack.pop())
                operation = None
                # First, multiply all the non-rationals.
                for arg in args:
                    if callable(arg):
                        continue
                    if operation is None:
                        operation = arg
                        continue
                    operation = arith_d.MulIOp(operation, arg)
                # Then, multiply with the rationals.
                for arg in args:
                    if callable(arg):
                        operation = arg(operation)
                stack.append(operation)
            case sympy.Add():
                summand = stack.pop()
                add = summand
                for _ in range(1, len(term.args)):
                    add = arith_d.AddIOp(add, stack.pop())
                stack.append(add)
            case sympy.Mod():
                rhs = stack.pop()
                lhs = stack.pop()
                mod = arith_d.RemSIOp(lhs, rhs)
                stack.append(mod)
            case sympy.floor():
                # TODO: Since divsi rounds to zero, this seems to work.
                # But check whether floordivsi is needed.
                stack.append(stack.pop())
            case sympy.Rational():
                numerator = arith_d.constant(IndexType.get(), abs(term.p))
                denominator = arith_d.constant(IndexType.get(), abs(term.q))
                # Assumes that the negative term is always carried on the numerator
                if abs(term.p) > term.p:
                    zero = arith_d.constant(IndexType.get(), int(0))
                    numerator = arith_d.SubIOp(zero, numerator)
                mul = lambda x: x
                if abs(term.p) != 1:
                    mul = lambda x: arith_d.MulIOp(x, numerator)
                operation = lambda x: arith_d.DivSIOp(mul(x), denominator)
                stack.append(operation)
            case sympy.UnevaluatedExpr():
                continue
            case _:
                raise CodegenError(f"Can not handle {term} yet")
    if len(stack) != 1:
        raise CodegenError(f"Expected single result, got {len(stack)}")
    return stack[0]


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
    # This is similar to tkl.store with fixed start indices for now.
    try:
        memory, elements_per_thread, mapping = node.args
    except ValueError as e:
        raise ValidationError("Malformed arguments") from e

    assert mapping is None, "mapping is not supported yet"

    vector_shape = cast_py_literal(emitter, (elements_per_thread,))
    # memory has no IR node yet.
    kb_src, kb_ir_type, kb_py_type = cast_kernel_buffer(emitter, memory)

    if not hasattr(node, "index"):
        raise ValidationError("codegen expected read to have index attr.")

    start_indices = []
    for dim_indexing in node.index:
        start_indices.append(gen_sympy_index(emitter, node.index[dim_indexing].start))

    element_type = kb_ir_type.element_type
    vector_type = VectorType.get(vector_shape, element_type)
    result = vector_d.load(vector_type, kb_src, start_indices)
    emitter.bind_node_proxy(node, IRProxyValue(result))


@handle_op(write)
def handle_write(emitter: WaveEmitter, node: fx.Node):
    raise NotImplementedError("Write: Currently only stub implementation")


###############################################################################
# Math Ops
###############################################################################


@handle_op(mma)
def handle_mma(emitter: WaveEmitter, node: fx.Node):
    raise NotImplementedError("MMA: Currently only stub implementation")


@handle_op(operator.add)
def handle_add(emitter: WaveEmitter, node: fx.Node):
    try:
        lhs, rhs = node.args
    except ValueError as e:
        raise ValidationError("Malformed arguments") from e
    lhs = cast_py_value(emitter, lhs)
    rhs = cast_py_value(emitter, rhs)

    if lhs.ir_value.type != rhs.ir_value.type:
        raise ValidationError("Expected lhs and rhs to have same type.")
    element_type = get_type_or_element_type(lhs.ir_value.type)

    lhs = lhs.ir_value
    rhs = rhs.ir_value

    if _is_float_type(element_type):
        result = arith_d.addf(lhs, rhs)
    elif _is_integer_like_type(element_type):
        result = arith_d.addi(lhs, rhs)
    else:
        raise ValidationError(f"Found unhanlded operand type for add: {element_type}")

    emitter.bind_node_proxy(node, IRProxyValue(result))


@handle_op(operator.getitem)
def handle_getitem(emitter: WaveEmitter, node: fx.Node):
    raise NotImplementedError("getitem: Currently only stub implementation")


@handle_op(operator.neg)
def handle_neg(emitter: WaveEmitter, node: fx.Node):
    raise NotImplementedError("neg: Currently only stub implementation")


@handle_op(operator.sub)
def handle_sub(emitter: WaveEmitter, node: fx.Node):
    raise NotImplementedError("sub: Currently only stub implementation")


###############################################################################
# Control Flow ops
###############################################################################


@handle_op(reduction)
def handle_reduction(emitter: WaveEmitter, node: fx.Node):
    raise NotImplementedError("Reduction: Currently only stub implementation")
