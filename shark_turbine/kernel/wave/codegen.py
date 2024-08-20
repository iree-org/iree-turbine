import operator
import sympy
from typing import Any, Callable, ClassVar, Optional, List, Type
from dataclasses import dataclass
import torch.fx as fx
import torch.utils._pytree as pytree

from ..compiler.ir import (
    Attribute,
    DenseElementsAttr,
    FloatAttr,
    IndexType,
    InsertionPoint,
    IntegerAttr,
    IntegerType,
    IrType,
    Location,
    MemRefType,
    OpResult,
    ShapedType,
    Value,
    VectorType,
    amdgpu_d,
    arith_d,
    func_d,
    gpu_d,
    memref_d,
    stream_d,
    scf_d,
    vector_d,
)
from shark_turbine.aot.support.ir_utils import _is_float_type, _is_integer_like_type

# TK infrastructure imports.
from shark_turbine.kernel.lang.global_symbols import *
from ..ops.wave_ops import (
    write,
    register,
    mma,
    read,
    reduction,
    get_custom,
    get_result,
    allocate,
    shared_memory_barrier,
    CustomOp,
)
from ..lang.wave_types import IndexMapping, IndexSymbol
from ..compiler.base import CodegenError, ValidationError, NDEBUG
from ..compiler.kernel_codegen import BoundKernelSignature
from .._support.tracing import CapturedTrace
from ..compiler.builder import IRProxyValue
from ..compiler.utils import strides_from_symbolic_shape
from ..compiler.vector_codegen import (
    cast_kernel_buffer,
    cast_py_literal,
    cast_py_value,
    cast_vector,
)
from .constraints import Constraint, HardwareConstraint, MMAType, TilingConstraint

# Indexing imports.
from .._support.indexing import IndexingContext, IndexExpr, IndexSequence


@dataclass
class WaveEmitter:
    """Emits a warp function as a `func` with a signature derived from the gm."""

    root_sig: BoundKernelSignature
    trace: CapturedTrace
    constraints: list[Constraint]
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
        self.induction_vars: dict[IndexSymbol, Value] = {}

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
            if node.op == "output":
                return node.args

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
        values = [v.ir_value if isinstance(v, IRProxyValue) else v for v in values]
        return values

    def bind_node_proxy(self, node: fx.Node, proxy: IRProxyValue):
        """Binds a node's result to a Python/IR proxy object."""
        assert NDEBUG or (isinstance(node, fx.Node) and isinstance(proxy, IRProxyValue))
        self._node_values[node] = [proxy]

    def bind_node_proxies(self, node: fx.Node, proxies: List[IRProxyValue]):
        assert NDEBUG or (
            isinstance(node, fx.Node)
            and all(isinstance(p, IRProxyValue) for p in proxies)
        )
        self._node_values[node] = proxies


def get_type_or_element_type(operand_type: IrType):
    assert isinstance(operand_type, IrType)
    if isinstance(operand_type, ShapedType):
        return operand_type.element_type
    else:
        return operand_type


def gen_sympy_index(emitter: WaveEmitter, expr: sympy.Expr) -> OpResult:
    stack: list[OpResult] = []

    induction_var_syms = []
    induction_vars = []
    for constraint in emitter.constraints:
        if isinstance(constraint, TilingConstraint):
            assert (
                constraint.dim in emitter.induction_vars
            ), f"Could not find induction var for {constraint.dim} dimension"
            induction_var_syms.append(constraint.induction_var)
            induction_vars.append(emitter.induction_vars[constraint.dim])

    # TODO: factor this out
    all_symbols = emitter.thread_ids + emitter.workgroup_ids + induction_vars
    dynamics = dict(
        zip(
            [THREAD_0, THREAD_1, THREAD_2, WORKGROUP_0, WORKGROUP_1, WORKGROUP_2]
            + induction_var_syms,
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


def get_constant_attr(value: Any, element_type: IrType) -> Attribute:
    if _is_integer_like_type(element_type):
        return IntegerAttr.get(element_type, int(value))
    if _is_float_type(element_type):
        return FloatAttr.get(element_type, float(value))
    raise CodegenError(f"Cannot create a constant attribute for type `{element_type}`")


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
    try:
        shape, dtype, value = node.args
    except ValueError as e:
        raise ValidationError("Malformed arguments") from e
    if hasattr(node, "thread_shape"):
        shape = [node.thread_shape]
    vector_shape = cast_py_literal(emitter, shape)
    element_type = IrType.parse(dtype.ir_type_asm())
    vector_type = VectorType.get(vector_shape, element_type)
    register = arith_d.ConstantOp(
        vector_type,
        DenseElementsAttr.get_splat(
            vector_type, get_constant_attr(value, element_type)
        ),
    ).result
    emitter.bind_node_proxy(node, IRProxyValue(register))


@handle_op(allocate)
def handle_allocate(emitter: WaveEmitter, node: fx.Node):
    try:
        shape, distributed_shape, dtype, address_space = node.args
    except ValueError as e:
        raise ValidationError("Malformed arguments") from e
    memref_shape = cast_py_literal(emitter, distributed_shape)
    element_type = IrType.parse(dtype.ir_type_asm())
    address_space = Attribute.parse("#gpu.address_space<workgroup>")
    memref_type = MemRefType.get(memref_shape, element_type, None, address_space)
    alloc = memref_d.alloc(memref_type, [], [])
    emitter.bind_node_proxy(node, IRProxyValue(alloc))


def _get_start_indices(
    emitter: WaveEmitter, src_indices: dict[IndexExpr, IndexSequence | IndexExpr]
) -> list[OpResult]:
    start_indices = []
    for dim_indexing in src_indices:
        i = src_indices[dim_indexing]
        if isinstance(i, IndexSequence):
            i = i.start
        start_indices.append(gen_sympy_index(emitter, i))

    return start_indices


def _compute_offset(indices: list[int], strides: list[int]) -> int:
    return int(sum(i * s for i, s in zip(indices, strides)))


def _get_symbolic_shape(node: fx.Node) -> tuple[IndexExpr]:
    return get_custom(node).type.symbolic_shape


def _is_identity_mapping(
    mapping: IndexMapping,
    input_shape: Optional[tuple[IndexExpr]] = None,
    output_shape: Optional[tuple[IndexExpr]] = None,
) -> bool:
    if not mapping.is_identity():
        return False

    if input_shape is not None and mapping.input_shape != input_shape:
        return False

    if output_shape is not None and mapping.output_shape != output_shape:
        return False

    return True


def _construct_gather_scatter_indices(
    emitter: WaveEmitter,
    symbolc_shape: tuple[IndexExpr],
    index: tuple[IndexExpr],
    mapping: IndexMapping,
    elements_per_thread: int,
    is_read: bool,
) -> tuple[OpResult, OpResult, OpResult]:
    # Apply symbolc_shape order to indices, e.g. if original mapping is
    # {M: iter(0), N: iter(1)} and symbolc_shape is (N, M), result will
    # be (iter(1), iter(0))
    if is_read:
        assert (
            mapping.is_output_identity()
        ), "non-identity output mapping is not supported yet"
        index_mapping = mapping.map_input_indices(symbolc_shape)
    else:
        assert (
            mapping.is_input_identity()
        ), "non-identity input mapping is not supported yet"
        index_mapping = mapping.map_output_indices(symbolc_shape)

    iters = mapping.iters

    # As we only support identity input/output mapping for now, we can directly
    # substitute iterators with corresponding expanded index.
    subs = [(sym, expr.start) for sym, expr in zip(iters.keys(), index.values())]

    # Contruct input/output index, substituting iterators in input mapping with
    # expanded index.
    result_index = {key: m.subs(subs) for key, m in zip(symbolc_shape, index_mapping)}

    strides = strides_from_symbolic_shape(IndexingContext.current(), symbolc_shape)
    offsets = []
    subs = [(sym, 0) for sym in iters.keys()]
    for i in range(elements_per_thread):
        # Update most-minor dim, i.e. in case of identity mapping it will
        # be equivalent to just vector.load
        subs[-1] = (subs[-1][0], i)
        indices = [int(i.subs(subs)) for i in index_mapping]
        offsets.append(
            IntegerAttr.get(IndexType.get(), _compute_offset(indices, strides))
        )

    start_indices = _get_start_indices(emitter, result_index)
    offsets_vec_type = VectorType.get([elements_per_thread], IndexType.get())

    offsets_vec = arith_d.ConstantOp(
        offsets_vec_type, DenseElementsAttr.get(offsets, offsets_vec_type)
    )

    mask_vec_type = VectorType.get([elements_per_thread], IntegerType.get_signless(1))

    mask = vector_d.constant_mask(mask_vec_type, [elements_per_thread])

    return start_indices, offsets_vec, mask


@handle_op(read)
def handle_read(emitter: WaveEmitter, node: fx.Node):
    # This is similar to tkl.store with fixed start indices for now.
    try:
        memory, elements_per_thread, mapping, _ = node.args
    except ValueError as e:
        raise ValidationError("Malformed arguments") from e

    vector_shape = cast_py_literal(emitter, (elements_per_thread,))
    # memory has no IR node yet.
    kb_src, kb_ir_type, kb_py_type = cast_kernel_buffer(emitter, memory)

    if not hasattr(node, "index"):
        raise ValidationError("codegen expected read to have index attr.")

    index = node.index

    element_type = kb_ir_type.element_type
    vector_type = VectorType.get(vector_shape, element_type)
    input_shape = _get_symbolic_shape(memory)
    if mapping is None or _is_identity_mapping(mapping, input_shape=input_shape):
        start_indices = _get_start_indices(emitter, index)
        result = vector_d.load(vector_type, kb_src, start_indices)
    else:
        start_indices, offsets_vec, mask = _construct_gather_scatter_indices(
            emitter=emitter,
            symbolc_shape=input_shape,
            index=index,
            mapping=mapping,
            elements_per_thread=elements_per_thread,
            is_read=True,
        )

        zero = int(0) if _is_integer_like_type(element_type) else float(0)
        zero = arith_d.ConstantOp(vector_type.element_type, zero)
        passthru = vector_d.splat(vector_type, zero)

        result = vector_d.gather(
            vector_type, kb_src, start_indices, offsets_vec, mask, passthru
        )

    emitter.bind_node_proxy(node, IRProxyValue(result))


@handle_op(write)
def handle_write(emitter: WaveEmitter, node: fx.Node):
    try:
        register, memory, elements_per_thread, mapping = node.args
    except ValueError as e:
        raise ValidationError("Malformed arguments") from e

    # memory has no IR node yet.
    kb_dest, kb_ir_type, kb_py_type = cast_kernel_buffer(emitter, memory)
    insert_vector = cast_vector(emitter, register, element_type=kb_ir_type.element_type)
    insert_type = VectorType(insert_vector.type)
    vector_shape = cast_py_literal(emitter, (elements_per_thread,))

    # TODO: Support elements_per_thread size mismatch and broadcasting

    assert (
        tuple(insert_type.shape) == vector_shape
    ), f"Shape doesn't match: {tuple(insert_type.shape)} and {(vector_shape)}"

    if not hasattr(node, "index"):
        raise ValidationError("codegen expected read to have index attr.")

    index = node.index
    input_shape = _get_symbolic_shape(register)
    output_shape = _get_symbolic_shape(memory)
    if mapping is None or _is_identity_mapping(
        mapping, input_shape=input_shape, output_shape=output_shape
    ):
        start_indices = _get_start_indices(emitter, index)
        vector_d.store(insert_vector, kb_dest, start_indices)
    else:
        assert (
            input_shape == mapping.input_shape
        ), "non-identity input mapping is not supported yet"

        start_indices, offsets_vec, mask = _construct_gather_scatter_indices(
            emitter=emitter,
            symbolc_shape=output_shape,
            index=index,
            mapping=mapping,
            elements_per_thread=elements_per_thread,
            is_read=False,
        )

        vector_d.scatter(kb_dest, start_indices, offsets_vec, mask, insert_vector)


###############################################################################
# Math Ops
###############################################################################


def emit_mfma(
    m: int, n: int, k: int, vector_type: VectorType, acc: Value, values: list[Value]
):
    m = get_constant_attr(m, IntegerType.get_signless(32))
    n = get_constant_attr(n, IntegerType.get_signless(32))
    k = get_constant_attr(k, IntegerType.get_signless(32))
    blocks = get_constant_attr(1, IntegerType.get_signless(32))

    result = amdgpu_d.mfma(
        dest_d=vector_type,
        m=m,
        n=n,
        k=k,
        blocks=blocks,
        source_a=values[0],
        source_b=values[1],
        dest_c=acc,
    )
    return result


@handle_op(mma)
def handle_mma(emitter: WaveEmitter, node: fx.Node):
    try:
        lhs, rhs, acc = node.args
        acc = cast_vector(emitter, acc)
        values = [cast_vector(emitter, val) for val in [lhs, rhs]]
    except ValueError as e:
        raise ValidationError("Malformed arguments") from e

    vector_type = VectorType(acc.type)

    hardware_constraints = [
        constraint
        for constraint in emitter.constraints
        if isinstance(constraint, HardwareConstraint)
    ]
    if not hardware_constraints:
        raise CodegenError("No hardware constraints found.")

    m, n, k = hardware_constraints[0].mma_matrix_shapes
    result = emit_mfma(m, n, k, vector_type, acc, values)
    emitter.bind_node_proxy(node, IRProxyValue(result))


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
    try:
        axis, init_args, subgraph, implicit_capture = node.args
    except ValueError as e:
        raise ValidationError("Malformed arguments") from e

    # Flatten init_args and get IR values for each of them.
    flat_init_args, _ = pytree.tree_flatten((init_args))
    flat_init_args = [cast_py_value(emitter, arg) for arg in flat_init_args]

    # Without scheduling, we assume that we always start at 0.
    start = arith_d.constant(IndexType.get(), int(0))

    idxc = IndexingContext.current()
    # For now, we assume that dimensions that have tiling constraints on them,
    # do not have any other constraints.
    dim = axis.subs(idxc.subs)
    end = arith_d.constant(IndexType.get(), int(dim))

    step = None
    for constraint in emitter.constraints:
        if isinstance(constraint, TilingConstraint) and constraint.dim == axis:
            tile_size = constraint.tile_size.subs(idxc.subs)
            step = arith_d.constant(IndexType.get(), int(tile_size))

    if not step:
        raise CodegenError(
            "Could not determine step size for reduction due to missing tiling constraint."
        )

    forOp = scf_d.ForOp(
        start,
        end,
        step,
        [a.ir_value for a in flat_init_args],
    )
    emitter.induction_vars[axis] = forOp.induction_variable
    with InsertionPoint(forOp.body):
        # Add mapping for iter args.
        subgraph: fx.Graph = emitter.trace.get_subgraph(subgraph)
        iter_args: list[fx.Node] = get_custom(node).iter_args(subgraph)
        for i, v in enumerate(forOp.inner_iter_args):
            emitter.bind_node_proxy(iter_args[i], IRProxyValue(v))
        captured_vars: list[fx.Node] = get_custom(node).captured_vars(subgraph)
        for root_v, subgraph_v in zip(implicit_capture, captured_vars):
            emitter._node_values[subgraph_v] = emitter.lookup_node_values(root_v)
        # Emit the subgraph.
        return_values = emitter._emit_graph(subgraph)
        # Flattern return values.
        flat_ret_values, _ = pytree.tree_flatten((return_values))
        flat_ret_values = [
            cast_py_value(emitter, value).ir_value for value in flat_ret_values
        ]
        scf_d.YieldOp(flat_ret_values)

    emitter.bind_node_proxies(node, [IRProxyValue(v) for v in forOp.results_])


###############################################################################
# Synchronization ops
###############################################################################


@handle_op(shared_memory_barrier)
def handle_shared_memory_barrier(emitter: WaveEmitter, node: fx.Node):
    amdgpu_d.lds_barrier()


###############################################################################
# Miscellanous ops
###############################################################################


@handle_op(get_result)
def handle_get_result(emitter: WaveEmitter, node: fx.Node):
    try:
        value, res_idx = node.args
    except ValueError as e:
        raise ValidationError("Malformed arguments") from e

    for_op = emitter.lookup_node_values(value)[0].owner
    emitter.bind_node_proxy(node, IRProxyValue(for_op.results[res_idx]))
