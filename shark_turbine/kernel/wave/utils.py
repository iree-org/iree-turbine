from ..compiler.ir import (
    builtin_d,
    InsertionPoint,
    Location,
    Operation,
    transform_d,
    UnitAttr,
)
from typing import Callable
from .._support.tracing import CapturedTrace
from .._support.indexing import IndexExpr, IndexingContext, IndexSymbol
from ..lang.global_symbols import *
from ..ops.wave_ops import get_custom, Output, Write, MMA
from .constraints import HardwareConstraint
import torch.fx as fx

from iree.compiler.dialects.transform import (
    interpreter as transform_interpreter,
    any_op_t,
)

import sympy


def canonicalize_module(module: Operation):
    with module.context, Location.unknown():
        transform_module = builtin_d.Module.create()
        transform_module_op = module.operation
        transform_module_op.attributes["transform.with_named_sequence"] = UnitAttr.get()
        with InsertionPoint(transform_module.body):
            named_sequence = transform_d.NamedSequenceOp(
                "__transform_main", [any_op_t()], []
            )
            with InsertionPoint(named_sequence.body):
                target = named_sequence.body.arguments[0]
                apply_patterns = transform_d.ApplyPatternsOp(target)
                with InsertionPoint(apply_patterns.regions[0].blocks[0]):
                    transform_d.apply_patterns_canonicalization()
                transform_d.apply_cse(target)
                transform_d.YieldOp([target])
        transform_interpreter.apply_named_sequence(
            module,
            transform_module.body.operations[0],
            transform_module,
        )


def run_test(func: Callable[[], None]) -> Callable[[], None]:
    """Run a function as part of the test suite."""
    func()
    # Print a separator between tests
    print("-----")
    return func


def print_trace(trace: CapturedTrace, custom_print: bool = True):
    """
    Prints all subgraphs of a trace starting with the root graph.
    The graphs are printed first in the torch printing format and
    then using our custom node format.
    """
    # The root graph is at the back so we print the subgraphs in reverse order
    for subgraph in reversed(list(trace.region_graph.subgraphs.values())):
        print(subgraph)
        if custom_print:
            for node in subgraph.nodes:
                print(get_custom(node))


def DCE(trace: CapturedTrace):
    """
    Removes all operators that are not used in the graph,
    excluding output and global write nodes.
    Repeats this process till no more operators can be removed.
    """

    def is_removable_operator(node: fx.Node) -> bool:
        custom = get_custom(node)
        idxc = IndexingContext.current()
        is_global_write = (
            isinstance(custom, Write)
            and custom.type.address_space.subs(idxc.subs) == GLOBAL_ADDRESS_SPACE
        )
        return (
            not custom.users and not isinstance(custom, Output) and not is_global_write
        )

    while removable_nodes := trace.walk(is_removable_operator):
        for node in removable_nodes:
            get_custom(node).graph.erase_node(node)


def delinearize_index(index: IndexExpr, shape: list[int]) -> list[IndexExpr]:
    """
    Delinearizes a 1D index into a multi-dimensional index
    based on the shapes provided. The returned array contains
    the multi-dimensional index.

    Assume the index is x and the shape is [5, 4, 3]. In this case,
    this function returns [x % 3, (x // 3) % 4, (x // 12) % 5].

    """
    nd_index = []
    product = 1
    for i, size in enumerate(reversed(shape)):
        if i == 0:
            nd_index.append(index % size)
        else:
            nd_index.append(sympy.floor(index / product) % size)
        product *= size
    return nd_index[::-1]


def simplify_index(index: IndexExpr) -> IndexExpr:
    """
    Simplifies the index by applying the following bindings:
        - MMA acc_index bindings so the index of the MMA node is the acc_index.
    """
    mapping = {MMA_LHS: 0, MMA_RHS: 0, MMA_ACC: 1}
    return index.subs(mapping)


def get_mma_dimensional_mapping(trace: CapturedTrace) -> dict[IndexSymbol, int]:
    """
    Given a trace, determine the MMA dimensional mapping for all the
    MMA operations in the graph. For example, if we have
        acc = tkw.mma(a_reg, b_reg, acc)
    where a_reg has shape UxV, b has shape SxV and acc has shape UxS,
    we map U to the MMA M dimension (0), S to the MMA N dimension (1) and
    V to the MMA K dimension (2).
    """

    def is_mma(node):
        return isinstance(get_custom(node), MMA)

    mapping: dict[IndexSymbol, int] = {}
    for node in trace.walk(is_mma):
        custom: MMA = get_custom(node)
        m, n = custom.acc_type.symbolic_shape[-2:]
        lhs_shape = custom.lhs_type.symbolic_shape
        rhs_shape = custom.rhs_type.symbolic_shape
        acc_shape = custom.acc_type.symbolic_shape
        k = ((set(lhs_shape) & set(rhs_shape)) - set(acc_shape)).pop()
        mapping[m] = 0
        mapping[n] = 1
        mapping[k] = 2

    return mapping


def get_hardware_vector_size(
    dim: IndexSymbol,
    hardware_constraint: HardwareConstraint,
    mma_indices: dict[IndexSymbol, int],
) -> dict[IndexSymbol, int]:
    """
    Given a hardware constraint, return the vector sizes for the given dimension.
    This could be a hardware specific vector size or a user specified vector size.
    """
    if mma_indices:
        vector_size = hardware_constraint.mma_matrix_shapes[mma_indices[dim]]
    else:
        vector_size = hardware_constraint.vector_shapes[dim]
    return vector_size
