"""Code generation support for kernel entry-points.

In a typical code generation stack, there are three elements:

1. Dispatch code-generation: Embeds executables into some overall
   program and coordinates launches.
2. Kernel code-generation: Handles device-side kernel signatures
   and global marshalling physical kernel inputs to logical
   kernel inputs and grid functions.
3. Low-level code-generation: Generates DMAs and compute operations
   based on a logical program.

This level handles #2.
"""

from typing import Any, Callable, Optional, Type

from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass

import torch.fx as fx

from .._support.indexing import (
    IndexingContext,
    IndexSymbol,
)

from ..lang.kernel_buffer import (
    KernelBuffer,
    KernelBufferUsage,
    is_kernel_buffer_meta_derived,
)
from ..lang.wave_types import Memory
from ..lang.grid import Grid
from ..ops.wave_ops import get_custom, Placeholder, Read, Write

from .base import (
    CodegenError,
)

from .builder import (
    ModuleBuilder,
)

from .ir import (
    Block,
    FunctionType,
    IndexType,
    InsertionPoint,
    IrType,
    Location,
    Operation,
    Value,
    func_d,
)

from .utils import strides_from_symbolic_shape


# Filter function to check for placeholder nodes.
def is_placeholder(node: fx.Node):
    custom = get_custom(node)
    return isinstance(custom, Placeholder)


# Util fn to filter nodes in a graph based on specfied filter fn.
def filter_fx_graph(graph: fx.Graph, filter: Callable[[fx.Node], bool]):
    filtered_nodes: list[fx.Node] = []
    for node in graph.nodes:
        if filter(node):
            filtered_nodes.append(node)
    return filtered_nodes


class BindingType(Enum):
    KERNEL_BUFFER = 0
    INDEX_VALUE = 1
    SYMBOL_VALUE = 2


@dataclass
class BindingDesc:
    # The unique reference object that this is derived from. This will
    # be different for each kind of argument:
    #   FX node placeholders: ('node', fx.Node)
    #   Grid indices: ('grid', n)
    reference: tuple[str, Any]

    # Discrimnator type of this argument.
    binding_type: BindingType

    # Debug name derived from the source, if available.
    name: Optional[str] = None

    # If an INPUT_BUFFER, OUTPUT_BUFFER, or TEMPORARY_BUFFER, then this
    # is the backing KernelBuffer type.
    kernel_buffer_type: Optional[Type[KernelBuffer]] = None

    # If a SYMBOL_VALUE, then this is the corresponding IndexSymbol.
    symbol_type: Optional[Type[IndexSymbol]] = None

    def as_mlir_type(self) -> IrType:
        idx_context = IndexingContext.current()

        def sym_to_dim_asm(s: IndexSymbol) -> str:
            static_value = idx_context.get_static_value(s)
            return "?" if static_value is None else str(static_value)

        binding_type = self.binding_type
        if binding_type == BindingType.KERNEL_BUFFER:
            kb_t = self.kernel_buffer_type  # type: KernelBuffer
            element_type_asm = kb_t.dtype.ir_type_asm()
            symbolic_shape = kb_t.symbolic_shape
            if symbolic_shape is not None:
                shape_asm = "x".join(sym_to_dim_asm(s) for s in kb_t.symbolic_shape)
                spec_asm = f"{shape_asm}x{element_type_asm}"
                strides = []
            else:
                # Unranked. Not well supported, but for completeness.
                spec_asm = element_type_asm
            strides = strides_from_symbolic_shape(idx_context, kb_t.symbolic_shape)
            if strides is None:
                memref_asm = f"memref<{spec_asm}>"
            else:
                memref_asm = f"memref<{spec_asm}, strided<{strides}, offset: ?>>"
            return IrType.parse(memref_asm)
        elif binding_type == BindingType.INDEX_VALUE:
            return IndexType.get()
        elif binding_type == BindingType.SYMBOL_VALUE:
            return IndexType.get()
        else:
            raise AssertionError("Unhandled switch BindingType")


class KernelSignature:
    def __init__(self):
        self.bindings: list[BindingDesc] = []

    @property
    def grid_bindings(self) -> list[BindingDesc]:
        """Gets all grid axis bindings."""
        return [b for b in self.bindings if b.reference[0] == "grid"]

    @property
    def kernel_buffer_input_bindings(self) -> list[BindingDesc]:
        """Gets all kernel buffer bindings with input usage."""
        return [
            b
            for b in self.bindings
            if b.binding_type == BindingType.KERNEL_BUFFER
            and b.kernel_buffer_type.usage == KernelBufferUsage.INPUT
        ]

    @property
    def kernel_buffer_output_bindings(self) -> list[BindingDesc]:
        """Gets all kernel buffer bindings with input usage."""
        return [
            b
            for b in self.bindings
            if b.binding_type == BindingType.KERNEL_BUFFER
            and b.kernel_buffer_type.usage == KernelBufferUsage.OUTPUT
        ]

    @property
    def kernel_buffer_temporary_bindings(self) -> list[BindingDesc]:
        """Gets all kernel buffer bindings with input usage."""
        return [
            b
            for b in self.bindings
            if b.binding_type == BindingType.KERNEL_BUFFER
            and b.kernel_buffer_type.usage == KernelBufferUsage.TEMPORARY
        ]

    def add_from_graph_placeholders(self, graph: fx.Graph):
        # Extract all placeholder nodes.
        placeholder_nodes = filter_fx_graph(graph, is_placeholder)

        # Create bindings for placeholder nodes.
        for node in placeholder_nodes:
            t = node.type
            if is_kernel_buffer_meta_derived(t):
                self.bindings.append(
                    BindingDesc(
                        ("node", node),
                        BindingType.KERNEL_BUFFER,
                        name=node.target,
                        kernel_buffer_type=t,
                    )
                )
            elif issubclass(t, IndexSymbol):
                self.bindings.append(
                    BindingDesc(
                        ("node", node),
                        BindingType.SYMBOL_VALUE,
                        name=node.target,
                        symbol_type=t,
                    )
                )
            else:
                raise ValueError(
                    f"Unsupported placeholder node type: {t} (for node {node})"
                )

    def add_grid(self, grid: Type[Grid]):
        assert grid.symbolic_shape, "code emission requires a symbolically shaped grid"
        for index, s in enumerate(grid.symbolic_shape):
            self.bindings.append(
                BindingDesc(
                    ("grid", index), BindingType.INDEX_VALUE, name=f"grid{index}"
                )
            )

    def determine_input_output_buffers(self, graph: fx.Graph):
        # Extract all placeholder nodes.
        placeholder_nodes = filter_fx_graph(graph, is_placeholder)

        def only_read_dependencies(node):
            return all([isinstance(get_custom(x), Read) for x in node.users.keys()])

        def only_write_dependencies(node):
            if len(node.users) == 0:
                return False
            return all([isinstance(get_custom(x), Write) for x in node.users.keys()])

        for node in placeholder_nodes:
            index = None
            for i, binding in enumerate(self.bindings):
                if binding.reference[1] == node:
                    index = i
                    break
            if index == None:
                continue

            # TODO: Match KernelBufferUsage to what bufferType that is expected on IREE.
            usage = KernelBufferUsage.INPUT
            if only_read_dependencies(node):
                usage = KernelBufferUsage.INPUT

            if only_write_dependencies(node):
                usage = KernelBufferUsage.OUTPUT

            # Create new Memory type with the correct usage
            memory_type = self.bindings[index].kernel_buffer_type
            self.bindings[index].kernel_buffer_type = Memory[
                *memory_type.symbolic_shape,
                memory_type.address_space,
                memory_type.dtype,
                usage,
            ]
        return

    def __repr__(self):
        parts = []
        for b in self.bindings:
            part = repr(b.reference)
            if b.name:
                part = f"{b.name}: {part}"
            parts.append(part)
        return f"Signature({', '.join(parts)})"


class BoundKernelSignature(ABC):
    """Represents a KernelSignature bound to a concrete IR structure."""

    def __init__(self, sig: KernelSignature, entry_block: Block):
        self.sig = sig
        self.entry_block = entry_block
        self._bindings_by_reference: dict[Any, BindingDesc] = {
            b.reference: b for b in sig.bindings
        }

    def resolve_by_reference(self, reference: Any) -> Value:
        binding = self._bindings_by_reference[reference]
        return self.resolve(binding)

    @abstractmethod
    def resolve(self, binding: BindingDesc) -> Value:
        """Resolves a binding to a concrete Value.

        Note that for some implementations, this may involve creating IR. It
        is recommended to cache it.
        """
        ...


class FunctionalKernelSignature(BoundKernelSignature):
    """Simple BoundKernelSignature which maps all bindings to function args.

    Arguments are represented in binding order.
    """

    def __init__(self, sig: KernelSignature, entry_block: Block):
        super().__init__(sig, entry_block)
        block_args = list(entry_block.arguments)
        bindings = sig.bindings
        assert len(block_args) == len(
            bindings
        ), "Mismatched signature vs block arguments"
        self._mapping: dict[Any, Value] = {
            binding.reference: arg_value
            for binding, arg_value in zip(bindings, block_args)
        }

    def resolve(self, binding: BindingDesc) -> Value:
        try:
            return self._mapping[binding.reference]
        except KeyError:
            raise CodegenError(f"Binding {binding.reference} is not bound")

    @staticmethod
    def create(
        sig: KernelSignature, mb: ModuleBuilder, name: str = "kernel"
    ) -> tuple["FunctionalKernelSignature", Operation]:
        """Create a function in the module, returning the bound signature and the function."""
        with InsertionPoint(mb.body_block), Location.unknown():
            input_types = [b.as_mlir_type() for b in sig.bindings]
            ftype = FunctionType.get(input_types, [])
            func_op = func_d.FuncOp(name, ftype)
            arg_locs = [
                (Location.name(b.name) if b.name is not None else Location.unknown())
                for b in sig.bindings
            ]
            entry_block = func_op.add_entry_block(arg_locs)
        return FunctionalKernelSignature(sig, entry_block), func_op.operation
