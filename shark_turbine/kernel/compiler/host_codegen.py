from .kernel_codegen import KernelSignature
from .dispatch_codegen import StreamExecutable

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
    ArrayAttr,
    SymbolRefAttr,
    MemRefType,
    RankedTensorType,
    flow_d,
    func_d,
)

from .._support.indexing import IndexSymbol
from .kernel_codegen import BindingDesc


def memref_to_tensor(memrefs: list[IrType]):
    tensors = []
    for m in memrefs:
        assert isinstance(m, MemRefType)
        t = RankedTensorType.get(m.shape, m.element_type)
        tensors.append(t)
    return tensors


def get_dynamic_dims(bindings: list[BindingDesc], dynamic_symbols: list[IndexSymbol]):
    dynamic_dims: list[IndexSymbol] = []
    for b in bindings:
        for dim in b.kernel_buffer_type.symbolic_shape:
            if dim in dynamic_symbols:
                dynamic_dims.append(dim)
    return dynamic_dims


def isolated_test_call(
    mb: ModuleBuilder,
    exe: StreamExecutable,
    sig: KernelSignature,
    entrypoint: str,
    dynamic_symbols: list[IndexSymbol] = [],
):
    with InsertionPoint(mb.body_block), Location.unknown():
        input_types = [b.as_mlir_type() for b in sig.kernel_buffer_input_bindings]
        input_tensors = memref_to_tensor(input_types)
        argument_dims = get_dynamic_dims(
            sig.kernel_buffer_input_bindings, dynamic_symbols
        )
        input_tensors += [IndexType.get() for _ in argument_dims]

        output_types = [b.as_mlir_type() for b in sig.kernel_buffer_output_bindings]
        output_tensors = memref_to_tensor(output_types)
        result_dims = get_dynamic_dims(
            sig.kernel_buffer_output_bindings, dynamic_symbols
        )

        ftype = FunctionType.get(input_tensors, output_tensors)
        func_op = func_d.FuncOp("isolated_benchmark", ftype)
        arg_locs = [
            (Location.name(b.name) if b.name is not None else Location.unknown())
            for b in sig.kernel_buffer_input_bindings + sig.dynamic_dim_bindings
        ]
        entry_block = func_op.add_entry_block(arg_locs)
        offset = len(sig.kernel_buffer_input_bindings)
        dynamic_argument_map = {
            k: v for k, v in zip(dynamic_symbols, entry_block.arguments[offset:])
        }
        with InsertionPoint(entry_block):
            assert isinstance(entry_block, Block)
            # Create a flow.dispatch op to the kernel
            dispatch = SymbolRefAttr.get([exe.sym_name.value, entrypoint])
            entrypoints = ArrayAttr.get([dispatch])

            out = flow_d.DispatchOp(
                output_tensors,
                [dynamic_argument_map[dim] for dim in dynamic_symbols],
                entrypoints,
                entry_block.arguments,
                [dynamic_argument_map[dim] for dim in argument_dims],
                [dynamic_argument_map[dim] for dim in result_dims],
            )

            func_d.ReturnOp(out)
