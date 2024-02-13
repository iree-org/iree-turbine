from .kernel_codegen import KernelSignature
from .dispatch_codegen import StreamExecutable

from .builder import (
    ModuleBuilder,
)

from .ir import (
    Block,
    FunctionType,
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


def memref_to_tensor(memrefs: list[IrType]):
    tensors = []
    for m in memrefs:
        assert isinstance(m, MemRefType)
        t = RankedTensorType.get(m.shape, m.element_type)
        tensors.append(t)
    return tensors


def isolated_test_call(
    mb: ModuleBuilder, exe: StreamExecutable, sig: KernelSignature, entrypoint: str
):
    with InsertionPoint(mb.body_block), Location.unknown():
        input_types = [b.as_mlir_type() for b in sig.kernel_buffer_input_bindings]
        input_tensors = memref_to_tensor(input_types)
        output_types = [b.as_mlir_type() for b in sig.kernel_buffer_output_bindings]
        output_tensors = memref_to_tensor(output_types)

        ftype = FunctionType.get(input_tensors, output_tensors)
        func_op = func_d.FuncOp("isolated_benchmark", ftype)
        arg_locs = [
            (Location.name(b.name) if b.name is not None else Location.unknown())
            for b in sig.kernel_buffer_input_bindings
        ]
        entry_block = func_op.add_entry_block(arg_locs)
        with InsertionPoint(entry_block):
            assert isinstance(entry_block, Block)
            # Create a flow.dispatch op to the kernel
            dispatch = SymbolRefAttr.get([exe.sym_name.value, entrypoint])
            entrypoints = ArrayAttr.get([dispatch])

            out = flow_d.DispatchOp(
                output_tensors, [], entrypoints, entry_block.arguments, [], []
            )

            func_d.ReturnOp(out)
