from .kernel_codegen import KernelSignature
from .dispatch_codegen import StreamExecutable

from .builder import (
    ModuleBuilder,
)

from .ir import (
    ArrayAttr,
    Block,
    F32Type,
    F64Type,
    FunctionType,
    IndexType,
    InsertionPoint,
    IntegerAttr,
    IntegerType,
    IrType,
    Location,
    MemRefType,
    RankedTensorType,
    SymbolRefAttr,
    Value,
    arith_d,
    flow_d,
    func_d,
    hal_d,
    tensor_d,
)

from .._support.indexing import IndexSymbol
from .._support.location import capture_location
from .._support.location_config import LocationCaptureConfig
from .kernel_codegen import BindingDesc, KernelBufferUsage

from typing import Optional


def memref_to_tensor(memrefs: list[IrType], use_views: bool = False):
    if use_views:
        view_type = IrType.parse("!hal.buffer_view")

    tensors = []
    for m in memrefs:
        # append scalars as-it-is to tensors list
        if isinstance(m, (F32Type, F64Type, IndexType)) or (
            isinstance(m, IntegerType) and m.is_signless
        ):
            tensors.append(m)
            continue
        assert isinstance(m, MemRefType)
        t = view_type if use_views else RankedTensorType.get(m.shape, m.element_type)
        tensors.append(t)
    return tensors


def get_dynamic_dims(bindings: list[BindingDesc], dynamic_symbols: list[IndexSymbol]):
    dynamic_dims: list[IndexSymbol] = []
    for b in bindings:
        node_type = b.reference[1].type
        if node_type.physical_layout:
            if all(node_type.physical_layout.shape):
                continue
        for dim in b.kernel_buffer_type.symbolic_shape:
            if dim in dynamic_symbols:
                dynamic_dims.append(dim)
    return dynamic_dims


def to_index(v: Value) -> Value:
    t = v.type
    if isinstance(t, IndexType):
        return v

    if isinstance(t, IntegerType):
        return arith_d.index_cast(IndexType.get(), v)

    assert False, f"Expected IndexType or IntegerType, got {t}"


def isolated_test_call(
    mb: ModuleBuilder,
    exe: StreamExecutable,
    sig: KernelSignature,
    entrypoint: str,
    func_name: str = "isolated_benchmark",
    dynamic_symbols: list[IndexSymbol] = [],
    *,
    location_capture_config: Optional[LocationCaptureConfig] = None,
    async_dispatch: bool = False,
):
    with InsertionPoint(mb.body_block), Location.unknown():
        input_types = [b.as_mlir_type() for b in sig.kernel_buffer_bindings] + [
            b.as_mlir_type() for b in sig.scalar_bindings
        ]

        input_tensors = memref_to_tensor(input_types, use_views=async_dispatch)
        argument_dims = get_dynamic_dims(sig.kernel_buffer_bindings, dynamic_symbols)
        # Adding unique dynamic dims as inputs.
        input_tensors += [IndexType.get() for _ in list(dict.fromkeys(argument_dims))]
        # Add additional dynamic symbols as inputs.
        input_tensors += [
            IndexType.get() for _ in set(dynamic_symbols).difference(argument_dims)
        ]

        if async_dispatch:
            fence_type = IrType.parse("!hal.fence")
            input_tensors += [fence_type] * 2
            func_name = func_name + "$async"

        output_types = [b.as_mlir_type() for b in sig.kernel_buffer_output_bindings]
        output_tensors = memref_to_tensor(output_types, use_views=async_dispatch)
        result_dims = get_dynamic_dims(
            sig.kernel_buffer_output_bindings, dynamic_symbols
        )

        ftype = FunctionType.get(input_tensors, output_tensors)
        func_op = func_d.FuncOp(func_name, ftype)
        captured_loc = capture_location(location_capture_config)
        actual_loc = captured_loc.to_mlir() if captured_loc else Location.unknown()
        scalar_bindings = sig.scalar_bindings
        arg_locs = [
            (Location.name(b.name, actual_loc) if b.name is not None else actual_loc)
            for b in sig.kernel_buffer_bindings
            + scalar_bindings
            + sig.dynamic_dim_bindings
        ]
        if async_dispatch:
            arg_locs += [Location.unknown()] * 2

        entry_block = func_op.add_entry_block(arg_locs)
        scalars_offset = len(sig.kernel_buffer_bindings)
        scalars_count = len(scalar_bindings)
        dynamic_offset = scalars_offset + scalars_count

        with InsertionPoint(entry_block):
            arguments = entry_block.arguments
            if async_dispatch:
                in_fence = arguments[-2]
                out_fence = arguments[-1]
                arguments = list(arguments[:-2])

                for i, b in enumerate(sig.kernel_buffer_bindings):
                    shape = b.kernel_buffer_type.symbolic_shape

                    arg = arguments[i]
                    arg_type = memref_to_tensor([b.as_mlir_type()])[0]
                    target_dims = [
                        hal_d.buffer_view_dim(arg, d)
                        for d in range(len(shape))
                        if arg_type.is_dynamic_dim(d)
                    ]
                    arguments[i] = hal_d.tensor_import(
                        arg_type,
                        arg,
                        wait_fence=in_fence,
                        target_encoding=arg_type,
                        target_dims=target_dims,
                    )

            scalars_args = [
                to_index(v)
                for v, b in zip(
                    arguments[scalars_offset:dynamic_offset], scalar_bindings
                )
                if b.symbol_type is not None
            ]
            dynamic_args = [to_index(v) for v in arguments[dynamic_offset:]]
            dynamic_argument_map = {k: v for k, v in zip(dynamic_symbols, dynamic_args)}

            assert isinstance(entry_block, Block)
            # Create a flow.dispatch op to the kernel
            dispatch = SymbolRefAttr.get([exe.sym_name.value, entrypoint])
            entrypoints = ArrayAttr.get([dispatch])

            buffer_binding_count = len(sig.kernel_buffer_bindings)
            input_binding_count = len(sig.kernel_buffer_input_bindings)
            tied_operands = ArrayAttr.get(
                [
                    IntegerAttr.get(IndexType.get(), out_idx)
                    for out_idx in range(input_binding_count, buffer_binding_count)
                ]
            )

            out = flow_d.DispatchOp(
                memref_to_tensor(output_types),  # output_tensors,
                [dynamic_argument_map[dim] for dim in dynamic_symbols] + scalars_args,
                entrypoints,
                arguments,
                [dynamic_argument_map[dim] for dim in argument_dims],
                [dynamic_argument_map[dim] for dim in result_dims],
                tied_operands=tied_operands,
            )

            if async_dispatch:
                out = list(out.results)
                out_types = memref_to_tensor(
                    [b.as_mlir_type() for b in sig.kernel_buffer_output_bindings]
                )
                barrier = hal_d.tensor_barrier(out_types, out, signal_fence=out_fence)
                if len(out_types) == 1:
                    barrier = [barrier]

                view_type = IrType.parse("!hal.buffer_view")
                for i, b in enumerate(sig.kernel_buffer_output_bindings):
                    shape = b.kernel_buffer_type.symbolic_shape

                    out_type = out_types[i]
                    source_dims = [
                        tensor_d.dim(out[i], arith_d.constant(IndexType.get(), d))
                        for d in range(len(shape))
                        if out_type.is_dynamic_dim(d)
                    ]
                    out[i] = hal_d.tensor_export(
                        view_type, barrier[i], out_type, source_dims=source_dims
                    )

            func_d.ReturnOp(out)
