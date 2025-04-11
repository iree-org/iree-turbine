"""Code generation support for top-level IREE dispatch constructs.

This assumes that you have some form of code generation for the
"inside" of some kernels, as this layer is responsible for
embedding and generating the calls/dispatches.
"""

from typing import Any, Callable, Optional

from .._support.indexing import IndexSymbol, IndexExpr

from .base import (
    ValidationError,
)

from .builder import (
    ModuleBuilder,
)

from .ir import (
    Block,
    DictAttr,
    FunctionType,
    IndexType,
    InsertionPoint,
    IntegerAttr,
    IrType,
    Location,
    StringAttr,
    Value,
    arith_d,
    func_d,
    stream_d,
    iree_codegen_d,
)

from .kernel_codegen import (
    BindingDesc,
    BindingType,
    BoundKernelSignature,
    KernelSignature,
)

from ..lang.grid import Grid


class StreamExecutable:
    """Encapsulates a 'stream' compilable executable which can be dispatched to.

    This corresponds to a `stream.executable`, consisting of one or more exported
    dispatch functions.
    """

    __slots__ = [
        "_mb",
        "_exe_op",
        "_exe_block",
        "_loc",
        "sym_name",
        "def_module",
    ]

    def __init__(
        self,
        mb: ModuleBuilder,
        *,
        loc: Optional[Location] = None,
        name: str = "__executable",
    ):
        self._mb = mb
        if not loc:
            loc = mb.unknown_loc
        self._loc = loc

        # Construct the executable.
        with loc:
            with InsertionPoint(mb.body_block):
                self._exe_op = exe_op = stream_d.ExecutableOp(
                    name, sym_visibility="private"
                )
                exe_block = exe_op.body.blocks.append()
                self._exe_block: Block = exe_block
                stream_d.ExecutableEndOp(ip=InsertionPoint(exe_block))
            mb.symbol_table.insert(exe_op)
            self.sym_name: StringAttr = exe_op.sym_name

            # Construct the inner definitions module.
            with InsertionPoint.at_block_begin(exe_block):
                self.def_module = ModuleBuilder(context=mb.context)

    def define_entrypoint(
        self,
        name: str,
        sig: KernelSignature,
        grid: Grid,
        workgroup_size: list[int] = None,
        subgroup_size: int = None,
        dynamic_symbols: list[IndexSymbol] = [],
        llvm_configs: dict[str, str] = {},
    ) -> "DispatchEntrypoint":
        """Defines a dispatch function with a signature like:

        ```
        func.func @name(%in0 : !stream.binding, %in1 : !stream.binding,
                        %workload0 : index, %workload1 : index,
                        %result0 : !stream.binding, %result1 : !stream.binding)
        ```

        Also adds an export with workgroup function like:

        ```
        stream.executable.export private @name(%workload0 : index, %workload1 : index) -> (index, [[grid_arity...]]) {

        }
        ```

        The given name is not uniqued (must be unique as given by the caller).
        """
        kb_input_bindings = sig.kernel_buffer_input_bindings
        kb_output_bindings = sig.kernel_buffer_output_bindings
        dynamic_dim_bindings = sig.dynamic_dim_bindings
        scalar_bindings = sig.scalar_bindings

        # Input bindings are always user specified.
        # Output bindings are the real outputs.
        # Dynamic dim bindings are the dynamic dims of the input and output tensors.
        linear_bindings = (
            kb_input_bindings
            + kb_output_bindings
            + scalar_bindings
            + dynamic_dim_bindings
        )

        dynamic_dim_indices = {
            "begin": len(kb_input_bindings)
            + len(kb_output_bindings)
            + len(scalar_bindings),
            "end": len(linear_bindings),
        }

        with self._loc:
            binding_type = IrType.parse("!stream.binding")
            index_type = IndexType.get()

            # Define the dispatch function.
            def abi_type(binding: BindingDesc):
                if binding.binding_type == BindingType.KERNEL_BUFFER:
                    return binding_type
                return binding.as_mlir_type()

            def_ftype = FunctionType.get(
                [abi_type(b) for b in linear_bindings],
                [],
            )
            with InsertionPoint(self.def_module.body_block):
                def_func_op = func_d.FuncOp(name, def_ftype)
                def_func_block = def_func_op.add_entry_block()
                def_func_args = list(def_func_block.arguments)
                if workgroup_size is not None and subgroup_size is not None:
                    pipeline_attr = iree_codegen_d.DispatchLoweringPassPipelineAttr.get(
                        iree_codegen_d.DispatchLoweringPassPipeline.None_
                    )
                    translation_config = None
                    if llvm_configs:
                        # Add llvm_func_attrs to translation config if any is specified.
                        llvm_func_attrs = DictAttr.get(
                            {k: StringAttr.get(str(v)) for k, v in llvm_configs.items()}
                        )
                        translation_config = DictAttr.get(
                            {"llvm_func_attrs": llvm_func_attrs}
                        )
                    def_func_op.attributes[
                        "translation_info"
                    ] = iree_codegen_d.TranslationInfoAttr.get(
                        pipeline_attr,
                        None,
                        workgroup_size,
                        subgroup_size,
                        configuration=translation_config,
                    )

            # Define the export.
            with InsertionPoint.at_block_begin(self._exe_block):
                export_op = stream_d.ExecutableExportOp(name, name)
                export_block = export_op.workgroup_count.blocks.append(
                    *([b.as_mlir_type() for b in dynamic_dim_bindings])
                )

            workgroup_builder = WorkgroupBuilder(
                export_block, lambda vs: stream_d.ReturnOp(vs)
            )

            # TODO: Support passing workload to the dispatch function.
            from ..wave.codegen import gen_sympy_index

            # Map dynamic symbols to block arguments.
            dynamic_symbols_mapping = {
                k: v
                for k, v in zip(
                    dynamic_symbols, workgroup_builder.entry_block.arguments
                )
            }

            with InsertionPoint(workgroup_builder.entry_block):
                result_type = IndexType.get()
                workgroup_values = []
                for dim in grid.dims:
                    if isinstance(dim, IndexExpr):
                        workgroup_values.append(
                            gen_sympy_index(dynamic_symbols_mapping, dim)
                        )
                    else:
                        workgroup_values.append(
                            arith_d.constant(
                                result_type, IntegerAttr.get(result_type, dim)
                            )
                        )

                while len(workgroup_values) < 3:
                    workgroup_values.append(
                        arith_d.constant(result_type, IntegerAttr.get(result_type, 1))
                    )
            workgroup_builder.terminate(workgroup_values)

        # Map dynamic symbols to func arguments for dispatch entrypoint.
        dynamic_symbols_mapping = {
            k: v
            for k, v in zip(
                dynamic_symbols,
                def_func_args[
                    dynamic_dim_indices["begin"] : dynamic_dim_indices["end"]
                ],
            )
        }

        return DispatchEntrypoint(
            sig, def_func_block, linear_bindings, dynamic_symbols_mapping
        )


class WorkgroupBuilder:
    """Builder for a workgroup calculation block."""

    __slots__ = [
        "entry_block",
        "workload",
        "_term_ctor",
    ]

    def __init__(self, entry_block: Block, term_ctor: Callable[[list[Value]], None]):
        self.entry_block = entry_block
        self.workload = list(entry_block.arguments)
        self._term_ctor = term_ctor

    @property
    def location(self) -> Location:
        return self.entry_block.owner.location

    def terminate(self, returns: list[Value]):
        entry_block = self.entry_block
        with entry_block.owner.location, InsertionPoint(entry_block):
            self._term_ctor(returns)


class DispatchEntrypoint(BoundKernelSignature):
    def __init__(
        self,
        sig: KernelSignature,
        entry_block: Block,
        linear_bindings: list[BindingDesc],
        dynamic_symbols_mapping: dict[IndexSymbol, Value],
    ):
        super().__init__(sig, entry_block)
        self.dynamic_symbols_mapping = dynamic_symbols_mapping
        self._abi_value_by_reference: dict[tuple[str, Any], Value] = {
            b.reference: value
            for value, b in zip(entry_block.arguments, linear_bindings)
        }

    def get_dynamic_dims(self, binding: BindingDesc) -> list[Value]:
        """
        This function determines the dynamic dimensions of the binding.
        If the binding has a physical layout, we check whether the physical layout
        is completely static. In this case there are no dynamic dimensions.
        If the physical layout does have dynamic dimensions, then we return
        the dynamic dimensions based on the symbolic shape of the binding.
        """
        dynamic_dims = [
            self.dynamic_symbols_mapping[dim]
            for dim in binding.kernel_buffer_type.symbolic_shape
            if dim in self.dynamic_symbols_mapping
        ]
        node_type = binding.reference[1].type
        if node_type.physical_layout:
            physical_shape = node_type.physical_layout.shape
            if all(physical_shape):
                return []
            assert len(dynamic_dims) == physical_shape.count(
                None
            ), f"Expected {physical_shape.count(None)} dynamic dims, got {len(dynamic_dims)}"
        return dynamic_dims

    def resolve(self, binding: BindingDesc) -> Value:
        ref_type, ref_value = binding.reference
        if ref_type == "grid":
            return stream_d.dispatch_workgroup_id(
                IntegerAttr.get(IndexType.get(), ref_value)
            )

        if binding.binding_type == BindingType.KERNEL_BUFFER:
            # Issue a subspan to get into the memref domain.
            result_type = IndexType.get()
            zero_value = arith_d.constant(result_type, IntegerAttr.get(result_type, 0))
            linear_arg_value = self._abi_value_by_reference[binding.reference]
            return stream_d.binding_subspan(
                binding.as_mlir_type(),
                linear_arg_value,
                byte_offset=zero_value,
                dynamic_dims=self.get_dynamic_dims(binding),
            )

        if binding.binding_type == BindingType.SCALAR_VALUE:
            linear_arg_value = self._abi_value_by_reference[binding.reference]
            return linear_arg_value

        raise ValidationError(f"Unhandled binding type: {binding}")
