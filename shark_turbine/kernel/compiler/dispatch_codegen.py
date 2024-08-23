"""Code generation support for top-level IREE dispatch constructs.

This assumes that you have some form of code generation for the
"inside" of some kernels, as this layer is responsible for
embedding and generating the calls/dispatches.
"""

from typing import Any, Callable, Optional, Type

from .._support.indexing import (
    IndexingContext,
)

from .base import (
    CodegenError,
    ValidationError,
)

from .builder import (
    ModuleBuilder,
)

from .ir import (
    Attribute,
    Block,
    FunctionType,
    IndexType,
    InsertionPoint,
    IntegerAttr,
    IrType,
    Location,
    Operation,
    StringAttr,
    Value,
    arith_d,
    flow_d,
    func_d,
    stream_d,
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
        kb_temp_bindings = sig.kernel_buffer_temporary_bindings
        kb_output_bindings = sig.kernel_buffer_output_bindings
        # TODO: The way we are doing grid bindings is wrong. The Grid type
        # should be paramerized with special grid axis symbols which are
        # algebraically related to concrete shape dim symbols. For now, we are
        # just assuming that the grid dims can be resolved to constants , when
        # in reality, we should pass the workload and parameterize the grid
        # dims on the workloads.
        workload_axis_bindings = []

        # Input bindings are always user specified.
        # Grid/workgroup bindings are in the inputs section but are implied.
        # Temp bindings are a special kind of output bindings.
        # Output bindings are the real outputs.
        linear_bindings = (
            kb_input_bindings
            + workload_axis_bindings
            + kb_temp_bindings
            + kb_output_bindings
        )

        # TODO: This is sloppy. This assert will hit on some user errors for
        # unsupported type combinations and is just a last resort right now.
        # TODO: This is currently disabled because the grid_bindings don't match
        # workload bindings.
        # assert len(linear_bindings) == len(
        #     sig.bindings
        # ), f"Not all bindings converted: {linear_bindings} vs {sig.bindings}"

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
                    def_func_op.attributes["translation_info"] = Attribute.parse(
                        f"#iree_codegen.translation_info<None "
                        f"workgroup_size=[{','.join(str(x) for x in workgroup_size)}]"
                        f"subgroup_size={subgroup_size}>"
                    )

            # Define the export.
            with InsertionPoint.at_block_begin(self._exe_block):
                export_op = stream_d.ExecutableExportOp(name, name)
                export_block = export_op.workgroup_count.blocks.append(
                    *([b.as_mlir_type() for b in workload_axis_bindings])
                )

            workgroup_builder = WorkgroupBuilder(
                export_block, lambda vs: stream_d.ReturnOp(vs)
            )

            # TODO: Support passing workload to the dispatch function.
            with InsertionPoint(workgroup_builder.entry_block):
                result_type = IndexType.get()
                workgroup_values = [
                    arith_d.constant(result_type, IntegerAttr.get(result_type, dim))
                    for dim in grid.dims
                ]

                while len(workgroup_values) < 3:
                    workgroup_values.append(
                        arith_d.constant(result_type, IntegerAttr.get(result_type, 1))
                    )
            workgroup_builder.terminate(workgroup_values)

        return DispatchEntrypoint(sig, def_func_block, linear_bindings)


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
    ):
        super().__init__(sig, entry_block)
        self._abi_value_by_reference: dict[tuple[str, Any], Value] = {
            b.reference: value
            for value, b in zip(entry_block.arguments, linear_bindings)
        }

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
            # TODO: Need to also look up dynamic symbol values.
            return stream_d.binding_subspan(
                binding.as_mlir_type(),
                linear_arg_value,
                byte_offset=zero_value,
                dynamic_dims=[],
            )

        raise ValidationError(f"Unhandled binding type: {binding}")
