from typing import Any, Callable, Optional
import torch.fx as fx
import inspect

from ..compiler import builder, dispatch_codegen, kernel_codegen, host_codegen
from ..compiler.ir import Context, Operation
from .codegen import WaveEmitter
from .constraints import (
    Constraint,
    TilingConstraint,
    WorkgroupConstraint,
    get_grid_shape,
    WaveConstraint,
    HardwareConstraint,
)
from .codegen import WaveEmitter
from .expansion import expand_graph
from .promotion import promote_placeholders
from .hoisting import hoist_allocs
from .utils import canonicalize_module
from ..lang import Grid, IndexMapping
from ..lang.global_symbols import *
from ..ops import wave_ops
from ..ops.wave_ops import Reduction, CustomOp, get_custom
from .register_analysis import determine_register_shape
from .._support.indexing import IndexingContext, IndexExpr
import shark_turbine.kernel.lang as tkl
from .._support.tracing import (
    CapturedTrace,
    CompiledContext,
    KernelRegionGraph,
    Launchable,
)
from iree.compiler import compile_str
import iree.runtime as rt

__all__ = ["wave", "wave_trace_only"]


def wave(constraints: Optional[list[Constraint]] = None):
    def decorator(f: Callable[..., Any]) -> "LaunchableWave":
        return LaunchableWave(constraints, f.__name__, f)

    return decorator


def wave_trace_only(constraints: Optional[list[Constraint]] = None):
    def decorator(f: Callable[..., Any]) -> "Callable[[], CapturedTrace]":
        wave = LaunchableWave(constraints, f.__name__, f)
        return wave._trace  # type: ignore

    return decorator


def _invoke(vm_context, device, entry_function, inputs, outputs):
    arg_list = rt.VmVariantList(len(inputs))
    ret_list = rt.VmVariantList(len(outputs))

    for input in inputs:
        input_cpu = input.cpu().contiguous()
        device_array = rt.asdevicearray(device, input_cpu)
        arg_list.push_ref(device_array._buffer_view)

    vm_context.invoke(entry_function, arg_list, ret_list)

    for i, ret in enumerate(outputs):
        device_buffer_view = rt.HalBufferView.__iree_vm_cast__(ret_list.get_as_ref(i))
        device_array = rt.DeviceArray(device, device_buffer_view)

        # TODO: Make to_host accept out array/buffer, so we can avoid extra data copy.
        host_array = device_array.to_host()

        # Convert to torch tensor without actually importing torch.
        ret[:] = type(ret)(host_array)


class LaunchableWave(Launchable):
    def __init__(
        self,
        constraints: Optional[list[Constraint]],
        name: str,
        eager_function: Callable[[Any], Any],
    ):
        super().__init__(eager_function)

        self.constraints = constraints if constraints else []
        self.induction_vars: dict[CustomOp, IndexExpr] = {}
        self._name = name
        self._f = eager_function
        self._sig = inspect.signature(eager_function)

        self.grid_type = Grid[tuple(get_grid_shape(self.workgroup_constraints))]

    @property
    def workgroup_constraints(self) -> list[WorkgroupConstraint]:
        return [
            constraint
            for constraint in self.constraints
            if isinstance(constraint, WorkgroupConstraint)
        ]

    @property
    def tiling_constraints(self) -> list[TilingConstraint]:
        return [
            constraint
            for constraint in self.constraints
            if isinstance(constraint, TilingConstraint)
        ]

    @property
    def wave_constraints(self) -> list[WaveConstraint]:
        return [
            constraint
            for constraint in self.constraints
            if isinstance(constraint, WaveConstraint)
        ]

    @property
    def hardware_constraints(self) -> list[HardwareConstraint]:
        return [
            constraint
            for constraint in self.constraints
            if isinstance(constraint, HardwareConstraint)
        ]

    def _trace(self) -> CapturedTrace:
        region_graph = KernelRegionGraph()
        with CompiledContext(region_graph, grid_type=self.grid_type) as context:
            # Get all explictly defined custom ops
            custom_ops: dict[str, wave_ops.CustomOp] = {
                cls.tkw_op_name: cls
                for _, cls in inspect.getmembers(wave_ops, inspect.isclass)
                if issubclass(cls, wave_ops.CustomOp) and hasattr(cls, "tkw_op_name")
            }

            # Register custom ops
            for name, op in custom_ops.items():
                context.register_custom_op(name, op)

            with region_graph.subtracer() as subtracer:
                root_name, _ = subtracer.trace(self._f)
                trace = CapturedTrace(region_graph, root_name)

        return trace

    def create_induction_vars(self, trace: CapturedTrace) -> None:
        """
        Creates induction variables for all the reductions in the graph
        and associates tiling constraints all the reduction dimensions
        with the appropriate induction variables.

        """

        def is_reduction(node: fx.Node):
            custom = get_custom(node)
            return isinstance(custom, Reduction)

        reduction_nodes = trace.walk(is_reduction)
        for node in reduction_nodes:
            custom = get_custom(node)
            self.induction_vars[custom] = tkl.IndexSymbol("$ARG" + custom.axis.name)
            for tiling_constraint in self.tiling_constraints:
                if tiling_constraint.dim == custom.axis:
                    tiling_constraint.induction_var = self.induction_vars[custom]

    def initialize_wave_constraints(self, trace: CapturedTrace) -> None:
        """
        For each wave constraint, determines the appropriate wave id by looking
        for workgroup constraints along the same dimension and using information
        from the hardware constraints.

        """

        hardware_constraint = self.hardware_constraints[0]
        thread_ids = [THREAD_0, THREAD_1, THREAD_2]
        for wave_constraint in self.wave_constraints:
            for workgroup_constraint in self.workgroup_constraints:
                if wave_constraint.dim == workgroup_constraint.dim:
                    wave_constraint.wave_id = thread_ids[
                        workgroup_constraint.workgroup_dim
                    ]
                    if workgroup_constraint.workgroup_dim == 0:
                        wave_constraint.wave_id /= hardware_constraint.threads_per_wave

    def _trace_and_get_kernel_signature(
        self,
        args,
        kwargs,
        context: Optional[Context] = None,
        module_op: Optional[Operation] = None,
    ) -> CapturedTrace:
        # Trace the function.
        graph = self._trace()

        self.create_induction_vars(graph)
        self.initialize_wave_constraints(graph)

        idxc = IndexingContext.current()
        idxc.finalize()

        # Promote the placeholders to the appropriate address space.
        promote_placeholders(graph, self.constraints)
        hoist_allocs(graph)

        # Expansion
        expand_graph(graph, self.constraints)

        # Register analysis to determine register shapes.
        determine_register_shape(graph)

        self.grid_type.dims = [1, 1, 1]
        for constraint in self.workgroup_constraints:
            self.grid_type.dims[constraint.workgroup_dim] = (
                constraint.dim // constraint.tile_size
            ).subs(idxc.subs)
        grid = self.grid_type

        root_graph = graph.get_root_graph()
        kernel_sig = kernel_codegen.KernelSignature()
        kernel_sig.add_from_graph_placeholders(root_graph)
        kernel_sig.add_grid(self.grid_type)
        kernel_sig.determine_input_output_buffers(root_graph)

        mb = builder.ModuleBuilder(context=context, module_op=module_op)
        entrypoint_name = self._name
        exe = dispatch_codegen.StreamExecutable(mb, name=entrypoint_name)
        dispatch_entrypoint = exe.define_entrypoint(entrypoint_name, kernel_sig, grid)

        emitter = WaveEmitter(dispatch_entrypoint, graph, self.constraints)
        emitter.emit(graph.get_root_graph())
        emitter.finish()

        if kwargs.get("canonicalize", False):
            canonicalize_module(mb.module_op)

        return mb, graph, exe, kernel_sig, entrypoint_name

    def test_execute(self, args, kwargs):
        (
            mb,
            graph,
            exe,
            kernel_sig,
            entrypoint_name,
        ) = self._trace_and_get_kernel_signature(args, kwargs)

        if kwargs.get("run", False):
            # TODO: cache compiled code
            host_codegen.isolated_test_call(mb, exe, kernel_sig, entrypoint_name)
            asm = mb.module_op.get_asm()

            kernel_inputs = []
            kernel_outputs = []
            for arg, b in zip(args, kernel_sig.kernel_buffer_bindings):
                usage = b.kernel_buffer_type.usage
                if usage == kernel_codegen.KernelBufferUsage.INPUT:
                    kernel_inputs.append(arg)

                if usage == kernel_codegen.KernelBufferUsage.OUTPUT:
                    kernel_outputs.append(arg)

            # TODO: Have some default config.
            config = kwargs.get("run_config", None)
            if not config:
                raise ValueError("no config provided")

            backend = config["backend"]
            device = config["device"]
            flags = [
                f"--iree-hal-target-backends={backend}",
                "--iree-vm-bytecode-module-strip-source-map=true",
                "--iree-opt-strip-assertions=true",
                "--iree-vm-target-truncate-unsupported-floats",
            ]

            # TODO: More targets/backends support.
            if backend == "rocm":
                target = config["target"]
                flags.append(f"--iree-rocm-target-chip={target}")

            if config.get("print_ir_after_all", False):
                flags.append("--mlir-print-ir-after-all")

            res = compile_str(asm, target_backends=[backend], extra_args=flags)

            rt_config = rt.Config(device)
            device = rt_config.device
            vm_instance = rt_config.vm_instance
            mod = rt.VmModule.copy_buffer(vm_instance, res)

            vm_modules = [
                mod,
                rt.create_hal_module(vm_instance, device),
            ]
            ctx = rt.SystemContext(
                vm_modules=vm_modules,
                config=rt_config,
            )
            vm_context = ctx.vm_context

            func = mod.lookup_function("isolated_benchmark")
            _invoke(vm_context, device, func, kernel_inputs, kernel_outputs)

        return mb

    def aot_execute(self, args, kwargs):
        raise NotImplementedError("AOT execution for wave not implemented yet.")

    def eager_execute(self, args, kwargs):
        raise NotImplementedError("Eager execution for wave not implemented yet.")

    def __repr__(self):
        return f"tk.wave @{self._name}[{self.grid_type}]"
