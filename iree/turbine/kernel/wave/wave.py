# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Any, Callable, Optional
import torch.fx as fx
import inspect

from ..compiler import builder, dispatch_codegen, kernel_codegen, host_codegen
from ..compiler.ir import Context, Operation
from .codegen import WaveEmitter
from .constraints import (
    Constraint,
    HardwareConstraint,
    TilingConstraint,
    WaveConstraint,
    WorkgroupConstraint,
    get_grid_shape,
)
from .codegen import WaveEmitter
from .expansion import expand_graph
from .promotion import promote_placeholders
from .hoisting import hoist_allocs
from .utils import (
    canonicalize_module,
    compile_and_invoke,
    safe_subs,
    remove_chained_getresult,
    subs_idxc,
)
from .minimize_global_loads import minimize_global_loads
from .decompose_reduce_ops import decompose_reduce_ops
from .barriers import add_shared_memory_barriers
from ..lang import Grid, IndexMapping
from ..lang.global_symbols import *
from ..ops import wave_ops
from ..ops.wave_ops import Reduction, CustomOp, get_custom
from .index_sequence_analysis import (
    partition_strided_operators,
    set_node_indices,
    set_post_expansion_indices,
)
from .shared_memory_indexing import (
    apply_shared_memory_indexing_corrections,
    align_index_sizes,
)
from .thread_shape_analysis import determine_thread_shapes
from .scheduling.schedule import schedule_graph
from .._support.indexing import IndexingContext, IndexExpr
from .type_inference import infer_types
import iree.turbine.kernel.lang as tkl
from .._support.tracing import (
    CapturedTrace,
    CompiledContext,
    KernelRegionGraph,
    Launchable,
)

import sympy

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
            self.induction_vars[custom] = tkl.IndexSymbol("$ARG" + str(custom.axis))
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
        for wave_constraint in self.wave_constraints:
            for workgroup_constraint in self.workgroup_constraints:
                # The wave_id is the same as the thread_id, with the exception
                # of wave_id[0] = thread_id[0] / threads_per_wave. This is
                # a convention that we adopt.
                if wave_constraint.dim == workgroup_constraint.dim:
                    wave_constraint.wave_id = (
                        hardware_constraint.get_thread_id_from_workgroup_dim(
                            workgroup_constraint.workgroup_dim
                        )
                    )
                    if workgroup_constraint.workgroup_dim == 0:
                        wave_constraint.wave_id = sympy.floor(
                            wave_constraint.wave_id
                            / hardware_constraint.threads_per_wave
                        )

    def initialize_reductions(self, trace: CapturedTrace) -> None:
        """
        For each reduction, initializes the reduction count by looking at the
        tiling constraints associated with the reduction.

        """
        is_reduction = lambda node: isinstance(get_custom(node), Reduction)
        for reduction in trace.walk(is_reduction):
            for tiling_constraint in self.tiling_constraints:
                if tiling_constraint.dim == get_custom(reduction).axis:
                    reduction.count = subs_idxc(tiling_constraint.count)

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
        self.initialize_reductions(graph)

        idxc = IndexingContext.current()
        idxc.finalize()

        # Initialize Vector shapes
        self.hardware_constraints[0].subs_vector_shapes(idxc.subs)

        # Do type inference.
        infer_types(graph)

        # Promote the placeholders to the appropriate address space.
        promote_placeholders(graph, self.constraints)
        hoist_allocs(graph)

        # Set indices.
        set_node_indices(graph, self.constraints)

        # Analyze Thread Shapes per Op.
        determine_thread_shapes(graph)

        # Expansion
        expand_graph(graph, self.constraints)

        # Set post expansion indices.
        set_post_expansion_indices(graph, self.constraints)

        # Clean up chains of GetResults
        remove_chained_getresult(graph)

        # Optimizations.
        minimize_global_loads(graph, self.constraints)

        # Apply shared memory indexing corrections.
        apply_shared_memory_indexing_corrections(graph, self.constraints)

        # Partition strided operators.
        partition_strided_operators(graph, self.constraints)

        # Align sizes to WG/Tile sizes
        # This pass changes indexing keys, which can interfere with other passes,
        # so it should be called close to the end of pipeline.
        align_index_sizes(graph, self.constraints)

        # Decompose reduce Ops.
        decompose_reduce_ops(graph, self.constraints, idxc.subs)

        # Schedule the reduction ops.
        # Scheduling should always be used with use_scheduling_barriers=True,
        # as this is the only way we can ensure that LLVM enforces our desired schedule.
        # However, due a bug in LLVM, you will need to patch your local LLVM repo
        # with the following PR: https://github.com/kerbowa/llvm-project/commit/ee52732cddae42deed2e3387a83b20ec05860b4e
        # Specifically:
        # git remote add sched_fixes https://github.com/kerbowa/llvm-project.git
        # git fetch sched_fixes
        # git cherry-pick ee52732cddae42deed2e3387a83b20ec05860b4e
        # [Manually resolve conflicts consistent with the PR]
        if kwargs.get("schedule", False):
            use_scheduling_barriers = kwargs.get("use_scheduling_barriers", False)
            schedule_graph(graph, self.constraints, use_scheduling_barriers)

        # Add shared memory barriers.
        add_shared_memory_barriers(graph)

        # Determine grid shape.
        self.grid_type.dims = [1, 1, 1]
        for constraint in self.workgroup_constraints:
            self.grid_type.dims[constraint.workgroup_dim] = safe_subs(
                constraint.count, idxc.subs
            )
        grid = self.grid_type

        root_graph = graph.get_root_graph()
        kernel_sig = kernel_codegen.KernelSignature()
        kernel_sig.add_from_graph_placeholders(root_graph)
        dynamic_symbols = kwargs.get("dynamic_symbols", [])
        kernel_sig.add_from_dynamic_symbols(dynamic_symbols)
        kernel_sig.add_grid(self.grid_type)
        kernel_sig.determine_input_output_buffers(root_graph)

        mb = builder.ModuleBuilder(context=context, module_op=module_op)
        entrypoint_name = self._name
        exe = dispatch_codegen.StreamExecutable(mb, name=entrypoint_name)
        workgroup_size = self.hardware_constraints[0].threads_per_block
        subgroup_size = self.hardware_constraints[0].threads_per_wave
        dispatch_entrypoint = exe.define_entrypoint(
            entrypoint_name,
            kernel_sig,
            grid,
            workgroup_size,
            subgroup_size,
            dynamic_symbols,
        )

        emitter = WaveEmitter(
            dispatch_entrypoint, graph, self.constraints, dynamic_symbols
        )
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

        run = kwargs.get("run", False)
        run_bench = kwargs.get("run_bench", False)
        if run or run_bench:
            # TODO: cache compiled code
            dynamic_symbols = kwargs.get("dynamic_symbols", [])
            host_codegen.isolated_test_call(
                mb, exe, kernel_sig, entrypoint_name, dynamic_symbols
            )
            asm = mb.module_op.get_asm()

            kernel_inputs = []
            kernel_outputs = []
            for arg, b in zip(args, kernel_sig.kernel_buffer_bindings):
                usage = b.kernel_buffer_type.usage
                if usage == kernel_codegen.KernelBufferUsage.INPUT:
                    kernel_inputs.append(arg)

                if usage == kernel_codegen.KernelBufferUsage.OUTPUT:
                    kernel_outputs.append(arg)

            dynamic_symbols_map = kwargs.get("dynamic_symbols_map", {})
            if dynamic_symbols:
                kernel_inputs += [dynamic_symbols_map[sym] for sym in dynamic_symbols]

            config = kwargs.get("run_config", None)
            if not config:
                raise ValueError("no config provided")

            compile_and_invoke(
                asm,
                "isolated_benchmark",
                config,
                kernel_inputs,
                kernel_outputs,
                run,
                run_bench,
            )

        return mb

    def aot_execute(self, args, kwargs):
        raise NotImplementedError("AOT execution for wave not implemented yet.")

    def eager_execute(self, args, kwargs):
        raise NotImplementedError("Eager execution for wave not implemented yet.")

    def __repr__(self):
        return f"tk.wave @{self._name}[{self.grid_type}]"
