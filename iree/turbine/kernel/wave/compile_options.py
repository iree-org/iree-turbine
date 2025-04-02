from dataclasses import dataclass, field
from typing import Any
from .scheduling.schedule_enums import SchedulingType
from .._support.indexing import IndexExpr
from .utils.classes import KernelLaunchInfo
from ..compiler.kernel_codegen import KernelBufferUsage


@dataclass
class WaveCompileOptions:
    """
    Options for compiling the wave kernel.
    """

    # === General options ===
    canonicalize: bool = False
    func_name: str = "isolated_benchmark"

    # === Symbol mappings ===
    subs: dict[str, Any] = field(default_factory=list)
    dynamic_symbols_map: dict[str, IndexExpr] = field(default_factory=dict)
    dynamic_symbols: list[str] = field(default_factory=list)

    # === Scheduling options ===
    schedule: bool = SchedulingType.NONE
    use_scheduling_barriers: bool = False

    # === Runtime options ===
    kernel_launch_info: KernelLaunchInfo = field(default_factory=KernelLaunchInfo)
    kernel_usages: tuple[KernelBufferUsage] = None
    inplace: bool = True

    # === Backend options ===
    backend: str = "rocm"
    target: str = "gfx942"
    gpu_native_math_precision: bool = False
    iree_preprocessing_pass_pipeline: str = None

    # === Benchmark options ===
    run_bench: bool = False
    benchmark_batch_size: int = None
    benchmark_repetitions: int = None
    benchmark_results_file: str = None
    capture_trace: bool = False
    bench_with_constant_weights: bool = False
    bench_file: str = None

    # === Cache options ===
    kernel_hash: str = None

    # === Debug options ===
    create_vmfb_file: str = None
    override_mlir: str = None
    dump_binaries: str = None
    dump_intermediates: str = False
    compile_to_mlir: bool = False
    debug_info: bool = False
    use_local_scope: bool = False

    # === Performance options ===
    denorm_fp_math_f32: str = None
    waves_per_eu: int = None
    wave_runtime: str = None
    use_buffer_load_ops: bool = False
    use_buffer_store_ops: bool = False
    use_fast_math: bool = False

    # === Print options ===
    print_ir_after: list[str] = field(default_factory=list)
    print_ir_before: list[str] = field(default_factory=list)
    print_trace_begin: bool = False
    print_grid: bool = False
    print_signature: bool = False
    print_mlir: bool = False
    print_ir_after_all: bool = False
