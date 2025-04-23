# Copyright 2025 The IREE Authors
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from iree.compiler import compile_str
from ...compiler.ir import (
    builtin_d,
    InsertionPoint,
    Location,
    Operation,
    transform_d,
    UnitAttr,
    Value,
)
from iree.compiler.dialects.transform import (
    interpreter as transform_interpreter,
    any_op_t,
)
from iree.compiler.dialects import (
    _structured_transform_ops_gen as structured_transform_ops,
)
from ..compile_options import WaveCompileOptions


def compile_to_vmfb(
    asm: str,
    options: WaveCompileOptions,
):

    flags = [
        f"--iree-hal-target-backends={options.backend}",
        "--iree-vm-bytecode-module-strip-source-map=true",
        "--iree-opt-strip-assertions=true",
        "--iree-vm-target-truncate-unsupported-floats",
    ]

    # TODO: More targets/backends support.
    if options.backend == "rocm":
        flags.append(f"--iree-hip-target={options.target}")

    if options.gpu_native_math_precision:
        # Polynomial approximation passes in MLIR/IREE often generate
        # suboptimal code with redundant clamps and fptosi. This flag
        # allows us to skip unnecessary approx for GPU.
        flags.append("--iree-codegen-gpu-native-math-precision=true")

    if options.print_ir_after_all:
        flags.append("--mlir-print-ir-after-all")

    if options.iree_preprocessing_pass_pipeline:
        flags.append(
            f"--iree-preprocessing-pass-pipeline={options.iree_preprocessing_pass_pipeline}"
        )

    if options.dump_intermediates:
        flags.append(
            f"--iree-hal-dump-executable-intermediates-to={options.dump_intermediates}"
        )

    if options.dump_binaries:
        flags.append(f"--iree-hal-dump-executable-binaries-to={options.dump_binaries}")

    if options.run_bench:
        if options.benchmark_batch_size:
            flags.append(
                f"--iree-hal-benchmark-dispatch-repeat-count={options.benchmark_batch_size}"
            )

    res = compile_str(asm, target_backends=[options.backend], extra_args=flags)
    return res


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
                loops = structured_transform_ops.structured_match(
                    any_op_t(), target, ops=["scf.for", "scf.while"]
                )
                transform_d.apply_licm(loops)
                transform_d.YieldOp([target])
        transform_interpreter.apply_named_sequence(
            module,
            transform_module.body.operations[0],
            transform_module,
        )


def set_default_compile_config(options: WaveCompileOptions) -> WaveCompileOptions:
    """Return default config for compilation."""
    options.backend = "rocm"
    options.device = "hip"
    options.target = "gfx942"
    return options
