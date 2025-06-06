# RUN: python %s | FileCheck %s

from iree.turbine.kernel.lang.global_symbols import *
from iree.turbine.kernel.wave.compile import WaveCompileOptions, wave_compile
from iree.turbine.kernel.wave.utils.general_utils import (
    run_test,
)
from iree.turbine.kernel.wave.templates.speculative_decoding import (
    get_speculative_sampling_kernel,
)


@run_test
def test_speculative_decoding():
    # Get the kernel and its hyperparameters
    kernel, hyperparams, _, _ = get_speculative_sampling_kernel(
        batch_size=10,
        num_speculative_tokens=3,
        threshold_single=0.01,
        threshold_acc=0.01,
        num_draft_tokens=6,
        vocab_size=20,
        seq_len=12,
    )

    # Create the kernel with the hyperparameters
    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        compile_to_mlir=True,
    )
    kernel = wave_compile(options, kernel)
    print(kernel.asm)

    # CHECK-LABEL: func.func @speculative_sampling
    # CHECK: (%arg0: !stream.binding, %arg1: !stream.binding, %arg2: !stream.binding, %arg3: !stream.binding,
    # CHECK: %arg4: !stream.binding, %arg5: !stream.binding, %arg6: !stream.binding, %arg7: !stream.binding,
    # CHECK: %arg8: !stream.binding, %arg9: !stream.binding, %arg10: !stream.binding, %arg11: !stream.binding)
    # CHECK: vector.load
    # CHECK: vector.load
    # CHECK: vector.store
    # CHECK: scf.while
    # CHECK: vector.extractelement
    # CHECK: scf.condition
    # CHECK: do
    # CHECK: ^bb0
    # CHECK: scf.while
    # CHECK: scf.condition
    # CHECK: arith.divf
    # CHECK: arith.cmpf
    # CHECK: arith.xori
    # CHECK: arith.select
    # CHECK: scf.if
    # CHECK: vector.store
    # CHECK: scf.if
    # CHECK: affine.apply
    # CHECK: vector.maskedstore
    # CHECK: vector.load
    # CHECK: scf.yield
    # CHECK: scf.yield
    # CHECK: vector.store
    # CHECK: arith.select
    # CHECK: arith.addf
    # CHECK: arith.select
    # CHECK: gpu.shuffle  xor
    # CHECK: arith.minsi
    # CHECK: return
    # CHECK-LABEL: func.func @isolated_benchmark(
    # CHECK: flow.dispatch @speculative_sampling::@speculative_sampling(
    # CHECK-SAME: %arg0, %arg1, %arg2, %arg3,
    # CHECK-SAME: %arg4, %arg5, %arg6, %arg7,
    # CHECK-SAME: %arg8, %arg9, %arg10, %arg11)
