# RUN: python %s | FileCheck %s

from typing import Callable
import iree.turbine.kernel as tk
import iree.turbine.kernel.lang as tkl
import iree.turbine.kernel.wave as tkw
from iree.turbine.kernel.lang.global_symbols import *
from iree.turbine.kernel.wave.compile import WaveCompileOptions, wave_compile
from iree.turbine.kernel.wave.utils.general_utils import (
    run_test,
)
from iree.turbine.kernel.wave.templates.speculative_decoding import (
    get_speculative_decoding_kernel,
    get_speculative_sampling_kernel,
)


@run_test
def test_speculative_decoding():
    # Get the kernel and its hyperparameters
    kernel, hyperparams, _ = get_speculative_sampling_kernel(
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

    # CHECK: scf.while
    # CHECK: scf.condition
    # CHECK: scf.while
    # CHECK: scf.condition
    # CHECK: arith.divf
    # CHECK: arith.cmpf
    # CHECK: arith.xori
    # CHECK: arith.select
    # CHECK: scf.if
    # CHECK: vector.store
    # CHECK: scf.if
    # CHECK: vector.load
    # CHECK: vector.store
    # CHECK: arith.select
    # CHECK: arith.addf
    # CHECK: arith.select
    # CHECK: scf.yield
    # CHECK: scf.yield
    # CHECK: return
