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
    kernel, hyperparams, _, _ = get_speculative_sampling_kernel(batch_size=64, num_speculative_tokens=3)

    # Create the kernel with the hyperparameters
    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=False,
        compile_to_mlir=True,
    )
    kernel = wave_compile(options, kernel)
    print(kernel.asm)

    # CHECK: #map = affine_map<()[s0] -> (s0 + (s0 floordiv 64) * 64)>
    # CHECK: #translation = #iree_codegen.translation_info<pipeline = None workgroup_size = [64, 1, 1] subgroup_size = 64>
    # CHECK: module attributes {transform.with_named_sequence} {
    # CHECK:   stream.executable private @tree_speculative_sampling {
    # CHECK:     stream.executable.export public @tree_speculative_sampling workgroups() -> (index, index, index) {
    # CHECK:       %c1 = arith.constant 1 : index
    # CHECK:       stream.return %c1, %c1, %c1 : index, index, index
    # CHECK:     }
    # CHECK:     builtin.module {
    # CHECK:       func.func @tree_speculative_sampling(%arg0: !stream.binding, %arg1: !stream.binding, %arg2: !stream.binding) attributes {translation_info = #translation} {
    # CHECK:         %c0 = arith.constant 0 : index
    # CHECK:         %cst = arith.constant dense<0.000000e+00> : vector<1xf32>
    # CHECK:         %thread_id_x = gpu.thread_id  x
    # CHECK:         %0 = stream.binding.subspan %arg0[%c0] : !stream.binding -> memref<64xf32, strided<[1], offset: ?>>
    # CHECK:         %1 = affine.apply #map()[%thread_id_x]
    # CHECK:         %2 = vector.load %0[%1] : memref<64xf32, strided<[1], offset: ?>>, vector<1xf32>
    # CHECK:         %129 = stream.binding.subspan %arg1[%c0] : !stream.binding -> memref<64xf32, strided<[1], offset: ?>>
    # CHECK:         %130 = vector.load %129[%1] : memref<64xf32, strided<[1], offset: ?>>, vector<1xf32>
    # CHECK:         %194 = arith.subf %2, %130 : vector<1xf32>
    # CHECK:         %258 = arith.maximumf %194, %cst : vector<1xf32>
    # CHECK:         %322 = stream.binding.subspan %arg2[%c0] : !stream.binding -> memref<64xf32, strided<[1], offset: ?>>
    # CHECK:         vector.store %258, %322[%1] : memref<64xf32, strided<[1], offset: ?>>, vector<1xf32>
    # CHECK:         return
    # CHECK:       }
    # CHECK:     }
    # CHECK:   }
    # CHECK:   func.func @isolated_benchmark(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>) -> tensor<64xf32> {
    # CHECK:     %0 = flow.dispatch @tree_speculative_sampling::@tree_speculative_sampling(%arg0, %arg1, %arg2) : (tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> %arg2
    # CHECK:     return %0 : tensor<64xf32>
    # CHECK:   }
    # CHECK: }
