# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import torch
import math
import os
import iree.turbine.kernel as tk
from iree.turbine.kernel.lang.global_symbols import *
from iree.turbine.kernel.wave.utils.run_utils import (
    set_default_run_config,
)
from iree.turbine.kernel.wave.utils.general_utils import (
    get_default_scheduling_params,
)
from iree.turbine.kernel.wave.compile import WaveCompileOptions, wave_compile
from iree.turbine.kernel.wave.templates.speculative_decoding import (
    get_speculative_decoding_kernel,
)
from ..common.utils import (
    require_e2e,
    enable_scheduling_barriers,
    dump_generated_mlir,
)
import torch.nn.functional as F


test_cases = [
    (
        1,
        1,
      # [3, -1, -1, 4, 5, 18, 11, -1, -1, -1, 12, 18],
        [3, -1, -1, 4, 5, -1, 11, -1, -1, -1, 12, -1],
        [[0, 3, 4, 5], [6, 10, 11, -1]],
        [3, 2],
    ),
    (
        0,  # threshold_single
        0,  # threshold_acc
        [1, 2, -1, -1, -1, -1, 11, -1, -1, -1, 12, -1],
      # [1, 2, 18, -1, -1, -1, 11, -1, -1, -1, 12, 18],
        [[0, 1, 2, -1], [6, 10, 11, -1]],
        [2, 2],
    ),
]


@require_e2e
@pytest.mark.parametrize(
    "threshold_single, threshold_acc, expected_predicts, expected_accept_index, expected_accept_token_num",
    test_cases,
)
def testSpeculativeDecoding(
    threshold_single,
    threshold_acc,
    expected_predicts,
    expected_accept_index,
    expected_accept_token_num,
    request,
):
    device = "cuda"

    candidates = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5],
            [7, 8, 9, 10, 11, 12],
        ],
        dtype=torch.int32,
        device=device,
    )
    retrive_index = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10, 11],
        ],
        dtype=torch.int32,
        device=device,
    )
    retrive_next_token = torch.tensor(
        [
            [1, 2, -1, 4, 5, -1],
            [4, 2, 3, -1, 5, -1],
        ],
        dtype=torch.int32,
        device=device,
    )
    retrive_next_sibling = torch.tensor(
        [
            [-1, 3, -1, -1, -1, -1],
            [-1, -1, -1, -1, 1, -1],
        ],
        dtype=torch.int32,
        device=device,
    )

    target_logits = torch.full((2, 6, 20), 1, dtype=torch.float32, device=device)
    target_logits[0, 0, 3] = 10
    target_logits[0, 3, 4] = 10
    target_logits[0, 4, 5] = 10
    target_logits[1, 0, 11] = 10
    target_logits[1, 4, 12] = 10

    for i in range(target_logits.shape[0]):
        for j in range(target_logits.shape[1]):
            if torch.max(target_logits[i, j]) < 10:
                target_logits[i, j, 18] = 10

    temperatures = torch.tensor([0.01, 0.01], dtype=torch.float32, device=device)
    bs, num_draft_tokens = candidates.shape
    num_spec_step = len(expected_accept_index[0])
    predict_shape = (len(expected_predicts),)

    predicts = torch.full(predict_shape, -1, dtype=torch.int32, device=device)
    accept_index = torch.full((bs, num_spec_step), -1, dtype=torch.int32, device=device)
    accept_token_num = torch.full((bs,), 0, dtype=torch.int32, device=device)

    expanded_temperature = temperatures.unsqueeze(1).unsqueeze(1)
    target_probs = F.softmax(target_logits / expanded_temperature, dim=-1)
    draft_probs = torch.full_like(target_probs, 0, dtype=torch.float32, device=device)
    coins = torch.rand(bs, num_draft_tokens, device=device, dtype=torch.float32)

    vocab_size = target_probs.shape[2]

    run_bench = request.config.getoption("--runperf")
    dump_perf = request.config.getoption("--dump-perf-files-path")
    speculative_decoding, symbols = get_speculative_decoding_kernel(
        bs, num_spec_step, num_draft_tokens, vocab_size, threshold_single, threshold_acc, True
    )

    symbols.update(get_default_scheduling_params())

    options = WaveCompileOptions(
        subs=symbols,
        canonicalize=True,
        run_bench=run_bench,
        waves_per_eu=2,
        denorm_fp_math_f32="preserve-sign",
        schedule=False,
        use_scheduling_barriers=enable_scheduling_barriers,
        benchmark_batch_size=1000,
        benchmark_repetitions=3,
        benchmark_results_file=(
            os.path.join(dump_perf, "tk_" + request.node.name + ".json")
            if dump_perf
            else None
        ),
    )
    options = set_default_run_config(options)
    speculative_decoding = wave_compile(options, speculative_decoding)

    asm = speculative_decoding(
        predicts=predicts,
        accept_index=accept_index,
        accept_token_num=accept_token_num,
        candidates=candidates,
        retrive_index=retrive_index,
        retrive_next_token=retrive_next_token,
        retrive_next_sibling=retrive_next_sibling,
        uniform_samples=coins,
        target_probs=target_probs,
        draft_probs=draft_probs,
    )

    if dump_generated_mlir:
        filename = f"wave_speculative_decoding_{threshold_single}_{threshold_acc}.mlir"
        with open(filename, "w") as f:
            f.write(asm)

    assert (
        predicts.tolist() == expected_predicts
    ), f"Predicts mismatch for thresholds ({threshold_single}, {threshold_acc})"
    assert (
        accept_index.tolist() == expected_accept_index
    ), f"Accept index mismatch for thresholds ({threshold_single}, {threshold_acc})"
    assert (
        accept_token_num.tolist() == expected_accept_token_num
    ), f"Accept token num mismatch for thresholds ({threshold_single}, {threshold_acc})"
