# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import torch
import math
import iree.turbine.kernel as tk
from iree.turbine.kernel.lang.global_symbols import *
from iree.turbine.kernel.wave.utils.run_utils import (
    set_default_run_config,
)
from iree.turbine.kernel.wave.utils.general_utils import (
    get_default_scheduling_params,
)
from iree.turbine.kernel.wave.utils.torch_utils import (
    device_randn,
    device_zeros,
)
from iree.turbine.kernel.wave.compile import WaveCompileOptions, wave_compile
from iree.turbine.kernel.wave.constraints import MMAType
from iree.turbine.kernel.wave.templates.decode_attention import (
    get_decode_attention_kernels,
)
from iree.turbine.kernel.wave.scheduling.schedule import SchedulingType
import os
from torch.testing import assert_close
from ..common.utils import (
    require_e2e,
    param_bool,
    enable_scheduling_barriers,
    dump_generated_mlir,
)
from ..common.shapes import get_test_shapes
from iree.turbine.kernel.wave.templates.gqa_decode_attention import (
    get_gqa_decode_attention_kernels,
)
from iree.turbine.kernel.wave.templates.attention_common import (
    AttentionShape,
)
import torch.nn.functional as F


def tree_speculative_sampling_target_only(
    predicts,                   # [seq_len], mutable
    output_token_ids,           # [batch_size, num_speculative_tokens], mutable
    output_accepted_token_num,  # [batch_size], mutable
    candidates,                 # [batch_size, num_draft_tokens]
    retrive_index,              # [batch_size, num_draft_tokens]
    retrive_next_token,         # [batch_size, num_draft_tokens]
    retrive_next_sibling,       # [batch_size, num_draft_tokens]
    uniform_samples,            # [batch_size, num_draft_tokens]
    target_probs,               # [batch_size, num_draft_tokens, vocab_size]
    draft_probs,                # [batch_size, num_draft_tokens, vocab_size]
    batch_size,
    num_speculative_tokens,
    num_draft_tokens,
    d,
    threshold_single=1.0,
    threshold_acc=1.0,
    deterministic=True):
    
    threshold_acc = max(threshold_acc, 1e-9)
    for bx in range(batch_size):
        prob_acc = 0.0
        cur_prob_offset = 0  # bx * num_draft_tokens * d handled via indexing
        coin = uniform_samples[bx, 0]
        last_accepted_retrive_idx = retrive_index[bx, 0]
        output_token_ids[bx, 0] = last_accepted_retrive_idx
        num_accepted_tokens = 0
        cur_index = 0
        
        for j in range(1, num_speculative_tokens):
            cur_index = retrive_next_token[bx, cur_index]
            while cur_index != -1:
                draft_index = retrive_index[bx, cur_index]
                draft_token_id = candidates[bx, cur_index]
                target_prob_single = target_probs[bx, cur_prob_offset, draft_token_id]
                prob_acc += target_prob_single
                
                if (coin <= prob_acc / threshold_acc or 
                    target_prob_single >= threshold_single):
                    # accept token
                    prob_acc = 0.0
                    cur_prob_offset = cur_index
                    coin = uniform_samples[bx, cur_index]
                    predicts[last_accepted_retrive_idx] = draft_token_id
                    num_accepted_tokens += 1
                    output_token_ids[bx, num_accepted_tokens] = draft_index
                    last_accepted_retrive_idx = draft_index
                    break
                else:
                    draft_probs[bx, cur_index, draft_token_id] = target_probs[bx, cur_index, draft_token_id]
                    cur_index = retrive_next_sibling[bx, cur_index]
            
            if cur_index == -1:
                break
        
        output_accepted_token_num[bx] = num_accepted_tokens
        
        # Sample from relu(target_probs - draft_probs)
        last_offset = cur_prob_offset
        q = target_probs[bx, last_offset]
        p = draft_probs[bx, last_offset] if num_accepted_tokens != num_speculative_tokens - 1 else torch.zeros_like(q)
        relu_diff = torch.relu(q - p)
        sum_relu = relu_diff.sum()

        u = coin * sum_relu
        sampled_id = d - 1
        aggregate = 0.0
        for i in range(d):
            val = relu_diff[i]
            if val <= 0:
                continue
            aggregate += val
            if aggregate > u:
                sampled_id = i
                break

        predicts[last_accepted_retrive_idx] = sampled_id

        # Sample from residual distribution
#       if num_accepted_tokens < num_speculative_tokens - 1:
#           residual_probs = (target_probs[bx, cur_prob_offset] -
#                            draft_probs[bx, cur_prob_offset]).clamp(min=0)
#           sum_residual = residual_probs.sum()
#
#           if deterministic:
#               # Mimic the deterministic sampling in CUDA kernel
#               u = coin * sum_residual
#               cumsum = 0.0
#               sampled_id = d - 1  # default to last token
#               for i in range(d):
#                   cumsum += residual_probs[i]
#                   if cumsum > u:
#                       sampled_id = i
#                       break
#           else:
#               if sum_residual > 1e-8:
#                   sampled_id = torch.multinomial(residual_probs, 1).item()
#               else:
#                   sampled_id = d - 1
#
#           predicts[last_accepted_retrive_idx] = sampled_id


test_cases = [
    (
        1,
        1,
        [3, -1, -1, 4, 5, 18, 11, -1, -1, -1, 12, 18],
        [[0, 3, 4, 5], [6, 10, 11, -1]],
        [3, 2],
    ),
    (
        0,  # threshold_single
        0,  # threshold_acc
        [1, 2, 18, -1, -1, -1, 11, -1, -1, -1, 12, 18],
        [[0, 1, 2, -1], [6, 10, 11, -1]],
        [2, 2],
    ),
]


@require_e2e
@pytest.mark.parametrize(
    "threshold_single, threshold_acc, expected_predicts, expected_accept_index, expected_accept_token_num",
    test_cases,
)
def testReferenceSpeculativeDecoding(
    threshold_single,
    threshold_acc,
    expected_predicts,
    expected_accept_index,
    expected_accept_token_num,
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

    tree_speculative_sampling_target_only(
        predicts=predicts,
        output_token_ids=accept_index,
        output_accepted_token_num=accept_token_num,
        candidates=candidates,
        retrive_index=retrive_index,
        retrive_next_token=retrive_next_token,
        retrive_next_sibling=retrive_next_sibling,
        uniform_samples=coins,
        target_probs=target_probs,
        draft_probs=draft_probs,
        batch_size=bs,
        num_speculative_tokens=num_spec_step,
        num_draft_tokens=num_draft_tokens,
        d=vocab_size,
        threshold_single=threshold_single,
        threshold_acc=threshold_acc,
        deterministic=True,
    )

    assert (
        predicts.tolist() == expected_predicts
    ), f"Predicts mismatch for thresholds ({threshold_single}, {threshold_acc})"
    assert (
        accept_index.tolist() == expected_accept_index
    ), f"Accept index mismatch for thresholds ({threshold_single}, {threshold_acc})"
    assert (
        accept_token_num.tolist() == expected_accept_token_num
    ), f"Accept token num mismatch for thresholds ({threshold_single}, {threshold_acc})"
