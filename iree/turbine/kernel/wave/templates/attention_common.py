# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from dataclasses import dataclass
from typing import Optional

import iree.turbine.kernel.lang as tkl


@dataclass
class AttentionShape:
    num_query_heads: int
    num_kv_heads: int
    head_size: int
    head_size_kv: int
    # -----------------------
    # Prefill specific
    num_seqs: Optional[int] = None
    max_seq_len: Optional[int] = None
    total_seq_len: Optional[int] = None
    # -----------------------
    # Vanilla attention
    query_seq_len: Optional[int] = None
    kv_seq_len: Optional[int] = None


# Commonly-used attention symbols
H = tkl.sym.H
H_Q = tkl.sym.H
H_KV = tkl.sym.H
N_Q = tkl.sym.N_D
N_KV = tkl.sym.N_KV
D_Q = tkl.sym.D_Q
D_KV = tkl.sym.D_KV

BLOCK_H = tkl.sym.BLOCK_H
BLOCK_H_Q = tkl.sym.BLOCK_H
BLOCK_H_KV = tkl.sym.BLOCK_H
BLOCK_N_Q = tkl.sym.BLOCK_N_D
BLOCK_N_KV = tkl.sym.BLOCK_N_KV
BLOCK_D_Q = tkl.sym.BLOCK_D_Q
BLOCK_D_KV = tkl.sym.BLOCK_D_KV
