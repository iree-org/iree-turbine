#!/bin/bash

# SPDX-FileCopyrightText: 2024 The IREE Authors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

python main.py --limit 500 -j 8 | grep "NotImplementedError: Unimplemented torch op in the IREE compiler" | grep -o "'[^']*'" | sed "s/'//g" > unimplemented_torch_ops.txt
