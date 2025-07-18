# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import gc
import shutil
import tempfile
import torch
import unittest


from parameterized import parameterized
from pathlib import Path

from iree.turbine.support.conversions import TORCH_DTYPE_TO_IREE_TYPE_ASM
from iree.turbine.support.tools import iree_tool_prepare_input_args, read_raw_tensor

TYPES_TO_TEST = list(TORCH_DTYPE_TO_IREE_TYPE_ASM.keys())
TYPES_TO_TEST.remove(torch.int8)
TYPES_TO_TEST.remove(torch.uint8)
TYPES_TO_TEST.remove(torch.qint8)
TYPES_TO_TEST.remove(torch.quint8)


class RawTensorSaveTest(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self._tmp_dir = Path(tempfile.mkdtemp(type(self).__qualname__))

    def tearDown(self):
        gc.collect()
        shutil.rmtree(self._tmp_dir, ignore_errors=True)

    @parameterized.expand(TYPES_TO_TEST)
    def testBuiltInTorchTypes(self, dtype: torch.dtype = torch.float32):
        pre_clamp_torch = torch.tensor(
            [[0, 1, 3.3, 4.5, 11.5], [-1, -3, 900, 1e90, -1e90]], dtype=torch.float64
        )

        _min, _max = 0, 1
        if dtype.is_floating_point or dtype.is_complex:
            _min, _max = torch.finfo(dtype).min, torch.finfo(dtype).max
        elif dtype != torch.bool:
            _min, _max = torch.iinfo(dtype).min, torch.iinfo(dtype).max

        original_torch = pre_clamp_torch.clamp(_min, _max).to(dtype)

        paths = iree_tool_prepare_input_args(
            [original_torch],
            file_path_prefix=self._tmp_dir,
        )

        read_torch = read_raw_tensor(paths[0])

        self.assertEqual(original_torch.dtype, dtype)
        self.assertEqual(read_torch.dtype, dtype)
        self.assertEqual(original_torch.shape, read_torch.shape)
        self.assertTrue(torch.all(original_torch == read_torch))
