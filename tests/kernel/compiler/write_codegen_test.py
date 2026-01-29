# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for guarded store dispatch logic.

Note: Full IR emission tests require ThreadEmitter context and are covered
by vector_codegen integration tests. These tests verify dispatch behavior.
"""

import logging
import unittest

from iree.turbine.kernel.compiler.write_analysis import AnalysisResult
from iree.turbine.kernel.compiler.write_codegen import emit_guarded_store
from iree.turbine.kernel.compiler import base


class EmitGuardedStoreTest(unittest.TestCase):
    def setUp(self):
        self._original = base.options.enable_single_writer_guards

    def tearDown(self):
        base.options.enable_single_writer_guards = self._original

    def testProvenUniqueCallsStoreDirectly(self):
        """PROVEN_UNIQUE should call store function without guards."""
        base.options.enable_single_writer_guards = True
        called = []
        emit_guarded_store(None, AnalysisResult.unique(), lambda: called.append(1))
        self.assertEqual(len(called), 1)

    def testGuardsDisabledAlwaysCallsStore(self):
        """When guards disabled, all outcomes call store directly."""
        base.options.enable_single_writer_guards = False
        called = []
        emit_guarded_store(None, AnalysisResult.unique(), lambda: called.append(1))
        emit_guarded_store(None, AnalysisResult.guard(), lambda: called.append(1))
        self.assertEqual(len(called), 2)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
