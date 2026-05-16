# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import unittest
import sympy

from iree.turbine.kernel.lang import sym
from iree.turbine.kernel.lang.tkw_types import IndexMapping
from iree.turbine.kernel.compiler.write_analysis import (
    analyze_write,
    AnalysisOutcome,
    AnalysisResult,
    OwnerPredicate,
)

M = sym.M
N = sym.N
K = sym.K


class WriteAnalysisTest(unittest.TestCase):
    def testNoneMappingIsUnique(self):
        """None mapping (identity) should be PROVEN_UNIQUE."""
        result = analyze_write(None, (M, N))
        self.assertEqual(result.outcome, AnalysisOutcome.PROVEN_UNIQUE)

    def testIdentityMappingIsUnique(self):
        """Identity IndexMapping should be PROVEN_UNIQUE."""
        i0 = IndexMapping.iterator(0)
        i1 = IndexMapping.iterator(1)
        mapping = IndexMapping(2, {M: i0, N: i1}, {M: i0, N: i1})
        result = analyze_write(mapping, (M, N))
        self.assertEqual(result.outcome, AnalysisOutcome.PROVEN_UNIQUE)

    def testHasIdentityFlagIsUnique(self):
        """has_identity=True should bypass analysis and return PROVEN_UNIQUE."""
        i0 = IndexMapping.iterator(0)
        # Non-identity mapping (broadcast to constant)
        mapping = IndexMapping(1, {M: i0}, {N: 0})
        result = analyze_write(mapping, (N,), has_identity=True)
        self.assertEqual(result.outcome, AnalysisOutcome.PROVEN_UNIQUE)

    def testBroadcastIsOwnerPredicate(self):
        """Broadcast (constant output) should be OWNER_PREDICATE."""
        i0 = IndexMapping.iterator(0)
        # All outputs are constants - broadcast pattern
        mapping = IndexMapping(1, {M: i0}, {N: 0})
        result = analyze_write(mapping, (N,))
        self.assertEqual(result.outcome, AnalysisOutcome.OWNER_PREDICATE)
        self.assertEqual(result.predicate.axis, 0)

    def testFloorDivNeedsGuard(self):
        """Floor division pattern should be NEEDS_GUARD."""
        i0 = IndexMapping.iterator(0)
        # Floor division creates many-to-one mapping
        mapping = IndexMapping(1, {M: i0}, {N: sympy.floor(i0 / 2)})
        result = analyze_write(mapping, (N,))
        self.assertEqual(result.outcome, AnalysisOutcome.NEEDS_GUARD)


class OwnerPredicateTest(unittest.TestCase):
    def testFrozen(self):
        """OwnerPredicate should be immutable."""
        pred = OwnerPredicate(axis=0, value=0)
        with self.assertRaises(AttributeError):
            pred.axis = 1


class AnalysisResultTest(unittest.TestCase):
    def testUniqueFactory(self):
        result = AnalysisResult.unique()
        self.assertEqual(result.outcome, AnalysisOutcome.PROVEN_UNIQUE)

    def testOwnerFactory(self):
        result = AnalysisResult.owner(axis=1, value=0)
        self.assertEqual(result.outcome, AnalysisOutcome.OWNER_PREDICATE)
        self.assertEqual(result.predicate.axis, 1)

    def testGuardFactory(self):
        result = AnalysisResult.guard()
        self.assertEqual(result.outcome, AnalysisOutcome.NEEDS_GUARD)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
