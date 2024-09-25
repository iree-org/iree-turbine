<!--
SPDX-FileCopyrightText: 2024 The IREE Authors

SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

# Importers from various systems

This directory is self-contained and intended to be shared with other
projects with its source-of-truth in torch-mlir.

All MLIR API dependencies must route through the relative `ir.py`, which
it is expected that sub-projects will customize accordingly.
