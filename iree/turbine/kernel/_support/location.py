# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from iree.compiler.ir import Location, Context
from typing import Union
import sys
import inspect
from dataclasses import dataclass


@dataclass
class FileLineColInfo:
    """
    Location information containing the filename, line and column.
    """

    filename: str
    line: Union[int, tuple[int, int]]
    col: Union[int, tuple[int, int]]

    def to_mlir(self):
        assert Context.current is not None, "Must be called under MLIR context manager."

        line_is_range = isinstance(self.line, tuple)
        col_is_range = isinstance(self.col, tuple)
        if not line_is_range and not col_is_range:
            return Location.file(self.filename, self.line, self.col)
        line_start = self.line[0] if line_is_range else self.line
        line_end = self.line[1] if line_is_range else self.line
        col_start = self.col[0] if col_is_range else self.col
        col_end = self.col[1] if col_is_range else self.col
        return Location.file(self.filename, line_start, col_start, line_end, col_end)

    @staticmethod
    def capture_current_location():
        # Need to find a part of the call stack that doesn't belong to us.
        for f in inspect.stack():
            if "iree/turbine/kernel" not in f.filename:
                break
        if not f:
            f = inspect.stack()[-1]

        # Column information is only available for Python >= 3.11.
        assert sys.version_info.major == 3, "Unexpected Python version"
        if sys.version_info.minor < 11:
            return FileLineColInfo(f.filename, f.lineno, 0)

        return FileLineColInfo(
            f.filename,
            (f.positions.lineno, f.positions.end_lineno),
            (f.positions.col_offset, f.positions.end_col_offset),
        )
