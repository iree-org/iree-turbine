# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from iree.compiler.ir import Location, Context
from typing import Optional, List, Union
import sys
import inspect
from dataclasses import dataclass
from .location_config import LocationCaptureConfig, LocationCaptureLevel


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

        return FileLineColInfo.from_stack_frame(f)

    @staticmethod
    def from_stack_frame(frame: inspect.FrameInfo):
        # Column information is only available for Python >= 3.11.
        assert sys.version_info.major == 3, "Unexpected Python version"
        if sys.version_info.minor < 11:
            return FileLineColInfo(frame.filename, frame.lineno, 0)

        return FileLineColInfo(
            frame.filename,
            (frame.positions.lineno, frame.positions.end_lineno),
            (frame.positions.col_offset, frame.positions.end_col_offset),
        )


@dataclass
class StackTraceInfo:
    """
    Locations of an entire stack trace, each with FileLineColInfo.
    """

    frames: List[FileLineColInfo]

    def to_mlir(self) -> Location:
        assert Context.current is not None, "Must be called under MLIR context manager."
        if not self.frames:
            return Location.unknown()
        if len(self.frames) == 1:
            return self.frames[0].to_mlir()
        return Location.callsite(
            self.frames[0].to_mlir(), [f.to_mlir() for f in self.frames[1:]]
        )

    @staticmethod
    def capture_current_location(*, preserve_system_frames=False) -> "StackTraceInfo":
        # TODO: we may want to cache location info so we don't keep copying the
        # top of the stack everywhere. MLIR uniquing takes care of this when we
        # convert currently.
        frames = [
            FileLineColInfo.from_stack_frame(f)
            for f in inspect.stack()
            if "iree/turbine/kernel" not in f.filename or preserve_system_frames
        ]
        return StackTraceInfo(frames)


def capture_location(
    location_capture_config: Optional[LocationCaptureConfig],
) -> Optional[FileLineColInfo | StackTraceInfo]:
    if (
        not location_capture_config
        or location_capture_config.level == LocationCaptureLevel.NONE
    ):
        return None
    if location_capture_config.level == LocationCaptureLevel.FILE_LINE_COL:
        return FileLineColInfo.capture_current_location()
    if location_capture_config.level == LocationCaptureLevel.STACK_TRACE:
        return StackTraceInfo.capture_current_location()
    if location_capture_config.level == LocationCaptureLevel.STACK_TRACE_WITH_SYSTEM:
        return StackTraceInfo.capture_current_location(preserve_system_frames=True)
