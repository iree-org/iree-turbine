# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

NDEBUG = False

from dataclasses import dataclass


class CodegenError(Exception): ...


class ValidationError(CodegenError): ...


@dataclass
class CodegenOptions:
    """Configuration options for kernel code generation."""
    enable_single_writer_guards: bool = True
    guard_diagnostic_level: int = 0


options = CodegenOptions()
