# SPDX-FileCopyrightText: 2024 The IREE Authors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

NDEBUG = False


class CodegenError(Exception):
    ...


class ValidationError(CodegenError):
    ...
