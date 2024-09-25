# Copyright 2023 Nod Labs, Inc
# SPDX-FileCopyrightText: 2024 The IREE Authors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from . import gen
from . import lang
from . import wave


# Helpers that are good to have in the global scope.
def __getattr__(name):
    if name == "DEBUG":
        return lang.is_debug()
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


# Dynamic attributes so that IDEs see them.
DEBUG: bool
