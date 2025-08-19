# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from . import gen
from . import lang


# Helpers that are good to have in the global scope.
def __getattr__(name):
    if name == "DEBUG":
        return lang.is_debug()
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


# Dynamic attributes so that IDEs see them.
DEBUG: bool
