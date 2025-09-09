# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from ..exports.signature import OpSignature
from ....runtime import Launchable
from ..runtime import get_module_asm as generic_get_module_asm
from ..runtime import get_launchable as generic_get_launchable


def get_module_asm(
    signature: OpSignature, func_name: str | None = None, use_custom: bool = True
) -> str:
    func_name = func_name or signature.func_name
    module_factory = lambda: signature.get_nn_module(use_custom=use_custom)
    arg_factory = lambda: signature.get_sample_args(splat_value=0)
    return generic_get_module_asm(
        module_factory, arg_factory, func_name, force_single_dispatch=True
    )


def get_launchable(
    signature: OpSignature, *, use_custom=True, cache_only=False
) -> Launchable:
    func_name = signature.func_name
    module_factory = lambda: signature.get_nn_module(use_custom=use_custom)
    arg_factory = lambda: signature.get_sample_args(splat_value=0)
    return generic_get_launchable(
        module_factory=module_factory,
        arg_factory=arg_factory,
        func_name=func_name,
        cache_only=cache_only,
        force_single_dispatch=signature.force_single_dispatch,
    )
