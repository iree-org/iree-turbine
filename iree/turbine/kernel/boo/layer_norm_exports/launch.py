# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from .layer_norm import LayerNormSignature
from ....runtime import Launchable
from ..runtime import get_launchable as generic_get_launchable


# TODO: describe that `use_custom` means using the templated MLIR kernel rather than regular pytorch call.
# It is unclear though whether the "regular" python call is just using pytorch or is getting intercepted
# at torch.fx level...  FIXME: it is removed entirely for now, not doing the templated stuff
# TODO(azinenko): this looks generalizable over signature type
def get_launchable(signature: LayerNormSignature, *, cache_only=False) -> Launchable:
    func_name = signature.get_func_name()
    module_factory = lambda: signature.get_nn_module()
    arg_factory = lambda: signature.get_sample_args(splat_value=0)
    return generic_get_launchable(
        module_factory=module_factory,
        arg_factory=arg_factory,
        func_name=func_name,
        cache_only=cache_only,
        force_single_dispatch=True,
    )
