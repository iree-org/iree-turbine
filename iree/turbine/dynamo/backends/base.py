# Copyright 2025 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from ..passes import turbine_cpu_pass_pipeline
from ...transforms.general.custom_op_expansion import ExpandCustomOpsPass
from iree.turbine.runtime.launch import Launchable
from iree.turbine.support.logging import aot_logger as logger

import torch
from torch._dynamo.backends.common import aot_autograd
from functorch.compile import make_boxed_func
from iree.compiler.extras.fx_importer import (
    FxImporter,
)


def _backend(gm: torch.fx.GraphModule, example_inputs):
    """Generic backend which creates and preloads a launchable."""
    # Export the graph module to mlir.
    gm = turbine_cpu_pass_pipeline(gm, example_inputs)
    logger.debug("Traced Graph Module:\n%s", str(gm))
    fx_importer = FxImporter()
    fx_importer.import_graph_module(gm)
    module_op = fx_importer.module_op
    logger.debug("Successfully imported gm to mlir:\n%s", module_op)
    expansion_pass = ExpandCustomOpsPass(module_op)
    expansion_pass.run()

    # determine a device to preload (compile) for
    device = None
    # Scan args for tensors and infer device.
    for arg in example_inputs:
        if isinstance(arg, torch.Tensor):
            tensor_device = arg.device
            if device is None:
                device = tensor_device
            else:
                if tensor_device != device:
                    raise RuntimeError(
                        f"Args must be on the same device: "
                        f"{tensor_device} vs {device}"
                    )

    # If there are no tensor inputs, try to get a device from node metadata.
    if device is None:
        for node in gm.graph.nodes:
            maybe_tensor = node.meta.get("val")
            device = (
                device
                if not isinstance(maybe_tensor, torch.Tensor)
                else maybe_tensor.device
            )
            if device is not None:
                break

    if device is None:
        raise RuntimeError("Could not infer a device for `iree_turbine` backend.")

    # Temporary workaround for issue #984. Remove the workaround once resolved.
    entry_point = "main" if len(example_inputs) == 0 else "main$async"

    # Create a Launchable from the MLIR source.
    launch = Launchable.jit_compile(str(module_op), entry_point=entry_point)

    launch.preload(device=device)

    call_func = lambda *args: launch(*[arg.data for arg in args], device=device)
    return make_boxed_func(call_func)


backend = aot_autograd(fw_compiler=_backend)
