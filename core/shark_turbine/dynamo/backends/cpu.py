# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import functools
import sys

from ...runtime.device import (
    DeviceState,
)

from ..executor import (
    SpecializedExecutable,
)

from iree.compiler.api import (
    Invocation,
    Session,
    Source,
    Output,
)

from iree.compiler.ir import (
    Context,
)
from iree.compiler.passmanager import (
    PassManager,
)

from iree.runtime import (
    VmModule,
)

from ...importers.fx_importer import FxImporter

import torch
from torch._dynamo.backends.common import aot_autograd
from ..passes import turbine_cpu_pass_pipeline

DEFAULT_COMPILER_FLAGS = ("--iree-input-type=torch",)


def _base_backend(gm: torch.fx.GraphModule, example_inputs):
    # Set up the session, context and invocation.
    # Note that we do this on one in-memory module in a few phases:
    #  1. Build it from the FX graph.
    #  2. Run torch MLIR passes to lower it to a suitable form for
    #     input.
    #  3. Run IREE's main compiler.
    #  4. Output to an mmap buffer.
    session = Session()
    session.set_flags(*DEFAULT_COMPILER_FLAGS)
    session.set_flags("--iree-hal-target-backends=llvm-cpu")
    context = session.context
    importer = FxImporter(context=context)
    module = importer.module
    inv = session.invocation()
    # TODO: Should capture diagnostics.
    inv.enable_console_diagnostics()
    inv.import_module(module.operation)

    # Apply decompositions.
    gm = turbine_cpu_pass_pipeline(gm, example_inputs)

    # Import phase.
    importer.import_graph_module(gm)
    print(module, file=sys.stderr)
    with context:
        pm = PassManager.parse("builtin.module(torch-to-iree)")
        pm.run(module.operation)
    print(module, file=sys.stderr)

    # IREE compilation phase.
    inv.execute()

    # Output phase.
    output = Output.open_membuffer()
    inv.output_vm_bytecode(output)

    # Set up for runtime.
    device_state = _get_device_state()
    # TODO: Switch to wrap_buffer once https://github.com/openxla/iree/issues/14926
    # is fixed.
    # vmfb_module = VmModule.wrap_buffer(
    #     device_state.instance,
    #     output.map_memory(),
    #     destroy_callback=output.close,
    # )
    vmfb_module = VmModule.copy_buffer(
        device_state.instance,
        output.map_memory(),
    )
    output.close()

    return SpecializedExecutable(vmfb_module, device_state)


backend = aot_autograd(fw_compiler=_base_backend)


# IREE runtime globals. For the CPU right now, there is no device selection,
# so it is easy.
@functools.lru_cache(maxsize=None)
def _get_device_state() -> DeviceState:
    return DeviceState(driver="local-task")
