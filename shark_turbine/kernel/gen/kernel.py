# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Custom op registeration for TK"""

import inspect

import torch

from typing import Callable, Any

from ..lang.kernel_buffer import is_kernel_buffer_meta_derived

from ..lang import (
    InputBuffer,
    OutputBuffer,
    Grid,
    IndexExpr,
)

from .thread import LaunchableThread

from ..compiler.ir import (
    SymbolRefAttr,
    ArrayAttr,
    flow_d,
    IrType,
)

from ...runtime.op_reg import (
    def_library,
    CustomOp,
    KernelBuilder,
    KernelSelection,
    TensorArg,
)

from .._support.tracing import AOTLaunchContext
from .._support.indexing import IndexingContext

TK_LIBRARY = def_library("tk")


__all__ = [
    "kernel",
]


def kernel(*symbolic_shape: IndexExpr):
    def decorator(f: Callable):
        # Convert all InputBuffer to inputs and OutputBuffers to outputs
        sig = inspect.signature(f)
        params = sig.parameters
        inputs: list[tuple[str, Any]] = []
        outputs: list[tuple[str, Any]] = []
        for arg_name, param in params.items():
            # TODO: Implement more input arguements.
            if not is_kernel_buffer_meta_derived(param.annotation):
                raise NotImplementedError(
                    "Only KernelBuffer is supported as input for now"
                )

            if param.annotation.usage == InputBuffer.usage:
                inputs.append((arg_name, param.annotation))
            elif param.annotation.usage == OutputBuffer.usage:
                outputs.append((arg_name, param.annotation))

        name_spec = f"kernel_{f.__name__}__@UNIQUE@"
        input_signature = ["Tensor " + name for name, _ in inputs]
        output_signature = ["Tensor " + name for name, _ in outputs]

        @CustomOp.register(library=TK_LIBRARY)
        class TKCustomOp(CustomOp):
            signature = (
                f"{name_spec}({', '.join(input_signature)}) -> "
                f"({', '.join(output_signature)})"
            )

            def select(self, ksel: KernelSelection):
                # Infer the result tensor based on the input tensor
                idxc = IndexingContext()

                i = 0
                for arg_name, arg in inputs:
                    if is_kernel_buffer_meta_derived(arg):
                        x = ksel.arg_tensor(i)
                        # We currently only do static dimensions.
                        # TODO: Support dynamic dimensions.
                        x.spec_dims = list(x.t.shape)
                        assert isinstance(x, TensorArg)
                        idxc.bind_shaped(arg_name, arg, list(x.t.shape))
                        i += 1
                    else:
                        raise NotImplementedError(
                            "Only KernelBuffer is supported as input for now"
                        )

                idxc.finalize()

                i = 0
                for _, arg in outputs:
                    if is_kernel_buffer_meta_derived(arg):
                        shape = arg.symbolic_shape
                        static_shape = [idxc.get_static_value(x) for x in shape]
                        x = torch.empty(*static_shape)
                        ksel.return_tensor(x)
                        # TODO: Support dynamic dimensions.
                        # Set spec_dims for output so that we can infer the
                        # type of the output tensor.
                        ksel.result_descs[i].spec_dims = list(x.shape)
                        i += 1
                    else:
                        raise NotImplementedError(
                            "Only KernelBuffer is supported as input for now"
                        )

            def generate(self, ksel: KernelSelection, kb: KernelBuilder):
                entrypoint = f"tk_{self.name}"
                # Create a flow.dispatch op to the kernel
                dispatch = SymbolRefAttr.get([entrypoint, entrypoint])
                entrypoints = ArrayAttr.get([dispatch])

                result_types = [
                    IrType.parse(x.mlir_type_asm) for x in ksel.result_descs
                ]

                out = flow_d.DispatchOp(
                    result_types, [], entrypoints, kb.arg_bindings, [], []
                )

                kb.yield_results(*out.results_)

                # Build the kernel as a stream executable.
                args = []
                for arg in ksel.arg_descs:
                    if isinstance(arg, TensorArg):
                        args.append(arg.t)
                    else:
                        raise NotImplementedError("Non TensorArg arg binding")

                for res in ksel.result_descs:
                    if isinstance(res, TensorArg):
                        args.append(res.t)
                    else:
                        raise NotImplementedError("Non TensorArg result binding")

                launchable = LaunchableThread(Grid[symbolic_shape], entrypoint, f)
                with AOTLaunchContext(kb.module_body.owner) as launch_ctx:
                    launch_ctx.launch(launchable, args, {})

        return TKCustomOp

    return decorator
