# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import unittest

import torch

import shark_turbine.kernel as tk
import shark_turbine.kernel.lang as tkl

M = tk.lang.sym.M
K = tk.lang.sym.K


class Test(unittest.TestCase):
    def testIotaEager(self):
        @tk.gen.thread(M)
        def iota_kernel(out: tk.lang.OutputBuffer[M, tkl.index]):
            i = tk.lang.program_id(0)
            out[i] = i

        out = torch.empty(8, dtype=torch.int32)
        with tk.gen.TestLaunchContext():
            iota_kernel(out)
        print(out)

    def testIotaFx(self):
        @tk.gen.thread(M)
        def iota_kernel(out: tk.lang.KernelBuffer[M, tkl.index]):
            i = tk.lang.program_id(0)
            out[i] = i

        print(iota_kernel._trace().region_graph)
        # Prints:
        # .graph():
        #     %out : shark_turbine.kernel.lang.types.KernelBuffer [num_users=1] = placeholder[target=out]
        #     %program_id : [num_users=1] = call_function[target=shark_turbine.kernel.lang.prims.program_id](args = (0,), kwargs = {})
        #     %_global_buffer_setitem : [num_users=0] = call_function[target=shark_turbine.kernel._support.tracing._global_buffer_setitem](args = (%out, %program_id, %program_id), kwargs = {})
        #     return None

    def testSoftmax(self):
        @tk.gen.thread(M)
        def softmax_kernel(
            input: tk.lang.InputBuffer[M, K, tkl.f32],
            output: tk.lang.OutputBuffer[M, K, tkl.f32],
        ):
            row_index = tk.lang.program_id(0)
            input_row = input[row_index, :]
            numerator = torch.exp(input_row - tk.lang.max(input_row))
            output_row = numerator / torch.sum(numerator)
            output[row_index, :] = output_row
            # Some debugging info if in debug mode and processing the first row.
            if tk.DEBUG and row_index == 0:
                print(f"*** Input: {input}")
                print(f"*** Output: {output}")
                print(
                    f"*** Input Row[{row_index}]: {type(output_row).__name__}({input_row.shape})"
                )
                print(
                    f"*** Output Row: {type(output_row).__name__}({output_row.shape})"
                )

        def softmax(x):
            y = torch.empty_like(x)
            softmax_kernel[x.shape[0]](x, y)
            return y

        input = torch.rand((128, 64))
        # generated = softmax(input)
        # actual = torch.softmax(input, -1)
        # torch.testing.assert_close(generated, actual)
        print(softmax_kernel._trace().region_graph)
        # Prints:
        # graph():
        #     %input_1 : shark_turbine.kernel.lang.types.KernelBuffer [num_users=1] = placeholder[target=input]
        #     %output : shark_turbine.kernel.lang.types.KernelBuffer [num_users=1] = placeholder[target=output]
        #     %program_id : [num_users=1] = call_function[target=shark_turbine.kernel.lang.prims.program_id](args = (0,), kwargs = {})
        #     %getitem : [num_users=2] = call_function[target=operator.getitem](args = (%input_1, (%program_id, slice(None, None, None))), kwargs = {})
        #     %max_1 : [num_users=1] = call_function[target=torch.max](args = (%getitem,), kwargs = {})
        #     %sub : [num_users=1] = call_function[target=operator.sub](args = (%getitem, %max_1), kwargs = {})
        #     %exp : [num_users=2] = call_function[target=torch.exp](args = (%sub,), kwargs = {})
        #     %sum_1 : [num_users=1] = call_function[target=torch.sum](args = (%exp,), kwargs = {})
        #     %truediv : [num_users=1] = call_function[target=operator.truediv](args = (%exp, %sum_1), kwargs = {})
        #     %program_id_1 : [num_users=1] = call_function[target=shark_turbine.kernel.lang.prims.program_id](args = (0,), kwargs = {})
        #     %_kernel_buffer_setitem : [num_users=0] = call_function[target=shark_turbine.kernel._support.tracing._kernel_buffer_setitem](args = (%output, (%program_id_1, slice(None, None, None)), %truediv), kwargs = {})
        #     return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
