# Copyright 2023 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch

import logging
import unittest

from iree.turbine.aot import decompositions


class DecompTest(unittest.TestCase):
    def testDefault(self):
        table = decompositions.current_aot_decompositions()
        self.assertTrue(table)

    def testMseLossBackward(self):
        class Backward(torch.nn.Module):
            def forward(self, grad_output, inputs, target, reduction):
                return torch.ops.aten.mse_loss_backward(
                    grad_output, inputs, target, reduction
                )

        inputs = torch.linspace(-2, 2, 6).reshape(2, 3)
        target = inputs.flip(1) + 0.5
        # ATen uses a reduction index, the functional API uses its name.
        for reduction, name in enumerate(("none", "mean", "sum")):
            with self.subTest(reduction=name):
                grad_output = (
                    torch.linspace(-1, 1, 6).reshape(2, 3)
                    if name == "none"
                    else torch.tensor(0.75)
                )
                args = (grad_output, inputs, target, reduction)
                exported = torch.export.export(Backward(), args)
                exported = exported.run_decompositions(
                    decompositions.current_aot_decompositions()
                )
                self.assertNotIn(
                    torch.ops.aten.mse_loss_backward.default,
                    [node.target for node in exported.graph.nodes],
                )
                differentiable = inputs.clone().requires_grad_()
                loss = torch.nn.functional.mse_loss(
                    differentiable, target, reduction=name
                )
                (expected,) = torch.autograd.grad(loss, differentiable, grad_output)
                torch.testing.assert_close(exported.module()(*args), expected)

    def testExtendToEmpty(self):
        with decompositions.extend_aot_decompositions(from_current=False) as t:
            self.assertFalse(t, msg=f"{t}")
            current_table = decompositions.current_aot_decompositions()
            self.assertFalse(current_table, msg=f"{current_table}")

    def testNestedExtend(self):
        initial_table = decompositions.current_aot_decompositions()
        with decompositions.extend_aot_decompositions(from_current=False) as empty_t:
            with decompositions.extend_aot_decompositions(
                add_ops=[
                    torch.ops.aten.masked_fill.Tensor,
                    torch.ops.aten.masked_fill.Scalar,
                ]
            ):
                current_table = decompositions.current_aot_decompositions()
                self.assertEqual(2, len(current_table), msg=f"{current_table}")
                with decompositions.extend_aot_decompositions(
                    remove_ops=[
                        torch.ops.aten.masked_fill.Tensor,
                    ]
                ):
                    current_table = decompositions.current_aot_decompositions()
                    self.assertEqual(1, len(current_table), msg=f"{current_table}")
        current_table = decompositions.current_aot_decompositions()
        self.assertDictEqual(current_table, initial_table)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
