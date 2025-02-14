# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import iree.runtime as rt
import logging
import numpy as np
import torch
import torch.nn as nn
import unittest

import iree.turbine.aot as aot


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) model class.
    Defines a neural network with four linear layers and sigmoid activations.
    """

    def __init__(self) -> object:
        super().__init__()
        # Define model layers
        self.layer0 = nn.Linear(8, 8, bias=True)
        self.layer1 = nn.Linear(8, 4, bias=True)
        self.layer2 = nn.Linear(4, 2, bias=True)
        self.layer3 = nn.Linear(2, 2, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MLP model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after forward pass.
        """
        x = self.layer0(x)
        x = torch.sigmoid(x)
        x = self.layer1(x)
        x = torch.sigmoid(x)
        x = self.layer2(x)
        x = torch.sigmoid(x)
        x = self.layer3(x)
        return x


model = MLP()
example_x = torch.empty(97, 8, dtype=torch.float32)
exported = aot.export(model, example_x)
exported.print_readable()
compiled_binary = exported.compile(save_to=None)


def run_inference() -> np.ndarray:
    """
    Runs inference on the compiled model.

    Returns:
        np.ndarray: The result of inference as a NumPy array.
    """
    config = rt.Config("local-task")
    vmm = rt.load_vm_module(
        rt.VmModule.wrap_buffer(config.vm_instance, compiled_binary.map_memory()),
        config,
    )
    x = np.random.rand(97, 8).astype(np.float32)
    y = vmm.main(x)
    logger.debug(f"Inference result: {y.to_host()}")
    return y.to_host()

    return y.to_host()


class ModelTest(unittest.TestCase):
    def test_mlp_export_simple(self) -> None:
        output = run_inference()

        self.assertIsNotNone(output, "inference output should not be None")
        self.assertEqual(
            output.shape, (97, 2), "output shape doesn't match the expected (97, 2)"
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    # Run unit tests
    unittest.main()
