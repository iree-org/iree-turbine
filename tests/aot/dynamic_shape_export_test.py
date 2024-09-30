import torch

import pytest

from shark_turbine.aot import *


@pytest.mark.parametrize(
    "import_symbolic_shape_expressions",
    [
        True,
        False,
    ],
)
def test_exported_program_dynamic_shapes(import_symbolic_shape_expressions):
    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()

            self.branch1 = torch.nn.Sequential(torch.nn.Linear(64, 32), torch.nn.ReLU())
            self.branch2 = torch.nn.Sequential(
                torch.nn.Linear(128, 64), torch.nn.ReLU()
            )
            self.buffer = torch.ones(32)

        def forward(self, x1, x2):
            out1 = self.branch1(x1)
            out2 = self.branch2(x2)
            return (out1 + self.buffer, out2)

    example_args = (torch.randn(32, 64), torch.randn(32, 128))

    # Create a dynamic batch size
    batch = torch.export.Dim("batch")
    # Specify that the first dimension of each input is that batch size
    dynamic_shapes = {"x1": {0: batch}, "x2": {0: batch}}

    output = export(
        M(),
        args=example_args,
        dynamic_shapes=dynamic_shapes,
        import_symbolic_shape_expressions=import_symbolic_shape_expressions,
    )
    output.print_readable()
    asm = str(output.mlir_module)

    if import_symbolic_shape_expressions:
        assert "bind_symbolic_shape" in asm
    else:
        assert "bind_symbolic_shape" not in asm
