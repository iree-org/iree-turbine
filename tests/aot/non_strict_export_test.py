import logging
import unittest
from torch import nn
import torch

from shark_turbine.aot import *

logger = logging.getLogger(__file__)


class NonStrictExportTest(unittest.TestCase):
    def testNonStrictExport(self):
        mdl = SimpleParams()
        random_input = torch.randn((20))
        exported = export(mdl, args=(random_input,), strict_export=False)
        mlir_str = str(exported.mlir_module)
        self.assertIn("func.func", mlir_str)

    def testStrictExportFailure(self):
        mdl = SimpleParams()
        random_input = torch.randn((20))
        with self.assertRaises(Exception):
            export(mdl, args=(random_input,), strict_export=True)


class SimpleParams(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Linear(20, 30)

    def forward(self, x):
        logger.warning(f"This will fail without disabling strict export mode")
        return self.classifier(x)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
