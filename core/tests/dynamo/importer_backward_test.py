# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from testutils import *


class ImportTests(unittest.TestCase):
    def testImportCustomLossModule(self):
        def foo(x, y):
            loss = ((0.5 * x - y) ** 2).mean()
            loss.backward()
            return loss

        opt_foo = torch.compile(foo, backend=create_backend())
        opt_foo(torch.randn(10), torch.randn(10, requires_grad=True))

    # TODO: using func.grad for backward test

    # TODO: MNIST Classifier using LeNet for backward test


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
