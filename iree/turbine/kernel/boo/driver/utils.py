# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from pathlib import Path

from iree.turbine.kernel.boo.op_exports.registry import BooOpRegistry


def load_commands(commands_file: str) -> list[str]:
    """Loads commands of a given kind from a text file.

    Only keep commands that are known to be parseable.
    """
    # try an absolute path
    path = Path(commands_file)
    # if the path doesn't point anywhere, try relative to cwd and this file.
    if not path.is_file():
        path = Path.cwd() / commands_file
    if not path.is_file():
        path = Path(__file__) / commands_file
    if not path.is_file():
        raise ValueError(
            f"'commands-file' specification, '{commands_file}', cannot be found."
        )

    commands = [
        c
        for c in path.read_text().splitlines()
        if BooOpRegistry.find_key_from_command(c) is not None
    ]
    return commands
