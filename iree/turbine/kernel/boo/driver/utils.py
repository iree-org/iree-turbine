# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from pathlib import Path
import argparse

from iree.turbine.kernel.boo.op_exports.registry import BooOpRegistry


def get_timing_parser() -> argparse.ArgumentParser:
    """This parser separates out timing-specific args from MIOpen commands."""
    timing_parser = argparse.ArgumentParser()
    timing_parser.add_argument(
        "--time", "-t", type=int, help="Enable timing", default=1
    )
    timing_parser.add_argument(
        "--iter",
        type=int,
        help="Exact number of iterations (disables auto-adjustment; "
        "shorthand for --min-iter X --min-time 0)",
        default=None,
    )
    timing_parser.add_argument(
        "--min-iter",
        type=int,
        help="Minimum number of iterations when auto-adjusting (default: 100)",
        default=100,
    )
    timing_parser.add_argument(
        "--min-time",
        type=float,
        help="Minimum benchmark duration in seconds (default: 3.0)",
        default=3.0,
    )
    return timing_parser


def resolve_timing_args(timing_args: argparse.Namespace) -> None:
    """Resolve --iter shorthand into --min-iter and --min-time 0."""
    if timing_args.iter is not None:
        timing_args.min_iter = timing_args.iter
        timing_args.min_time = 0.0


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
