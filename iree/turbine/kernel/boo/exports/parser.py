# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from abc import ABC, abstractmethod
import argparse

from .signature import OpSignature


class OpCLIParser(ABC):
    @staticmethod
    @abstractmethod
    def get_signature(args: argparse.Namespace) -> OpSignature:
        """Constructs a signature from the given command line arguments."""
        ...

    @staticmethod
    @abstractmethod
    def get_miopen_parser() -> argparse.ArgumentParser:
        """Returns a pre-configured argument parser with MIOpen-compatible options."""
        ...

    @classmethod
    def command_to_signature(
        cls, command: str, ignore_unhandled_args: bool = False
    ) -> OpSignature:
        """Convert a textual command to signature by parsing it."""
        parser = cls.get_miopen_parser()
        if ignore_unhandled_args:
            args, _ = parser.parse_known_args(command.split())
        else:
            args = parser.parse_args(command.split())
        return cls.get_signature(args)

    @classmethod
    @abstractmethod
    def get_op_name(cls) -> str:
        """Returns the base name of the operation."""
        ...
