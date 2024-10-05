# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Debug flags and settings."""

from typing import Optional
from dataclasses import dataclass
import logging
import re
import os

__all__ = [
    "flags",
    "NDEBUG",
]

# We use the native logging vs our .logging setup because our logging depends
# on this module. It will spew to stderr with issues.
logger = logging.getLogger("turbine.bootstrap")

# The TURBINE_DEBUG environment variable is a comma separated list of settings
# of the form "(-)?name[=value]".
# Available settings:
#   log_level: A log level name to enable.
#   asserts: Whether to enable all assertions (defaults to enabled).
FLAGS_ENV_NAME = "TURBINE_DEBUG"
SETTING_PART_PATTERN = re.compile(r"""^([\\+\\-])?([^=]+)(=(.*))?$""")

# Some settings can also be set in dedicated environment variables. Those are
# mapped here.
ENV_SETTINGS_MAP = {
    "TURBINE_LOG_LEVEL": "log_level",
}

# Whether debug/prolific assertions are disabled.
NDEBUG: bool = False


@dataclass
class DebugFlags:
    log_level: int = logging.WARNING
    asserts: bool = False
    runtime_trace_dir: Optional[str] = None

    def set(self, part: str):
        m = re.match(SETTING_PART_PATTERN, part)
        if not m:
            logger.warn("Syntax error in %s flag: '%s'", FLAGS_ENV_NAME, part)
            return
        name = m.group(2)
        value = m.group(4)
        if value:
            logical_sense = value.upper() not in ["FALSE", "OFF", "0"]
        else:
            logical_sense = m.group(1) != "-"

        if name == "log_level":
            log_level_mapping = logging.getLevelNamesMapping()
            try:
                self.log_level = log_level_mapping[value.upper()]
            except KeyError:
                logger.warn("Log level '%s' unknown (ignored)", value)
        elif name == "asserts":
            self.asserts = logical_sense
            global NDEBUG
            NDEBUG = not logical_sense
        elif name == "runtime_trace_dir":
            self.runtime_trace_dir = value
        else:
            logger.warn("Unrecognized %s flag: '%s'", FLAGS_ENV_NAME, name)

    @staticmethod
    def parse(settings: str) -> "DebugFlags":
        new_flags = DebugFlags()
        parts = settings.split(",")
        for part in parts:
            part = part.strip()
            if not part:
                continue
            new_flags.set(part)
        return new_flags

    @staticmethod
    def parse_from_env() -> "DebugFlags":
        settings = os.getenv(FLAGS_ENV_NAME)
        if settings is None:
            new_flags = DebugFlags()
        else:
            new_flags = DebugFlags.parse(settings)
        for env_name, setting_name in ENV_SETTINGS_MAP.items():
            env_value = os.getenv(env_name)
            if env_value is not None:
                new_flags.set(f"{setting_name}={env_value}")
        logger.debug("Parsed debug flags from env %s: %r", FLAGS_ENV_NAME, new_flags)
        return new_flags


flags = DebugFlags.parse_from_env()
