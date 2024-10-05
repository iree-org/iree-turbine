# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import hashlib
import os
from pathlib import Path
import logging

from ..support.debugging import flags
from ..support.logging import get_logger, DefaultFormatter

logger = get_logger("turbine.runtime")


class RuntimeTracer:
    """Supports fine grained tracing of runtime interactions.

    The default implementation no-ops.
    """

    __slots__ = ["enabled"]

    def __init__(self):
        self.enabled: bool = False

    def save_jit_kernel_artifacts(
        self, *, cache_key: str, module_asm: bytes, binary: memoryview
    ) -> str:
        return cache_key

    def info(self, msg, *args, **kwargs):
        ...

    def error(self, msg, *args, **kwargs):
        ...

    def exception(self, msg, *args, **kwargs):
        ...

    def log_structured(self, *, tag: str, msg: str, columns: list):
        ...


class DirectoryTracer(RuntimeTracer):
    __slots__ = [
        "dir",
        "logger",
    ]

    def __init__(self, dir: Path):
        self.dir = dir
        self.enabled = True
        # Configure a root logger that outputs what we want.
        trace_logger = self.logger = logging.getLogger("turbine.runtime.tracer")
        log_file = dir / "runtime.log"
        trace_logger.setLevel(logging.DEBUG)
        handler = logging.FileHandler(log_file)
        handler.setFormatter(DefaultFormatter())
        trace_logger.addHandler(handler)
        trace_logger.propagate = False
        logger.info(f"Set up turbine runtime tracing to %s", log_file)
        trace_logger.info("Started process %d", os.getpid())

    def save_jit_kernel_artifacts(
        self, *, cache_key: str, module_asm: bytes, binary: memoryview
    ) -> str:
        hasher = hashlib.sha1(cache_key.encode(), usedforsecurity=False)
        tracing_key = hasher.digest().hex()
        try:
            with open(self.dir / f"{tracing_key}.mlir", "wb") as f:
                f.write(module_asm)
            with open(self.dir / f"{tracing_key}.vmfb", "wb") as f:
                f.write(binary)
        except IOError:
            self.logger.exception(f"Error saving artifact for {tracing_key}")
        finally:
            self.logger.info(f"Saved artifacts for {tracing_key}")
        return tracing_key

    def info(self, msg, *args, **kwargs):
        self.logger.info(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self.logger.error(msg, *args, **kwargs)

    def exception(self, msg, *args, **kwargs):
        self.logger.exception(msg, *args, **kwargs, stacklevel=2)

    def log_structured(self, *, tag: str, msg: str, columns: list):
        columns_joined = "\t".join(str(c) for c in columns)
        self.logger.info("%s\n::%s\t%s", msg, tag, columns_joined)


# Determine whether configured to do real tracing.
def _setup_default_tracer() -> RuntimeTracer:
    if flags.runtime_trace_dir:
        try:
            trace_dir = Path(flags.runtime_trace_dir)
            trace_dir.mkdir(parents=True, exist_ok=True)
            return DirectoryTracer(trace_dir)
        except IOError:
            logger.exception("Error configuring runtime tracing to: %s", trace_dir)
            return RuntimeTracer()

    return RuntimeTracer()


tracer: RuntimeTracer = _setup_default_tracer()
