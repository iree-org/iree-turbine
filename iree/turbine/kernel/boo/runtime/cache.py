# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import shutil

from pathlib import Path
from typing import Union, OrderedDict

from ....runtime import Launchable

__all__ = [
    "set_cache_dir",
    "clear_cache",
    "is_cache_enabled",
    "toggle_cache_on",
    "LaunchableRuntimeCache",
]

## file cache environment variables

_default_cache_base_dir = Path.home() / ".cache" / "turbine_kernels" / "boo"
BOO_CACHE_DIR = Path(os.environ.get("BOO_CACHE_DIR", _default_cache_base_dir))
BOO_CACHE_ON = int(os.environ.get("BOO_CACHE_ON", 1))


def set_cache_dir(cache_dir: Union[Path, str, None] = None) -> Path:
    global BOO_CACHE_DIR
    if cache_dir:
        BOO_CACHE_DIR = Path(cache_dir)
    return BOO_CACHE_DIR


def clear_cache():
    if not BOO_CACHE_DIR.is_dir():
        return
    shutil.rmtree(BOO_CACHE_DIR)


def is_cache_enabled() -> bool:
    return bool(BOO_CACHE_ON)


def toggle_cache_on(enabled: int):
    global BOO_CACHE_ON
    if enabled in {0, 1}:
        BOO_CACHE_ON = enabled
        return
    raise ValueError(f"expected `enabled` to be either 0 or 1, got {enabled}")


## runtime launchable cache


class LaunchableRuntimeCache:
    def __init__(self, cache_limit: int | None = None):
        self.cache_limit = cache_limit
        self.session_cache: OrderedDict[str, Launchable] = OrderedDict()

    def _add_to_session_cache(self, key: str, launchable: Launchable):
        self.session_cache[key] = launchable
        self.session_cache.move_to_end(key)
        if (
            self.cache_limit is not None
            and len(self.session_cache.keys()) > self.cache_limit
        ):
            self.session_cache.popitem(last=False)

    def _get(self, key: str) -> Launchable | None:
        return self.session_cache.get(key, None)

    @staticmethod
    def add(key: str, launchable: Launchable):
        global _launchable_cache
        if "_launchable_cache" not in globals():
            _launchable_cache = LaunchableRuntimeCache()
        _launchable_cache._add_to_session_cache(key, launchable)

    @staticmethod
    def get(key: str) -> Launchable | None:
        global _launchable_cache
        if "_launchable_cache" not in globals():
            _launchable_cache = LaunchableRuntimeCache()
        return _launchable_cache._get(key)

    @staticmethod
    def get_launchable_cache():
        global _launchable_cache
        if "_launchable_cache" in globals():
            return _launchable_cache
        _launchable_cache = LaunchableRuntimeCache()
        return _launchable_cache

    @staticmethod
    def clear():
        global _launchable_cache
        if not "_launchable_cache" in globals():
            return
        _launchable_cache.session_cache.clear()

    @staticmethod
    def set_cache_limit(new_cache_limit: int | None):
        global _launchable_cache
        if "_launchable_cache" in globals():
            _launchable_cache.cache_limit = new_cache_limit
            return
        _launchable_cache = LaunchableRuntimeCache(new_cache_limit)
