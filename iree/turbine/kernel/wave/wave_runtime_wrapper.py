# Copyright 2025 The IREE Authors
# Licensed under the Apache License 2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import sys
import subprocess
import hashlib
from pathlib import Path

# Global variable to store the imported wave_runtime module
_wave_runtime_module = None


def _get_wave_runtime_source_dir() -> Path:
    current_file = Path(__file__)
    source_dir = current_file.parent / "runtime"
    return source_dir


def _get_cpp_file_hash() -> str:
    """Get the hash of the C++ source file for versioning."""
    cpp_file = _get_wave_runtime_source_dir() / "runtime.cpp"
    if not cpp_file.exists():
        return "unknown"

    with open(cpp_file, "rb") as f:
        content = f.read()
        return hashlib.sha256(content).hexdigest()[:8]  # Use first 8 chars for brevity


def _get_wave_cache_dir() -> Path:
    """Get the wave cache directory from environment variable or use default."""
    from .cache import get_cache_base_dir

    cache_dir = get_cache_base_dir()
    cpp_hash = _get_cpp_file_hash()
    wave_runtime_dir = cache_dir / f"wave_runtime_{cpp_hash}"
    return wave_runtime_dir


def _find_wave_runtime_module(build_lib_dir: Path) -> Path | None:
    """Find the wave_runtime module file in the build_lib directory."""
    if not build_lib_dir.exists():
        return None

    for file_path in build_lib_dir.rglob("*"):
        if file_path.name.startswith("wave_runtime"):
            return file_path
    return None


def _print_error_with_output(message: str, e: Exception, stdout=None, stderr=None):
    """Print error message with optional stdout/stderr output."""
    print(f"{message}: {e}")
    if stdout:
        print(f"Command output: {stdout.decode() if stdout else 'No stdout'}")
    if stderr:
        print(f"Error output: {stderr.decode() if stderr else 'No stderr'}")


def _build_wave_runtime_in_cache(source_dir: Path, cache_dir: Path) -> Path:
    """Build wave_runtime in the cache directory and return the path to the built module."""
    try:
        print(f"Building wave_runtime from {source_dir} in cache directory {cache_dir}")

        # Create cache directory if it doesn't exist
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Build the extension in the cache directory
        build_cmd = [
            sys.executable,
            "setup.py",
            "build_ext",
            "--build-temp",
            str(cache_dir / "build_temp"),
            "--build-lib",
            str(cache_dir / "build_lib"),
        ]

        subprocess.run(
            build_cmd,
            cwd=source_dir,
            check=True,
            capture_output=True,
        )

        # Find the built module file
        build_lib_dir = cache_dir / "build_lib"
        module_path = _find_wave_runtime_module(build_lib_dir)

        if module_path is None:
            raise FileNotFoundError("Could not find built wave_runtime module")

        return module_path

    except subprocess.CalledProcessError as e:
        _print_error_with_output("Failed to build wave_runtime", e, e.stdout, e.stderr)
        raise
    except Exception as e:
        _print_error_with_output("Error building wave_runtime", e)
        raise


def _load_wave_runtime_from_path(module_path: Path):
    """Load the wave_runtime module from a specific file path."""
    try:
        # Add the parent directory to sys.path temporarily
        module_dir = module_path.parent
        original_path = sys.path.copy()
        sys.path.insert(0, str(module_dir))

        # Import the module
        import importlib.util

        spec = importlib.util.spec_from_file_location("wave_runtime", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Restore original sys.path
        sys.path = original_path

        return module
    except Exception as e:
        _print_error_with_output(f"Failed to load wave_runtime from {module_path}", e)
        raise


def _try_import_wave_runtime():
    """Try to import wave_runtime from cache first, then globally if not found."""
    cache_dir = _get_wave_cache_dir()
    build_lib_dir = cache_dir / "build_lib"

    # First try to load from cache if it exists
    module_path = _find_wave_runtime_module(build_lib_dir)
    if module_path is not None:
        try:
            return _load_wave_runtime_from_path(module_path)
        except Exception as e:
            _print_error_with_output("Failed to load wave_runtime from cache", e)

    # Fall back to global import
    try:
        return __import__("wave_runtime")
    except ImportError:
        return None


def _build_and_load_wave_runtime():
    """Build wave_runtime in cache directory and load it, returning the loaded module."""
    source_dir = _get_wave_runtime_source_dir()
    cache_dir = _get_wave_cache_dir()

    # Build wave_runtime in cache directory
    module_path = _build_wave_runtime_in_cache(source_dir, cache_dir)

    # Load the built module
    return _load_wave_runtime_from_path(module_path)


def _ensure_wave_runtime_built():
    """Ensure wave_runtime is built and available."""
    global _wave_runtime_module

    if _wave_runtime_module is not None:
        return _wave_runtime_module

    # First try to import normally (in case it's already installed)
    _wave_runtime_module = _try_import_wave_runtime()
    if _wave_runtime_module is not None:
        return _wave_runtime_module

    # Build and load from cache directory
    _wave_runtime_module = _build_and_load_wave_runtime()
    return _wave_runtime_module


def get_wave_runtime():
    """Get the wave_runtime module, building it if necessary."""
    return _ensure_wave_runtime_built()


# Create a module-like object that forwards all attributes to the actual wave_runtime
class WaveRuntimeProxy:
    def __getattr__(self, name):
        module = get_wave_runtime()
        return getattr(module, name)

    def __dir__(self):
        module = get_wave_runtime()
        return dir(module)


# Create the proxy instance
wave_runtime = WaveRuntimeProxy()
