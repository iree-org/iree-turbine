# defines the kernel manager, which should allow registering kernel libraries, each with their own cache_mangement
from pathlib import Path
from typing import Union, Optional, Dict

import torch
from ..dynamo.backends.base import backend_generator
from .bag.namespaces import auto_op


class Kernel:
    """Device-specific executable. I want this to be 1-1 equivalent to a pre-compiled dispatch."""


class SpecializedTemplate:
    """Maybe this is the same thing as KernelSelection, but I want this to be 1-1 equivalent to device-agnostic MLIR."""


class KernelTemplate:
    """should be similar to a CustomOp. I.e. uses uses something like KernelBuilder and KernelSelection"""


class KernelNamespace:
    __slots__ = [
        "name",
        "cache_manager",
        "template_registry",
    ]

    def find_template(self, op_name: str) -> KernelTemplate:
        """search template_registry for op_name in template_registry"""
        ...


class KernelManager:
    def __init__(
        self,
        options: Optional[Dict] = None,
    ):
        self.options = options
        self.namespaces = {"auto_op": auto_op.AutoOpNamespace}

    def auto_kernel(self, fn, save_kernel: bool):
        """If save_kernel, this should register fn with namespace 'auto_ops' and store:
        1. template (cached fx graph module?),
        2. specialized templates (mlir),
        3. kernels generated (vmfb)?
        """
        if not save_kernel:
            return torch.compile(fn, backend_generator(**self.options))
        raise NotImplementedError("saving kernels to a file is not yet implemented")

    def load(namespace: str):
        """should enable using a namespace with the kernel manager. Each namespace should manage its kernels with a cache manager"""
        ...

    def set_cache_dir(namespace: str):
        """should set a cache directory for the specified kernel namespace"""
        ...

    def find_template_op(namespace: str, op_name: str):
        """should find a custom op within the namespace (not specialized)"""
        ...

    def load_kernel_from_file(namespace: str, path):
        """should call into cache_manager from namespace to load a specific kernel from a file."""
