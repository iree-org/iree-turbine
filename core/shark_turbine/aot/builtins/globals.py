# Copyright 2023 Nod Labs, Inc
# Portions Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Any, Callable, Optional

import torch.nn as nn

from ..support.procedural import (
    AbstractTypedef,
    Abstractifiable,
    GlobalsDef,
    TreeAbstractifiable,
    abstractify_single_value,
)

from ..support.ir_utils import (
    NameMapCallback,
    GlobalAttributes,
)

from ..support.utils import (
    TreeSpec,
    tree_flatten,
    tree_map,
)


class export_global(GlobalsDef, Abstractifiable):
    """Exports a single global into a CompiledModule."""

    __slots__ = ["_name", "_value", "_schema"]

    def __init__(
        self,
        value: Any,
        *,
        name: str = "global",
        mutable: Optional[bool] = None,
        external: Optional[bool] = None,
        external_scope: Optional[str] = None,
        name_mapper: Optional[NameMapCallback] = None,
        uninitialized: Optional[bool] = None,
        attrs: Optional[GlobalAttributes] = None,
    ):
        if attrs is None:
            attrs = GlobalAttributes(
                mutable=mutable,
                external=external,
                external_scope=external_scope,
                name_mapper=name_mapper,
                uninitialized=uninitialized,
            )
        super().__init__(attrs)
        self._name = name
        self._value = value
        _, self._schema = tree_flatten(self._value)

    def items(self):
        yield (self._name, self._value)

    def schema(self) -> TreeSpec:
        return self._schema

    def abstractify(self) -> AbstractTypedef:
        return abstractify_single_value(self._value)


class export_global_tree(GlobalsDef, Abstractifiable):
    """Exports a tree of globals into a CompiledModule."""

    def __init__(
        self,
        tree,
        *,
        mutable: Optional[bool] = None,
        external: Optional[bool] = None,
        external_scope: Optional[str] = None,
        name_mapper: Optional[NameMapCallback] = None,
        uninitialized: Optional[bool] = None,
        attrs: Optional[GlobalAttributes] = None,
    ):
        if attrs is None:
            attrs = GlobalAttributes(
                mutable=mutable,
                external=external,
                external_scope=external_scope,
                name_mapper=name_mapper,
                uninitialized=uninitialized,
            )
        super().__init__(attrs)
        self._tree = tree
        self._items, self._schema = tree_flatten(tree)
        self._names, _ = tree_flatten(_transform_tree_to_names("", tree))
        assert len(self._items) == len(
            self._names
        ), f"Name and value tree are different sizes: {len(self._items)} != {len(self._names)}"

    def items(self):
        for name, value in zip(self._names, self._items):
            yield name, value

    def schema(self) -> TreeSpec:
        return self._schema

    def abstractify(self) -> AbstractTypedef:
        return tree_map(abstractify_single_value, self._tree)


class export_parameters(GlobalsDef, TreeAbstractifiable):
    """Exports parameters from an nn.Module.

    These are exposed to procedural programs as a dictionary of param/values.
    """

    __slots__ = [
        "_param_list",
        "_schema",
        "_tree",
    ]

    def __init__(
        self,
        nn_module: nn.Module,
        *,
        mutable: Optional[bool] = None,
        external: Optional[bool] = None,
        external_scope: Optional[str] = None,
        name_mapper: Optional[NameMapCallback] = None,
        uninitialized: Optional[bool] = None,
        attrs: Optional[GlobalAttributes] = None,
    ):
        if attrs is None:
            attrs = GlobalAttributes(
                mutable=mutable,
                external=external,
                external_scope=external_scope,
                name_mapper=name_mapper,
                uninitialized=uninitialized,
            )
        super().__init__(attrs)
        self._param_list = list(nn_module.named_parameters())
        self._tree = dict(self._param_list)
        _, self._schema = tree_flatten(self._tree)

    def items(self):
        for name, value in self._param_list:
            yield (name, value)

    def schema(self) -> TreeSpec:
        return self._schema

    def abstractify_tree(self):
        return tree_map(abstractify_single_value, self._tree)

    def __getitem__(self, key):
        return self._tree[key]

    def __repr__(self):
        names = [name for name, _ in self._param_list]
        return f"<export_parameters {', '.join(names)}>"


def _transform_tree_to_names(prefix: str, tree):
    """Produces a topologically similar tree but where each value is a fully qualified name."""
    join = lambda key: f"{prefix}.{key}" if prefix else key
    # No need to check for cycles as pytree already did something with it and
    # validates.
    if isinstance(tree, dict):
        return tree.__class__(
            (k, _transform_tree_to_names(join(k), v)) for k, v in tree.items()
        )
    elif isinstance(tree, (list, tuple)):
        return tree.__class__(
            _transform_tree_to_names(join(str(index)), v)
            for index, v in enumerate(tree)
        )
    else:
        return prefix
