# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""This pass will rename any #stream.parameter.named<> attributes on globals.

It can either be used as-is or by sub-classing (i.e. in a model specific
subclass that renames from A->B, etc).

By default, no attributes are touched unless:

* rename_map= has an exact match
* rename_callback= returns a change
"""

from typing import Callable, Dict, List, Optional, Tuple, Union

import re

from iree.compiler.ir import (
    Attribute,
    Operation,
    StringAttr,
)

from ..rewriter import *
from iree.compiler.ir import Context

ScopedName = Tuple[Optional[str], str]
MaybeScopedName = Union[str, ScopedName]


class RenameParametersPass(Pass):
    def __init__(
        self,
        root_op: Operation,
        *,
        rename_map: Optional[Dict[MaybeScopedName, MaybeScopedName]] = None,
        rename_callback: Callable[
            [Optional[str], str], Optional[ScopedName]
        ] = lambda scope, name: None,
    ):
        super().__init__(root_op)
        self.context = root_op.context
        self.rename_map = rename_map
        self.rename_callback = rename_callback
        with self.context:
            # Make a prototype named parameter attribute so we can get its
            # typeid.
            self.parameter_attr_typeid = Attribute.parse(
                '#stream.parameter.named<""::"">'
            ).typeid

    def run(self):
        globals = self.globals
        for _, global_op in self.globals.items():
            attrs = global_op.op.attributes
            try:
                initial_value = attrs["initial_value"]
            except KeyError:
                continue

            if initial_value.typeid == self.parameter_attr_typeid:
                updated_value = self.remap(initial_value)
                if updated_value != initial_value:
                    attrs["initial_value"] = updated_value

    def remap(self, parameter_attr: Attribute) -> Attribute:
        comps = _parse_parameter_attr(parameter_attr)
        if not comps:
            return parameter_attr
        if len(comps) == 1:
            orig_scope = None
            name = comps[0]
        else:
            orig_scope, name = comps

        def norm_map_result(result: MaybeScopedName) -> ScopedName:
            if isinstance(result, str):
                return orig_scope, result
            if len(result) == 1:
                return orig_scope, result[0]
            else:
                return result[0], result[1]

        def make_attr(scoped_name: ScopedName) -> Attribute:
            if scoped_name[0] is None:
                name = StringAttr.get(scoped_name[1])
                return Attribute.parse(
                    f"#stream.parameter.named<{name}> : {parameter_attr.type}"
                )
            else:
                scope = StringAttr.get(scoped_name[0])
                name = StringAttr.get(scoped_name[1])
                return Attribute.parse(
                    f"#stream.parameter.named<{scope}::{name}> : {parameter_attr.type}"
                )

        # Check the rename map.
        # Check with a fully-qualified name.
        result = self.rename_map.get((orig_scope, name))
        if result is not None:
            return make_attr(norm_map_result(result))
        # Check with just the
        result = self.rename_map.get(name)
        if result is not None:
            return make_attr(norm_map_result(result))

        # Check the callback.
        result = self.rename_callback(orig_scope, name)
        if result is not None:
            return make_attr(result)

        return parameter_attr


def _parse_parameter_attr(attr: Attribute) -> Optional[List[str]]:
    # Returns one of:
    #  None if failed to parse of not a simple named parameter without attributes.
    #  [name] for names with a default scope
    #  [scope, name] for scoped names
    # TODO: Burn this with fire. We should add Python bindings for these attributes
    # vs string parsing them.
    # TODO: The parameter attribute correctly parses C escapes but prints unescaped :(
    asm = str(attr)
    PREFIX = "#stream.parameter.named<"
    STR_PATTERN = re.compile(r'"(.*?)(?!\\)"')
    if asm.startswith(PREFIX):
        asm = asm[len(PREFIX) :]
    results = []
    # Parse a str
    m = STR_PATTERN.search(asm)
    if not m or m.start() != 0:
        return None
    results.append(m.group(1))
    asm = asm[m.end() :]
    # Parse ::
    if asm.startswith("::"):
        asm = asm[2:]
    else:
        return results
    # Parse a str
    m = STR_PATTERN.search(asm)
    if not m or m.start() != 0:
        return None
    results.append(m.group(1))
    asm = asm[m.end() :]
    if not asm.startswith(">"):
        return None
    return results


if __name__ == "__main__":
    pass_main(RenameParametersPass)
