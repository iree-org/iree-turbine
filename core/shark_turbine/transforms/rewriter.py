# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Dict, List, Optional, Set, Union, Type, cast

import argparse
import sys

from iree.compiler.ir import (
    Block,
    BlockArgument,
    Context,
    FlatSymbolRefAttr,
    IntegerAttr,
    Operation,
    OpResult,
    OpView,
    Region,
    StringAttr,
    Value,
)

from iree.compiler.passmanager import (
    PassManager,
)

from .builder import Builder
from .merger import Merger

__all__ = [
    "Builder",
    "GlobalLoadMatcher",
    "GlobalsDict",
    "NamedOpMatcher",
    "OpMatchResult",
    "Pass",
    "Transpose2DMatcher",
    "match_children",
    "pass_main",
]

###############################################################################
# Matching
###############################################################################


class OpMatchResult:
    def __init__(self, op: Operation):
        self.op = op

    def __repr__(self):
        return f"{type(self).__name__}({self.op})"


OperationParent = Union[None, Operation, OpView, Region, Block, OpMatchResult]
OperationParentOrList = Union[OperationParent, List[OperationParent]]
MaybeOperation = Union[None, Value, OpMatchResult, Operation, OpView]


class OpMatcher:
    """Base class for things that match an operation."""

    def __call__(self, maybe_op: MaybeOperation) -> Optional[OpMatchResult]:
        if maybe_op is None:
            return None
        if isinstance(maybe_op, OpMatchResult):
            op = maybe_op.op
        elif isinstance(maybe_op, Operation):
            op = maybe_op
        elif isinstance(maybe_op, OpView):
            op = maybe_op.operation
        elif isinstance(maybe_op, Value):
            if OpResult.isinstance(maybe_op):
                op = _op_as_operation(OpResult(maybe_op).owner)
            elif BlockArgument.isinstance(maybe_op):
                return None
        else:
            raise ValueError(f"Unexpected OpMatcher input: {type(maybe_op)}")

        return self._match(op)

    def _match(self, op: Operation) -> Optional[OpMatchResult]:
        raise NotImplementedError


class NamedOpMatcher(OpMatcher):
    """Matches operations by name."""

    def __init__(self, *op_names: str):
        self.op_names = op_names

    def _match(self, op: Operation) -> Optional[OpMatchResult]:
        if op.name in self.op_names:
            return self.match(op)
        return None

    def match(self, op: Operation) -> Optional[OpMatchResult]:
        return OpMatchResult(op)


def get_child_blocks(of: OperationParentOrList) -> List[Block]:
    """Gets all child blocks of an Operation, Region, or Block (self)."""
    blocks: List[Block] = []
    if of is None:
        return blocks

    if isinstance(of, OpMatchResult):
        of = of.op

    if isinstance(of, (Operation, OpView)):
        for r in of.regions:
            for b in r.blocks:
                blocks.append(b)
    elif isinstance(of, Region):
        for b in of.blocks:
            blocks.append(b)
    elif isinstance(of, Block):
        blocks.append(of)
    elif isinstance(of, List):
        for p in of:
            blocks.extend(get_child_blocks(p))
    else:
        raise ValueError(f"Must be an Operation, Region, or Block. Got: {type(of)}")
    return blocks


def match_children(
    of: OperationParentOrList, *matchers: OpMatcher
) -> List[OpMatchResult]:
    """Matches children of a parent.

    For any child, the match result from the first matcher which matches
    will be added to the result list.
    """
    results = []
    blocks = get_child_blocks(of)
    for b in blocks:
        for op in b.operations:
            for m in matchers:
                result = m(op.operation)
                if result:
                    results.append(result)
                    break
    return results


###############################################################################
# Specific op matchers
###############################################################################


class FuncOpMatcher(NamedOpMatcher):
    """Matches func.func functions."""

    def __init__(self):
        super().__init__("func.func")


class GlobalOpResult(OpMatchResult):
    @property
    def sym_name(self) -> str:
        return StringAttr(self.op.attributes["sym_name"]).value


class GlobalOpMatcher(NamedOpMatcher):
    """Matches global operations."""

    def __init__(self):
        super().__init__("util.global")

    def match(self, op: Operation) -> Optional[GlobalOpResult]:
        return GlobalOpResult(op)


class Transpose2DResult(OpMatchResult):
    @property
    def input(self) -> Value:
        return self.op.operands[0]


class Transpose2DMatcher(NamedOpMatcher):
    def __init__(self):
        super().__init__("torch.aten.transpose.int")

    def match(self, op: Operation) -> Optional[Transpose2DResult]:
        result = Transpose2DResult(op)
        if not ConstantIntMatcher(0)(op.operands[1]) or not ConstantIntMatcher(1)(
            op.operands[2]
        ):
            return None
        return result


class ConstantIntMatcher(NamedOpMatcher):
    def __init__(self, value: int):
        super().__init__("torch.constant.int")
        self.value = value

    def match(self, op: Operation):
        value_attr = IntegerAttr(op.attributes["value"])
        if value_attr.value != self.value:
            return None
        return OpMatchResult(op)


class GlobalLoadResult(OpMatchResult):
    def __init__(self, op: Operation):
        super().__init__(op)
        self.resolved_global: Optional[GlobalOpResult] = None

    @property
    def global_ref(self) -> str:
        return FlatSymbolRefAttr(self.op.attributes["global"]).value


class GlobalLoadMatcher(NamedOpMatcher):
    def __init__(self, globals: Optional["GlobalsDict"] = None):
        super().__init__("util.global.load", "torch_c.from_builtin_tensor")
        self.globals = globals

    def match(self, op: Operation) -> Optional[GlobalLoadResult]:
        # Skip over any builtin tensor conversion.
        if op.name == "torch_c.from_builtin_tensor":
            op = _value_as_op_or_none(op.operands[0])
            if not op:
                return None

        result = GlobalLoadResult(op)
        if self.globals:
            result.resolved_global = self.globals.get(result.global_ref)
        return result


###############################################################################
# Passes
###############################################################################

GlobalsDict = Dict[str, GlobalOpResult]


class Pass:
    """Callable which performs some mutation on the IR."""

    def __init__(self, root_op: Operation):
        self.root_op = root_op
        self.builder = Builder(root_op.context)

    def run(self):
        raise NotImplementedError

    @property
    def funcs(self) -> List[OpMatchResult]:
        return match_children(self.root_op, FuncOpMatcher())

    @property
    def globals(self) -> GlobalsDict:
        results = match_children(self.root_op, GlobalOpMatcher())
        return {r.sym_name: r for r in results}

    def merge_module(self, source_module: Operation) -> Merger:
        """Merges the given source module into the root.

        See documentation for the Merger for more information.
        """
        merger = Merger(source_module, self.root_op)
        merger.merge()
        return merger

    def inline(self):
        """Runs the inliner."""
        with self.root_op.context:
            pm = PassManager.parse("builtin.module(inline)")
            pm.run(self.root_op)

    def cleanup(self):
        """Runs module cleanup passes."""
        with self.root_op.context:
            pm = PassManager.parse("builtin.module(canonicalize, symbol-dce)")
            pm.run(self.root_op)

    def replace_op(self, old_op: Operation, *new_results: Value):
        old_results = old_op.results
        assert len(old_results) == len(
            new_results
        ), "Can only replace_op with the same arity"
        for old_result, new_result in zip(old_results, new_results):
            old_result.replace_all_uses_with(new_result)
        self.erase_unused_op(old_op)

    def erase_unused_op(self, op: Operation):
        """Recursively erases any unused torch ops, starting with op.

        Torch ops generally are not erased automatically, but as part of
        pattern matching, when we know we want to replace them, we can do
        this ourself.
        """
        worklist: Set[Operation] = set()
        worklist.add(op)
        while worklist:
            ops = worklist
            worklist = set()
            for op in ops:
                if not _is_erasable_value_op(op):
                    continue
                if not _op_is_live(op):
                    for operand in op.operands:
                        if OpResult.isinstance(operand):
                            worklist.add(operand.owner)
                    op.erase()


def pass_main(pass_class: Type[Pass], *, argv=None):
    """Simple main entry-point which reads a file, runs a callback and outputs."""
    parser = argparse.ArgumentParser(description="Rewrite driver")
    parser.add_argument("input_file", help="File to process")
    parser.add_argument("-o", dest="output_file", help="Output file")
    args = parser.parse_args(argv)

    with Context() as context:
        with open(args.input_file, "r") as f:
            module_op = Operation.parse(f.read(), source_name=args.input_file)

        p = pass_class(module_op)
        p.run()

        if args.output_file:
            with open(args.output_file, "wb") as f:
                module_op.print(file=f, binary=True)
        else:
            module_op.print(file=sys.stdout)


###############################################################################
# Utilities
###############################################################################


def _value_as_op_or_none(value: Value) -> Optional[Operation]:
    if OpResult.isinstance(value):
        return _op_as_operation(OpResult(value).owner)
    return None


def _op_as_operation(op: Union[Operation, OpView]) -> Operation:
    if isinstance(op, OpView):
        return op.operation
    else:
        return op


def _op_is_live(op: Operation) -> bool:
    for r in op.results:
        try:
            next(r.uses)
            return True
        except StopIteration:
            pass
    return False


def _is_erasable_value_op(op: Operation):
    name = op.name
    return name.startswith("torch.") or name.startswith("torch_c.")
