import inspect
import types
import typing
import functools
import copy
import torch
from typing import Any, Callable, Optional, TypeAlias
from .constraints import Constraint

import numpy as np
from sympy import Symbol
from sympy.core.expr import Expr
from .._support.shaped_type import ShapedType

from .. import lang as tkl
from .. import wave as tkw
from ..wave import IndexMapping


def wave_sim(constraints: Optional[list[Constraint]] = None):
    """Kernel simulator decorator.

    This decorator wraps kernel function into simulator harness.

    When kernel is invoked, underlying function is copied and symbolic
    expressions captured in '__globals__' and '__closure__' are replaced with
    numerical values, extracted from input tensors shapes.

    Also, all kernel APIs are replaced by 'proxy' API, which just execute ops
    immediately on host intead of tracing them.
    """

    def decorator(f: Callable[..., Any]) -> Callable[..., Any]:
        args_handler = _process_func_annotations(f)

        def func_wrapper(*args):
            global _api_subs
            global _symbolic_shapes
            subs = copy.copy(_api_subs)
            if args_handler:
                args_handler(args, subs)

            new_func = _resolve_symbols(f, subs)
            try:
                return new_func(*args)
            finally:
                _symbolic_shapes = {}

        return func_wrapper

    return decorator


IndexExpr: TypeAlias = Expr
HandlerFunc: TypeAlias = Callable[[tuple[...], dict[Any, Any]], None]


def _to_indices(src: tuple[IndexExpr, ...]) -> tuple[int, ...]:
    return tuple(int(i) for i in src)


def _get_shaped_handler(
    arg_idx: int, shape: tuple[IndexExpr, ...], prev_handler: HandlerFunc
) -> HandlerFunc:
    def handler(args: tuple[...], subs: dict[Any, Any]) -> None:
        if prev_handler:
            prev_handler(args, subs)

        arg = args[arg_idx]
        _symbolic_shapes[id(arg)] = shape
        for i, sym in enumerate(shape):
            if isinstance(sym, Symbol):
                subs[sym] = arg.shape[i]

    return handler


def _visit_annotation(
    ann, arg_idx: int, prev_handler: HandlerFunc
) -> Optional[HandlerFunc]:
    if isinstance(ann, ShapedType):
        return _get_shaped_handler(arg_idx, ann.symbolic_shape, prev_handler)

    return None


def _process_func_annotations(func: Callable[..., Any]) -> Optional[HandlerFunc]:
    """Process symbols in func annotation, so iteration dimensions can be extracted.

    Returns a function which extract shapes from kernel args and generates a
    substitution map, which can be used to replace symbols inside the kernel
    with actual values, came from arguments.
    """
    handler = None
    ann = inspect.get_annotations(func)
    for i, arg in enumerate(inspect.signature(func).parameters):
        arg_ann = ann.get(arg, None)
        if arg_ann is None:
            continue

        new_handler = _visit_annotation(arg_ann, i, handler)
        if new_handler is not None:
            handler = new_handler

    return handler


def _resolve_symbols(func: Callable[..., Any], symbols: dict[Any, Any]):
    """Copy function and update __globals__ and __closure__ vars

    Copy function while updating __globals__ and __closure__ vars according to
    'symbols' map.
    """
    old_closure = func.__closure__
    new_closure = None

    sym_subs = [(key, val) for key, val in symbols.items() if isinstance(key, Symbol)]

    def resolve_impl(val):
        if isinstance(val, Symbol):
            return symbols.get(val, None)
        elif isinstance(val, Expr):
            return val.subs(sym_subs)
        elif isinstance(val, tkw.IndexMapping):
            ret = val.substitute(sym_subs)

            inp_shape = _to_indices(
                sym.subs(sym_subs) for sym in ret.input_mapping.keys()
            )
            out_shape = _to_indices(
                sym.subs(sym_subs) for sym in ret.output_mapping.keys()
            )
            iter_shape = _to_indices(sym.subs(sym_subs) for sym in ret.iteration_shape)
            setattr(ret, "inp_shape", inp_shape)
            setattr(ret, "out_shape", out_shape)
            setattr(ret, "iter_shape", iter_shape)
            return ret

        try:
            return symbols.get(val, None)
        except:
            # For non-hashable types `symbols.get` will raise an exception,
            # we don't care about them anyways, so ignore.
            return None

    if old_closure is not None:
        cell_cls = type(old_closure[0])

        def resolve_cell(cell):
            res = resolve_impl(cell.cell_contents)
            if res is None:
                res = cell
            else:
                res = cell_cls(res)

            return res

        new_closure = tuple(resolve_cell(cell) for cell in old_closure)

    def resolve_global(val):
        res = resolve_impl(val)
        if res is None:
            res = val

        return res

    new_globals = {key: resolve_global(val) for key, val in func.__globals__.items()}

    g = types.FunctionType(
        func.__code__,
        new_globals,
        name=func.__name__,
        argdefs=func.__defaults__,
        closure=new_closure,
    )
    g = functools.update_wrapper(g, func)
    g.__kwdefaults__ = func.__kwdefaults__
    return g


_symbolic_shapes = {}
_api_subs = {}


def _get_symbolic_shape(a: Any) -> tuple[IndexExpr, ...]:
    assert id(a) in _symbolic_shapes, "Symbolic shape is not available"
    return _symbolic_shapes[id(a)]


def _set_symbolic_shape(a: Any, shape: tuple[IndexExpr, ...]) -> None:
    _symbolic_shapes[id(a)] = shape


class _RegisterProxy:
    def __getitem__(self, indices: tuple[...]):
        shape = indices[:-1]
        dtype = indices[-1]
        return _ShapedRegister(shape, dtype)


class _ShapedRegister:
    def __init__(self, shape: tuple[IndexExpr, ...], dtype: Any) -> None:
        self.shape = shape
        self.dtype = dtype

    def __call__(self, init: Any) -> "Register":
        return torch.full(self.shape, init, dtype=self.dtype)


class _TklProxy:
    f16 = torch.float16
    f32 = torch.float32
    Register = _RegisterProxy()


def _reduction_proxy(axis: int, init_args: list[Any]):
    def decorator(func: Callable[..., Any]) -> Any:
        return func(*init_args)

    return decorator


def _read_proxy(
    memory: "Memory",
    elements_per_thread: Optional[IndexExpr] = None,
    mapping: Optional[IndexMapping] = None,
) -> "Register":
    if mapping:
        input_sym_shape = _get_symbolic_shape(memory)
        inp_mapping = mapping.map_input_indices(input_sym_shape)
        res_mapping = mapping.map_output_indices()
        iters = mapping.iters

        def mapping_func(ind_mapping, indices):
            subs = [(ind, val) for ind, val in zip(iters, indices)]
            return _to_indices(ind.subs(subs) for ind in ind_mapping)

        iter_shape = mapping.iter_shape
        res_shape = mapping.out_shape

        res = torch.zeros(res_shape)
        _set_symbolic_shape(res, mapping.output_shape)
        for index in np.ndindex(*iter_shape):
            inp_mapped = mapping_func(inp_mapping, index)
            res_mapped = mapping_func(res_mapping, index)
            res[res_mapped] = memory[inp_mapped]

    else:
        res = memory.clone()
        _set_symbolic_shape(res, _get_symbolic_shape(memory))

    return res


def _write_proxy(
    src: "Register",
    dst: "Memory",
    elements_per_thread: Optional[IndexExpr] = None,
    mapping: Optional[IndexMapping] = None,
) -> None:
    if mapping:
        input_sym_shape = _get_symbolic_shape(src)
        inp_mapping = mapping.map_input_indices(input_sym_shape)
        res_mapping = mapping.map_output_indices()
        iters = mapping.iters

        def mapping_func(ind_mapping, indices):
            subs = [(ind, val) for ind, val in zip(iters, indices)]
            return _to_indices(ind.subs(subs) for ind in ind_mapping)

        iter_shape = mapping.iter_shape

        for index in np.ndindex(*iter_shape):
            inp_mapped = mapping_func(inp_mapping, index)
            res_mapped = mapping_func(res_mapping, index)
            dst[res_mapped] = src[inp_mapped]

        return

    dst[:] = src


def _mma_proxy(a: "Register", b: "Register", acc: "Register") -> "Register":
    a_shape = _get_symbolic_shape(a)
    b_shape = _get_symbolic_shape(b)
    res = torch.matmul(a, b.T) + acc
    _set_symbolic_shape(res, (a_shape[0], b_shape[0]))
    return res


class _TkwProxy:
    reduction = _reduction_proxy
    read = _read_proxy
    write = _write_proxy
    mma = _mma_proxy


_api_subs[tkl] = _TklProxy
_api_subs[tkw] = _TkwProxy
