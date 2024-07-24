import inspect
import types
import typing
import functools
import copy
import torch
from typing import Any, Callable, Optional, TypeAlias
from .constraints import Constraint

from sympy import Symbol
from sympy.core.expr import Expr
from .._support.shaped_type import ShapedType

from .. import lang as tkl
from .. import wave as tkw


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
            subs = copy.copy(_api_subs)
            if args_handler:
                args_handler(args, subs)

            new_func = _resolve_symbols(f, subs)
            return new_func(*args)

        return func_wrapper

    return decorator


IndexExpr: TypeAlias = Expr
HandlerFunc: TypeAlias = Callable[[tuple[...], dict[Any, Any]], None]


def _get_shaped_handler(
    arg_idx: int, shape: tuple[IndexExpr, ...], prev_handler: HandlerFunc
) -> HandlerFunc:
    def handler(args: tuple[...], subs: dict[Any, Any]) -> None:
        if prev_handler:
            prev_handler(args, subs)

        arg = args[arg_idx]
        for i, sym in enumerate(shape):
            if isinstance(sym, Symbol):
                subs[sym] = arg.shape[i]

    return handler


def _visit_annotation(
    ann, arg_idx: int, prev_handler: HandlerFunc
) -> None | HandlerFunc:
    if isinstance(ann, ShapedType):
        return _get_shaped_handler(arg_idx, ann.symbolic_shape, prev_handler)

    return None


def _process_func_annotations(func: Callable[..., Any]) -> HandlerFunc:
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
            return tkw.IndexMapping(_resolve_symbols(val.mapping_func, symbols))

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


_api_subs = {}


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
    mapping: Callable[..., Any] = None,
    shape: tuple[IndexExpr, ...] = None,
) -> "Register":
    if mapping:
        assert shape
        mapping_func = mapping.mapping_func
        res = torch.zeros(shape)
        for index in np.ndindex(*shape):
            mapped = mapping_func(*index)
            res[index] = memory[mapped]

        return res

    return memory.clone()


def _write_proxy(
    src: "Register",
    dst: "Memory",
    elements_per_thread: Optional[IndexExpr] = None,
    mapping: Callable[..., Any] = None
):
    if mapping:
        mapping_func = mapping.mapping_func
        for index in np.ndindex(*src.shape):
            mapped = mapping_func(*index)
            dst[mapped] = src[index]

        return

    dst[:] = src


def _mma_proxy(a: "Register", b: "Register", acc: "Register") -> "Register":
    return torch.matmul(a, b.T) + acc


class _TkwProxy:
    reduction = _reduction_proxy
    read = _read_proxy
    write = _write_proxy
    mma = _mma_proxy


_api_subs[tkl] = _TklProxy
_api_subs[tkw] = _TkwProxy
