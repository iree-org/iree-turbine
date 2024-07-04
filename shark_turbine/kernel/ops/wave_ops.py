from abc import ABC, abstractclassmethod
from dataclasses import dataclass, field, fields
from functools import wraps
import sys
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Optional,
    Self,
    Sequence,
    Type,
    TypeVar,
    final,
)
import sympy
import torch.fx as fx

if TYPE_CHECKING:
    from ..lang.wave_types import AddressSpace, Memory, Register
from .._support.indexing import IndexExpr, IndexSymbol
from .._support.dtype import DataType
from .._support.regions import RegionGraph
from .base import OpDispatcher

T = TypeVar("T", bound=Type[Any])
AccT = TypeVar("AccT")
CustomOpT = TypeVar("CustomOpT", bound="CustomOp")
PlaceholderT = TypeVar("PlaceholderT", bound="Placeholder")


# Stubs to enable type checking of the custom ops:
# This is currently hand-written and should in future be generated from the custom ops
def register(
    shape: tuple[IndexExpr, ...], dtype: DataType, value: float
) -> "Register": ...


def read(
    memory: "Memory", elements_per_thread: Optional[IndexExpr] = None
) -> "Register": ...


def mma(lhs: "Register", rhs: "Register", acc: "Register") -> "Register": ...


def reduction(
    axis: IndexExpr, args: Sequence["Register"]
) -> Callable[[Callable[[AccT], AccT]], AccT]: ...


def write(
    register_: "Register",
    memory: "Memory",
    elements_per_thread: Optional[IndexExpr | int] = None,
): ...


def define_op(op_name: str) -> Callable[[T], T]:
    def decorator(cls: T) -> T:
        cls.tkw_op_name = op_name

        def new_function(*args: Any, **kwargs: dict[str, Any]):
            dispatcher = OpDispatcher.current()
            try:
                handler = getattr(dispatcher, f"handle_{op_name}")
            except AttributeError:
                raise AttributeError(
                    f"The current OpDispatcher ({dispatcher}) does not register a handler for {op_name}"
                )

            return handler(*args, **kwargs)

        new_function.__name__ = op_name
        current_module = sys.modules[cls.__module__]
        setattr(current_module, op_name, new_function)
        cls._tracing_function = new_function

        return cls

    return decorator


def get_custom(node: fx.Node) -> "CustomOp":
    """Get the corresponding CustomOp for a given fx.Node."""
    if isinstance(node, CustomOp):
        print("Careful! You passed a custom op where an fx.Node was required.")
        return node
    if not isinstance(node, fx.Node):
        raise ValueError("Expected an fx.Node")

    # If the node was created as a CustomOp it has a corresponding field
    if hasattr(node, "tkw_op"):
        return node.tkw_op.from_fx_node(node)
    if node.op == "placeholder":
        return Placeholder.from_fx_node(node)
    if node.op == "output":
        return Output.from_fx_node(node)
    return Unknown.from_fx_node(node)


@dataclass
class CustomOp(ABC):
    """
    Base class for all custom fx nodes.
    """

    graph: Optional[fx.Graph] = field(default=None, init=False)
    fx_node: Optional[fx.Node] = field(default=None, init=False)
    tkw_op_name: str = field(default="unknown", init=False)
    _tracing_function: Optional[Callable[..., Any]] = field(default=None, init=False)
    index: Optional[IndexExpr] = field(default=None, init=False)

    @classmethod
    def from_fx_node(cls: Type[CustomOpT], node: fx.Node) -> CustomOpT:
        instance = cls(*node.args)
        instance.fx_node = node
        instance.graph = node.graph
        return instance

    def __post_init__(self):
        # Subclasses do not inherit hash and eq from the superclass
        self.__class__.__hash__ = CustomOp.__hash__
        self.__class__.__eq__ = CustomOp.__eq__

    def __hash__(self):
        return hash(self.fx_node)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, CustomOp):
            return False
        return self.fx_node == other.fx_node

    def __str__(self) -> str:
        return self.custom_string({})

    def custom_string(self, value_map: dict[str, str]) -> str:
        # print all variables of the node apart from graph and fx_node
        vars_list = [f"{key}={value}" for key, value in vars(self).items()][:-2]
        vars_str = ", ".join(vars_list)
        return f"{self.tkw_op_name}({vars_str})"

    def add_to_graph(self, region_graph: RegionGraph) -> fx.Node:
        arg_list = tuple([value for _, value in vars(self).items()])
        self.graph = region_graph
        self.fx_node = region_graph.create_node(
            "call_function",
            target=self._tracing_function,
            args=arg_list,
            kwargs={},
        )
        self.fx_node.tkw_op = self.__class__
        self.fx_node.tkw_op_name = self.tkw_op_name
        return self.fx_node

    def _add_proxy_to_graph(self, region_graph: RegionGraph):
        arg_list = tuple([value for _, value in vars(self).items()])
        self.graph = region_graph
        self.fx_node = region_graph.create_proxy(
            "call_function",
            target=self._tracing_function,
            args=arg_list,
            kwargs={},
        )

    def update_arg(self, idx_or_name: int | str, value: Any):
        """
        Update the value of an argument in the node while keeping the
        underlying fx.Node consistent.
        """
        dataclass_fields = fields(self)[5:]
        if isinstance(idx_or_name, str):
            if idx_or_name not in [field.name for field in dataclass_fields]:
                raise ValueError(f"Field {idx_or_name} not found")
            idx = [field.name for field in dataclass_fields].index(idx_or_name)
        else:
            idx = idx_or_name
        if isinstance(value, CustomOp):
            value = value.fx_node
        # Skip the fields defined by the abstract base class
        if 0 <= idx < len(dataclass_fields):
            self.node_args[idx] = value  # TODO: Does this make sense?
            field_name = dataclass_fields[idx].name
            # Set the new value for the field
            setattr(self, field_name, value)
            fx_val = value.fx_node if isinstance(value, CustomOp) else value
            self.fx_node.update_arg(idx, fx_val)
        else:
            raise IndexError("Index out of range")

    def copy(self, new_name: Optional[str] = None) -> Self:
        """Returns a duplicate of this node in order to expand the graph."""
        self.graph.inserting_after(self.fx_node)
        new_node = self.graph.node_copy(self.fx_node)
        new_node.tkw_op = self
        if new_name:
            new_node.name = new_name
        return get_custom(new_node)

    @classmethod
    def handle(cls, graph, *args, **kwargs) -> fx.Node:
        node = cls(*args, **kwargs)
        node._add_proxy_to_graph(graph)
        node.fx_node.node.tkw_op = cls
        node.fx_node.node.tkw_op_name = cls.tkw_op_name
        return node.fx_node

    @property
    def name(self) -> str:
        if hasattr(self, "_name"):
            return self._name
        return self.fx_node.name

    @property
    def node_args(self) -> list[Any]:
        """Returns the args to this custom op using subclasses of CustomOp if possible."""
        return [
            get_custom(arg) if isinstance(arg, fx.Node) else arg
            for arg in self.fx_node.args
        ]

    @property
    def users(self) -> list[Any]:
        """Returns the users of this custom op using subclasses of CustomOp if possible."""
        return [get_custom(user) for user in self.fx_node.users]

    @property
    def indexing_dims(self) -> list[sympy.Symbol]:
        return []


@final
@dataclass
class Unknown(CustomOp):
    """
    Represents an fx.Node that has no corresponding CustomNode class.
    """

    args: Sequence[Any]
    kwargs: dict[Any, Any]

    @classmethod
    def from_fx_node(cls, node: fx.Node) -> "Unknown":
        instance = cls(node.args, node.kwargs)
        instance.fx_node = node
        instance.graph = node.graph
        return instance

    def custom_string(self, value_map: dict[str, str]) -> str:
        # print all variables of the node apart from graph and op
        vars_list = [f"{key}={value}" for key, value in vars(self).items()][:-2]
        vars_str = ", ".join(vars_list)
        return f"unknown: {self.fx_node.name}({vars_str})"


@dataclass
class Output(CustomOp):
    """
    Represents an output node in the graph, representing the return value of a
    traced function.
    """

    return_vals: Sequence[Any]

    @classmethod
    def from_fx_node(cls: Type[CustomOpT], node: fx.Node) -> CustomOpT:
        instance = cls(node.args)
        instance.fx_node = node
        instance.graph = node.graph
        return instance

    def add_to_graph(self, region_graph: RegionGraph) -> fx.Node:
        self.graph = region_graph
        self.fx_node = region_graph.create_node(
            "output",
            target="output",
            args=tuple([self.return_vals]),
            kwargs={},
        )
        self.fx_node.tkw_op = self.__class__
        return self.fx_node


@dataclass
class Placeholder(CustomOp):
    """
    Represents a placeholder node in the graph, i.e. an input to a function.
    """

    _name: str
    _type: Optional[DataType]
    tkw_op_name: str = field(default="placeholder", init=False)

    @classmethod
    def from_fx_node(cls: Type[PlaceholderT], node: fx.Node) -> PlaceholderT:
        instance = cls(node.name, node.type)
        instance.fx_node = node
        instance.graph = node.graph
        return instance

    def custom_string(self, value_map: dict[str, str]) -> str:
        # print all variables of the node apart from graph and op
        vars_list = [f"{key}={value}" for key, value in vars(self).items()][:-2]
        vars_str = ", ".join(vars_list)
        return f"{self.tkw_op_name}({vars_str})"

    @property
    def indexing_dims(self) -> list[sympy.Symbol]:
        return list(self._type.symbolic_shape)

    @property
    def type(self) -> "Memory":
        return self._type


@dataclass
class IterArg(Placeholder):
    """
    Represents a specific placeholder node in the graph that is an iter arg of
    a reduction node.
    """


# Ops modeling TKW operations in the kernel language


@define_op("register")
@dataclass
class NewRegister(CustomOp):
    shape: tuple[IndexExpr, ...]
    dtype: DataType
    value: float

    @property
    def indexing_dims(self) -> list[sympy.Symbol]:
        return list(self.shape)


@define_op("mma")
@dataclass
class MMA(CustomOp):
    lhs: fx.Node
    rhs: fx.Node
    acc: fx.Node

    @property
    def indexing_dims(self) -> list[sympy.Symbol]:
        combined_dims = (
            get_custom(self.lhs).indexing_dims
            + get_custom(self.rhs).indexing_dims
            + get_custom(self.acc).indexing_dims
        )
        unique_dims = list(dict.fromkeys(combined_dims))
        return unique_dims


@define_op("read")
@dataclass
class Read(CustomOp):
    memory: fx.Proxy
    elements_per_thread: Optional[Any] = None

    @property
    def indexing_dims(self) -> list[sympy.Symbol]:
        # TODO: This could contain ints.
        return list(self.memory.type.symbolic_shape)

    @property
    def type(self) -> "Memory":
        return self.memory.type


@define_op("reduction")
@dataclass
class Reduction(CustomOp):
    axis: IndexExpr
    init_args: Sequence[Any]
    subgraph_name: str
    implicit_captures: Sequence[fx.Proxy]

    @classmethod
    def handle(cls, graph, *args, **kwargs):
        def wrapper(f):
            with graph.subtracer() as subtracer:
                subgraph_name, implicit_captures = subtracer.trace(f)
            node = Reduction(
                *args,
                **kwargs,
                subgraph_name=subgraph_name,
                implicit_captures=implicit_captures,
            )
            # Remember which placeholders are init args. This connection gets
            # lost otherwise
            for nested_node in graph.subgraphs[subgraph_name].nodes:
                if nested_node.op == "placeholder":
                    if nested_node not in [
                        var.node
                        for var in graph.inner_freevars[graph.subgraphs[subgraph_name]]
                    ]:
                        nested_node.tkw_op = IterArg

            node._add_proxy_to_graph(graph)
            node.fx_node.node.tkw_op = cls
            return node.fx_node

        return wrapper

    @property
    def indexing_dims(self) -> list[IndexSymbol]:
        expand_dims: list[IndexSymbol] = []
        for user in self.users:
            for indexing_dim in user.indexing_dims:
                if indexing_dim not in expand_dims:
                    expand_dims.append(indexing_dim)
        return expand_dims


@define_op("write")
@dataclass
class Write(CustomOp):
    register_: fx.Proxy
    memory: fx.Proxy
    elements_per_thread: Optional[Any]

    @property
    def indexing_dims(self) -> list[sympy.Symbol]:
        # TODO: This could contain ints.
        return list(self.memory.type.symbolic_shape)

    @property
    def type(self) -> "Memory":
        return self.memory.type


@define_op("get_result")
@dataclass
class GetResult(CustomOp):
    value: fx.Proxy
    res_idx: int

    @property
    def type(self) -> "Memory":
        return self.value.type
