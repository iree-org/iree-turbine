from __future__ import annotations
from abc import ABC
from dataclasses import dataclass, field, fields
import operator
import sys
import copy
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
import torch.fx as fx

from ..lang.wave_types import Memory, Register, IndexMapping
from .._support.indexing import IndexExpr, IndexSymbol, IndexSequence
from .._support.dtype import DataType
from .._support.regions import RegionGraph
from .base import OpDispatcher
from ..lang.global_symbols import MMA_ACC, MMA_LHS, MMA_RHS

T = TypeVar("T", bound=Type[Any])
AccT = TypeVar("AccT")
CustomOpT = TypeVar("CustomOpT", bound="CustomOp")
PlaceholderT = TypeVar("PlaceholderT", bound="Placeholder")


# Stubs to enable type checking of the custom ops:
# This is currently hand-written and should in future be generated from the custom ops


def allocate(
    shape: tuple[IndexExpr], dtype: DataType, address_space: IndexSymbol
) -> "Memory":
    ...


def shared_memory_barrier():
    ...


def read(
    memory: "Memory",
    elements_per_thread: Optional[IndexExpr | int] = None,
    mapping: Optional[IndexMapping] = None,
) -> "Register":
    ...


def reduction(
    axis: IndexExpr, args: Sequence["Register"]
) -> Callable[[Callable[[AccT], AccT]], AccT]:
    ...


def register(shape: tuple[IndexExpr, ...], dtype: DataType, value: float) -> "Register":
    ...


def mma(lhs: "Register", rhs: "Register", acc: "Register") -> "Register":
    ...


def write(
    register_: "Register",
    memory: "Memory",
    elements_per_thread: Optional[IndexExpr | int] = None,
    mapping: Optional[IndexMapping] = None,
):
    ...


def exp2(src: "Register") -> "Register":
    ...


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


def define_py_op(py_op: Callable) -> Callable[[T], T]:
    """
    Register python internal operators as custom ops.
    This overloads python operator specific functions such as __add__ of
    fx.Proxy with a handler in order to control the tracing of the operator and
    map it to a dynamically created sublclass of UnaryPyOp or BinaryPyOp.
    """
    op_name = py_op.__name__

    def decorator(cls: T) -> T:
        # define new subclass of cls to represent this op
        @dataclass
        class NewSubclass(cls):
            pass

        NewSubclass.tkw_op_name = op_name
        NewSubclass.__name__ = f"{op_name.capitalize()}"
        NewSubclass.__module__ = cls.__module__
        current_module = sys.modules[cls.__module__]
        setattr(current_module, NewSubclass.__name__, NewSubclass)

        original_handler = None
        if hasattr(fx.Proxy, f"__{op_name}__"):
            original_handler = getattr(fx.Proxy, f"__{op_name}__")

        def new_function(*args: Any, **kwargs: dict[str, Any]):
            dispatcher = None
            try:
                dispatcher = OpDispatcher.current()
            except IndexError:
                handler = original_handler

            if dispatcher:
                try:
                    handler = getattr(dispatcher, f"handle_{op_name}")
                except AttributeError:
                    handler = original_handler

            return handler(*args, **kwargs)

        if original_handler:
            new_function.__name__ = op_name
            NewSubclass._tracing_function = new_function
            setattr(fx.Proxy, f"__{op_name}__", new_function)

        # Return cls unchanged so we can reuse the decorator to register more ops
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


def has_same_custom_type(lhs_type: Memory, rhs_type: Memory) -> bool:
    same_shape = lhs_type.symbolic_shape == rhs_type.symbolic_shape
    same_dtype = lhs_type.dtype == rhs_type.dtype
    return same_shape and same_dtype


@dataclass
class CustomOp(ABC):
    """
    Base class for all custom fx nodes.
    """

    graph: Optional[fx.Graph] = field(default=None, init=False)
    fx_node: Optional[fx.Node] = field(default=None, init=False)
    tkw_op_name: str = field(default="unknown", init=False)
    _tracing_function: Optional[Callable[..., Any]] = field(default=None, init=False)

    @classmethod
    def from_fx_node(cls: Type[CustomOpT], node: fx.Node) -> CustomOpT:
        instance = cls(*node.args)
        instance.fx_node = node
        instance.graph = node.graph
        if hasattr(node, "index"):
            instance.index = node.index
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
        ignore_list = ["fx_node", "graph"]
        vars_list = [
            f"{key}={value}"
            for key, value in vars(self).items()
            if key not in ignore_list and value is not None
        ]
        if hasattr(self.fx_node, "index") and self.fx_node.index:
            vars_list.append(f"index={self.fx_node.index}")
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
        self.fx_node.index = None
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

    def update_arg(self, idx_or_name: int | str, value: CustomOp | fx.Node):
        """
        Update the value of an argument in the node while keeping the
        underlying fx.Node consistent.
        """
        inherited_field_count = len(CustomOp.__dataclass_fields__)
        field_names = [field.name for field in fields(self)[inherited_field_count:]]
        if isinstance(idx_or_name, str):
            if idx_or_name not in field_names:
                raise ValueError(f"Field {idx_or_name} not found")
            idx = field_names.index(idx_or_name)
        else:
            idx = idx_or_name
        if isinstance(value, CustomOp):
            value = value.fx_node
        # Skip the fields defined by the abstract base class
        if 0 <= idx < len(field_names):
            field_name = field_names[idx]
            # Set the new value for the field
            setattr(self, field_name, value)
            self.fx_node.update_arg(idx, value)
        else:
            raise IndexError("Index out of range")

    def copy(
        self, new_name: Optional[str] = None, new_graph: Optional[fx.Graph] = None
    ) -> Self:
        """Returns a duplicate of this node."""
        graph = new_graph
        if new_graph is None:
            graph = self.graph
            graph.inserting_after(self.fx_node)
        new_node = graph.node_copy(self.fx_node)
        new_node.tkw_op = self
        new_node.tkw_op_name = self.tkw_op_name
        new_node.index = copy.deepcopy(self.fx_node.index)
        if new_name:
            new_node.name = new_name
        return get_custom(new_node)

    def replace_all_uses_with(self, new_node: CustomOp | fx.Node):
        """Replace all uses of the current node with the new node."""
        for user in self.users:
            user.update_arg(user.get_node_arg_index(self), new_node)

    def erase(self):
        """Erase the current node from the graph where it exists."""
        assert (
            not self.fx_node.users
        ), f"Attempting to erase {self.fx_node} which has {len(self.fx.users)} users!"
        self.graph.erase_node(self.fx_node)

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
    def node_args(self) -> dict[int, Any]:
        """Returns the args to this custom op using subclasses of CustomOp if possible."""
        custom_args = {}
        for i, arg in enumerate(self.fx_node.args):
            if isinstance(arg, fx.Node):
                custom_args[i] = get_custom(arg)
            if isinstance(arg, list) and all(isinstance(x, fx.Node) for x in arg):
                custom_args[i] = [get_custom(x) for x in arg]
        return custom_args

    def get_node_arg_index(self, arg: CustomOp) -> Optional[CustomOp | list[CustomOp]]:
        return next(key for key, value in self.node_args.items() if value == arg)

    @property
    def users(self) -> list[Any]:
        """Returns the users of this custom op using subclasses of CustomOp if possible."""
        return [get_custom(user) for user in self.fx_node.users]

    @property
    def indexing_dims(self) -> list[IndexSymbol]:
        return []

    @property
    def index(self) -> Optional[dict[IndexSymbol, IndexSequence]]:
        if hasattr(self.fx_node, "index"):
            return self.fx_node.index
        return None

    @index.setter
    def index(self, value: Any):
        """
        Updates the index of the node based on a per-dimension index sequence.
        """
        if value is None:
            return
        if isinstance(value, dict):
            for dim, key in value.items():
                assert isinstance(
                    key, IndexSequence
                ), f"Expected IndexSequence, got {key}"
                if not hasattr(self.fx_node, "index") or self.fx_node.index is None:
                    self.fx_node.index = {}
                self.fx_node.index[dim] = key
        else:
            raise ValueError("Index must be a dict")


@define_py_op(operator.getitem)
@define_py_op(operator.add)
@define_py_op(operator.sub)
@define_py_op(operator.mul)
@define_py_op(operator.truediv)
@dataclass
class BinaryPyOp(CustomOp, ABC):
    """
    Represents a binary python operator.
    """

    lhs: Any
    rhs: Any

    @property
    def indexing_dims(self) -> list[IndexSymbol]:
        combined_dims = []
        if isinstance(self.lhs, fx.Node):
            combined_dims += get_custom(self.lhs).indexing_dims
        if isinstance(self.rhs, fx.Node):
            combined_dims += get_custom(self.rhs).indexing_dims

        unique_dims = list(dict.fromkeys(combined_dims))
        return unique_dims

    @property
    def py_operator(self) -> str:
        return self.tkw_op_name

    @property
    def type(self) -> Memory:
        lhs_type = get_custom(self.lhs).type
        rhs_type = get_custom(self.rhs).type
        has_same_type = has_same_custom_type(lhs_type, rhs_type)
        if not has_same_type:
            raise ValueError("Expected lhs and rhs to have same type post-expansion")
        return lhs_type


@define_op("exp2")
@define_py_op(operator.neg)
@dataclass
class UnaryPyOp(CustomOp, ABC):
    """
    Represents a unary python operator.
    """

    arg: fx.Node

    @property
    def indexing_dims(self) -> list[IndexSymbol]:
        return get_custom(self.arg).indexing_dims

    @property
    def py_operator(self) -> str:
        return self.tkw_op_name

    @property
    def type(self) -> Memory:
        src_type = get_custom(self.arg).type
        return src_type


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
    tkw_op_name: str = field(default="output", init=False)

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
        self.fx_node.tkw_op_name = self.tkw_op_name
        return self.fx_node


@dataclass
class Placeholder(CustomOp):
    """
    Represents a placeholder node in the graph, i.e. an input to a function.
    """

    _name: str
    _type: Optional[Type[DataType] | Type[Memory]] = None
    tkw_op_name: str = field(default="placeholder", init=False)

    @classmethod
    def from_fx_node(cls: Type[PlaceholderT], node: fx.Node) -> PlaceholderT:
        instance = cls(node.name, node.type)
        instance.fx_node = node
        instance.graph = node.graph
        return instance

    def add_to_graph(self, region_graph: RegionGraph) -> fx.Node:
        self.graph = region_graph
        self.fx_node = region_graph.create_node("placeholder", target=self._name)
        self.fx_node.tkw_op = self.__class__
        self.fx_node.tkw_op_name = self.tkw_op_name
        return self.fx_node

    def custom_string(self, value_map: dict[str, str]) -> str:
        # print all variables of the node apart from graph and op
        vars_list = [f"{key}={value}" for key, value in vars(self).items()][:-2]
        vars_str = ", ".join(vars_list)
        return f"{self.tkw_op_name}({vars_str})"

    @property
    def indexing_dims(self) -> list[IndexSymbol]:
        return list(self._type.symbolic_shape) if self._type else []

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


@define_op("allocate")
@dataclass
class Allocate(CustomOp):
    """
    Represents an allocation in an address space (such as shared memory).
    """

    shape: tuple[IndexExpr]
    distributed_shape: tuple[IndexExpr]
    dtype: DataType
    address_space: AddressSpace

    @property
    def indexing_dims(self) -> list[IndexSymbol]:
        return list(self.shape)

    @property
    def type(self) -> "Memory":
        return Memory[*self.shape, self.address_space, self.dtype]


@define_op("barrier")
@dataclass
class SharedMemoryBarrier(CustomOp):
    """
    Represents a shared memory barrier in the graph.
    """

    def is_barrier_between(self, src: fx.Node, dst: fx.Node) -> bool:
        """
        Checks if there is a barrier between the source and destination nodes.
        """
        prev_node, next_node = self.fx_node.prev, self.fx_node.next
        found_src, found_dst = prev_node == src, next_node == dst
        while prev_node.prev.op != "root" and not found_src:
            prev_node, found_src = prev_node.prev, prev_node == src
        if not found_src:
            return False
        while next_node and not found_dst:
            next_node, found_dst = next_node.next, next_node == dst
        return found_dst


@define_op("register")
@dataclass
class NewRegister(CustomOp):
    shape: tuple[IndexExpr, ...]
    dtype: DataType
    value: float

    @property
    def indexing_dims(self) -> list[IndexSymbol]:
        return list(self.shape)

    @property
    def type(self) -> "Register":
        return Register[*self.shape, self.dtype]


@define_op("mma")
@dataclass
class MMA(CustomOp):
    lhs: fx.Node
    rhs: fx.Node
    acc: fx.Node

    @property
    def indexing_dims(self) -> list[IndexSymbol]:
        combined_dims = (
            get_custom(self.lhs).indexing_dims
            + get_custom(self.rhs).indexing_dims
            + get_custom(self.acc).indexing_dims
        )
        unique_dims = list(dict.fromkeys(combined_dims))
        return unique_dims

    @property
    def lhs_type(self) -> Memory:
        return get_custom(self.lhs).type

    @property
    def rhs_type(self) -> Memory:
        return get_custom(self.rhs).type

    @property
    def acc_type(self) -> Memory:
        return get_custom(self.acc).type

    @property
    def type(self) -> Memory:
        return self.acc_type

    def operand_index(
        self, operand_map: dict[IndexSymbol, int], shape: list[IndexExpr]
    ) -> list[IndexSequence]:
        indices: list[IndexSequence] = []
        for dim in shape:
            indices.append(self.index[dim].subs(operand_map))
        return indices

    @property
    def lhs_index(self) -> list[IndexSequence]:
        operand_map = {MMA_LHS: 1, MMA_RHS: 0, MMA_ACC: 0}
        return self.operand_index(operand_map, self.lhs_type.symbolic_shape)

    @property
    def rhs_index(self) -> list[IndexSequence]:
        operand_map = {MMA_LHS: 0, MMA_RHS: 1, MMA_ACC: 0}
        return self.operand_index(operand_map, self.rhs_type.symbolic_shape)

    @property
    def acc_index(self) -> list[IndexSequence]:
        operand_map = {MMA_LHS: 0, MMA_RHS: 0, MMA_ACC: 1}
        if self.acc_type is None:
            return None
        return self.operand_index(operand_map, self.acc_type.symbolic_shape)

    def custom_string(self, value_map: dict[str, str]) -> str:
        if self.index is None:
            return super().custom_string(value_map)
        custom_str = f"{self.tkw_op_name}("
        custom_str += f"lhs={self.lhs} (index = {self.lhs_index}), "
        custom_str += f"rhs={self.rhs} (index = {self.rhs_index}), "
        custom_str += f"acc={self.acc} (index = {self.acc_index}))"
        return custom_str


@define_op("read")
@dataclass
class Read(CustomOp):
    memory: fx.Proxy
    elements_per_thread: Optional[Any] = None
    mapping: Optional[IndexMapping] = None
    _write_dependency: Optional[list[fx.Node]] = None

    @property
    def indexing_dims(self) -> list[IndexSymbol]:
        if self.mapping is not None:
            return list(self.mapping.output_shape)
        # TODO: This could contain ints.
        return list(self.type.symbolic_shape)

    @property
    def type(self) -> "Memory":
        return get_custom(self.memory).type

    @property
    def write_dependency(self) -> fx.Node:
        return self._write_dependency

    @write_dependency.setter
    def write_dependency(self, value: fx.Node):
        self.update_arg(len(self.fx_node.args) - 1, value)


@define_op("reduction")
@dataclass
class Reduction(CustomOp):
    axis: IndexSymbol
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
            node.fx_node.node.tkw_op_name = cls.tkw_op_name
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

    def iter_args(self, graph: fx.Graph) -> list[fx.Node]:
        """
        The first N placeholders in the subgraph are the iter args, where
        the total number of placeholders in the subgraph is N + M, where M
        is the number of implicit captures.
        """
        iter_args = []
        for nested_node in graph.nodes:
            custom = get_custom(nested_node)
            if isinstance(custom, Placeholder):
                iter_args.append(nested_node)
        return iter_args[: -len(self.implicit_captures)]

    def captured_vars(self, graph: fx.Graph) -> list[fx.Node]:
        """
        The last M placeholders in the subgraph are the captured vars, where
        the total number of placeholders in the subgraph is N + M, where N
        is the number of iter args.
        """
        captured_vars = []
        for nested_node in graph.nodes:
            custom = get_custom(nested_node)
            if isinstance(custom, Placeholder):
                captured_vars.append(nested_node)
        return captured_vars[-len(self.implicit_captures) :]

    @property
    def type(self) -> list[Memory | Register]:
        return [get_custom(x).type for x in self.init_args]

    def outputs(self, graph: fx.Graph) -> list[fx.Node]:
        for node in graph.nodes:
            if isinstance(get_custom(node), Output):
                return node.args[0]


@define_op("write")
@dataclass
class Write(CustomOp):
    register_: fx.Proxy
    memory: fx.Proxy
    elements_per_thread: Optional[Any]
    mapping: Optional[IndexMapping] = None

    @property
    def indexing_dims(self) -> list[IndexSymbol]:
        if self.mapping is not None:
            return list(self.mapping.input_shape)
        # TODO: This could contain ints.
        return list(self.type.symbolic_shape)

    @property
    def type(self) -> "Memory":
        return get_custom(self.memory).type


@define_op("get_result")
@dataclass
class GetResult(CustomOp):
    value: fx.Node
    res_idx: int

    @property
    def type(self) -> "Memory":
        return get_custom(self.value).type[self.res_idx]

    @property
    def indexing_dims(self) -> list[IndexSymbol]:
        expand_dims: list[IndexSymbol] = []
        for user in self.users:
            for indexing_dim in user.indexing_dims:
                if indexing_dim not in expand_dims:
                    expand_dims.append(indexing_dim)
        return expand_dims
