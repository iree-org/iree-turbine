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
from ..lang.global_symbols import *
from .._support.indexing import IndexExpr, IndexSymbol, IndexSequence
from .._support.dtype import DataType
from .._support.regions import RegionGraph
from .base import OpDispatcher
import numpy as np

if TYPE_CHECKING:
    from ..wave.constraints import Constraint
    from ..wave.scheduling.resources import Operation

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


def extract(
    register: "Register",
    offsets: tuple[IndexExpr],
) -> "Register":
    ...


def extract_slice(
    register: "Register",
    offsets: tuple[IndexExpr],
    sizes: tuple[IndexExpr],
    strides: tuple[IndexExpr],
) -> "Register":
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
    axis: IndexExpr, init_args: Sequence["Register"]
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


def maximum(lhs: "Register", rhs: "Register") -> "Register":
    ...


def broadcast(
    arg: "Register", target_shape: Optional[IndexExpr | int] = None
) -> "Register":
    ...


def sum(
    src: "Register",
    acc: Optional["Register"] = None,
    dim: Optional[IndexExpr | int] = None,
) -> "Register":
    ...


def max(
    src: "Register",
    acc: Optional["Register"] = None,
    dim: Optional[IndexExpr | int] = None,
) -> "Register":
    ...


def shuffle(src: "Register", offset: int, width: int) -> "Register":
    ...


def cast(src: "Register", dtype: DataType) -> "Register":
    ...


def permute(src: "Register", target_shape: Sequence[IndexExpr]) -> "Register":
    ...


def reshape(inputs: Sequence["Register"]) -> "Register":
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


def define_interface_op(op_name: str) -> Callable[[T], T]:
    """
    Generate new subclass for op handling, deriving from the base interface class.
    Generated subclass can be used for emitting the op from compiler/python side,
    by calling the generated subclass name.

    The generated subclass name would be pascal case of the TKW op name. For example:
    "tkw.op_name" -> "OpName"
    "tkw.exp2" -> "Exp2"
    """

    def decorator(cls: T) -> T:
        # define new subclass of cls to represent this op
        @dataclass
        class NewSubclass(cls):
            pass

        NewSubclass.tkw_op_name = op_name
        pascal_op_name = op_name.replace("_", " ").title().replace(" ", "")
        NewSubclass.__name__ = f"{pascal_op_name}"
        NewSubclass.__module__ = cls.__module__
        current_module = sys.modules[NewSubclass.__module__]
        setattr(current_module, NewSubclass.__name__, NewSubclass)
        if cls.__name__ == NewSubclass.__name__:
            raise ValueError(
                f'Subclass cannot have same name as base interface class{cls.__name__}. Did you mean to use"define_op" instead.'
            )

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
        setattr(current_module, op_name, new_function)
        NewSubclass._tracing_function = new_function
        return cls

    return decorator


def get_custom(node: fx.Node) -> "CustomOp":
    """Get the corresponding CustomOp for a given fx.Node."""
    if isinstance(node, CustomOp):
        print("Careful! You passed a custom op where an fx.Node was required.")
        return node
    if not isinstance(node, fx.Node):
        raise ValueError(f"Expected an fx.Node but got {type(node)}")

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

    def add_to_graph(self, region_graph: RegionGraph, type: Any = None) -> fx.Node:
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
        if type is None:
            get_custom(self.fx_node).infer_type()
        else:
            self.fx_node.type = type
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
        self,
        new_name: Optional[str] = None,
        new_graph: Optional[fx.Graph] = None,
        arg_transform: Optional[Callable[[Any], Any]] = lambda x: x,
        anchor: Optional[fx.Node] = None,
    ) -> Self:
        """Returns a duplicate of this node."""
        graph = new_graph
        if new_graph is None:
            graph = self.graph
            if anchor is None:
                anchor = self.fx_node
            graph.inserting_after(anchor)
        new_node = graph.node_copy(self.fx_node, arg_transform=arg_transform)
        new_node.tkw_op = self
        new_node.tkw_op_name = self.tkw_op_name
        if hasattr(self.fx_node, "index"):
            new_node.index = copy.deepcopy(self.fx_node.index)
        if new_name:
            new_node.name = new_name
        if hasattr(self.fx_node, "vector_shapes"):
            new_node.vector_shapes = self.fx_node.vector_shapes
        if hasattr(self.fx_node, "reduction_dim"):
            new_node.reduction_dim = self.fx_node.reduction_dim
        return get_custom(new_node)

    def replace_all_uses_with(self, new_node: CustomOp | fx.Node):
        """Replace all uses of the current node with the new node."""
        if isinstance(new_node, CustomOp):
            new_node = new_node.fx_node
        self.fx_node.replace_all_uses_with(new_node)

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
            assert all(
                isinstance(v, IndexSequence) for v in value.values()
            ), f"Index must be a dict with values of type IndexSequence"
            self.fx_node.index = {}
            for dim, key in value.items():
                self.fx_node.index[dim] = key
        elif isinstance(value, list):
            self.fx_node.index = value
        else:
            raise ValueError("Index must be a dict")

    @property
    def rrt(self):
        if hasattr(self.fx_node, "rrt"):
            return self.fx_node.rrt

    @rrt.setter
    def rrt(self, value):
        if not isinstance(value, np.ndarray):
            raise ValueError("RRT must be a numpy array")
        self.fx_node.rrt = value

    @property
    def scheduling_parameters(self):
        if hasattr(self.fx_node, "scheduling_parameters"):
            return self.fx_node.scheduling_parameters

    @scheduling_parameters.setter
    def scheduling_parameters(self, value: Any):
        if not isinstance(value, dict):
            raise ValueError("Scheduling parameters must be a dict")
        self.fx_node.scheduling_parameters = value

    @property
    def expanded_dims(self) -> dict[IndexSymbol, int]:
        """
        During expansion each node is expanded along its indexing dimensions.
        The expanded_dims property stores the dimensions along which the node
        has been expanded as well as the scaling along that dimension.

        For example, a node with indexing dimensions [M, N] with
        dimensional scaling {M: 2, N: 2}, will be expanded to 4 nodes,
        with each expanded node mapping to the following expanded_dims
        {M: 0, N: 0}, {M: 0, N: 1}, {M: 1, N: 0}, {M: 1, N: 1}.
        """
        if hasattr(self.fx_node, "expanded_dims"):
            return self.fx_node.expanded_dims
        return None

    @expanded_dims.setter
    def expanded_dims(self, value: dict[IndexSymbol, int]):
        if not isinstance(value, dict):
            raise ValueError("Expanded dims must be a dict")
        self.fx_node.expanded_dims = value

    @property
    def anchor(self) -> fx.Node:
        """
        The anchor is a node that provides information to the node
        such as vector_shapes, indexing information etc.
        """
        if hasattr(self.fx_node, "anchor"):
            return self.fx_node.anchor
        return None

    @anchor.setter
    def anchor(self, value: fx.Node):
        self.fx_node.anchor = value

    @property
    def vector_shapes(self) -> dict[IndexSymbol, int]:
        if hasattr(self.fx_node, "vector_shapes"):
            return self.fx_node.vector_shapes
        return None

    @vector_shapes.setter
    def vector_shapes(self, value: dict[IndexSymbol, int]):
        self.fx_node.vector_shapes = value

    @property
    def type(self) -> Any:
        if hasattr(self.fx_node, "type"):
            return self.fx_node.type
        return None

    @type.setter
    def type(self, value: Any):
        self.fx_node.type = value

    def infer_type(self):
        """
        Infer the type of this operator using the types
        of its arguments.
        """
        pass

    def align_index(self, constraints: list["Constraint"]) -> None:
        """
        Align index to WG/Tile sizes.

        Some ops require their index sizes to be aligned to workgroup/tile sizes.
        They should do it in this method.

        Default implementation does nothing.
        """
        pass


@define_py_op(operator.add)
@define_py_op(operator.sub)
@define_py_op(operator.mul)
@define_py_op(operator.truediv)
@define_interface_op("maximum")
@dataclass
class BinaryPyOp(CustomOp, ABC):
    """
    Represents an elementwise binary python operator.

    DTYPE requirement: lhs and rhs needs to have the same dtpye.
    Shape requirement: lhs and rhs either have same shape or
                       their shape must be broadcastable to
                       one another.
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

    def infer_type(self):
        lhs_type = get_custom(self.lhs).type
        rhs_type = get_custom(self.rhs).type
        has_same_type = has_same_custom_type(lhs_type, rhs_type)
        if has_same_type:
            self.type = lhs_type
            return
        lhs_dim_set = set(lhs_type.symbolic_shape)
        rhs_dim_set = set(rhs_type.symbolic_shape)
        if lhs_dim_set.isdisjoint(rhs_dim_set):
            raise ValueError(
                "BinaryPyOp requires lhs and rhs shape to be at least broadcastable."
            )
        broadcasted_type = lhs_type if lhs_dim_set > rhs_dim_set else rhs_type
        self.type = broadcasted_type


@define_interface_op("exp2")
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

    def infer_type(self):
        src_type = get_custom(self.arg).type
        self.type = src_type


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

    def parent_op(self):
        return get_custom(self.graph.parent_op)

    def get_iter_idx(self):
        src_reduction = self.parent_op()
        return src_reduction.iter_args(self.graph).index(self.fx_node)


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


@define_op("shared_memory_barrier")
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
        while next_node.next.op != "root" and not found_dst:
            next_node, found_dst = next_node.next, next_node == dst
        return found_dst


@define_op("scheduling_barrier")
@dataclass
class SchedulingBarrier(CustomOp):
    """
    Represents a scheduling barrier in the graph.
    Takes in a list of operations that are allowed to cross
    the barrier.
    """

    operations: list[Operation]


@define_op("scheduling_group_barrier")
@dataclass
class SchedulingGroupBarrier(CustomOp):
    """
    Represents a scheduling group barrier in the graph.
    The scheduling group barrier defines scheduling groups.
    Each scheduling group contains different instructions in a specific order.
    The sync_id identifies scheduling groups that need to be aware of each other.
    """

    instructions: dict[Operation, int]
    sync_id: int


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

    def infer_type(self):
        self.type = self.acc_type

    def operand_index(
        self, operand_map: dict[IndexSymbol, int], shape: list[IndexExpr]
    ) -> dict[IndexSymbol, IndexSequence]:
        indices: dict[IndexSymbol, IndexSequence] = {}
        for dim in shape:
            indices[dim] = self.index[dim].subs(operand_map)
        return indices

    @property
    def lhs_index(self) -> dict[IndexSymbol, IndexSequence]:
        operand_map = {MMA_LHS: 1, MMA_RHS: 0, MMA_ACC: 0}
        return self.operand_index(operand_map, self.lhs_type.symbolic_shape)

    @property
    def rhs_index(self) -> dict[IndexSymbol, IndexSequence]:
        operand_map = {MMA_LHS: 0, MMA_RHS: 1, MMA_ACC: 0}
        return self.operand_index(operand_map, self.rhs_type.symbolic_shape)

    @property
    def acc_index(self) -> dict[IndexSymbol, IndexSequence]:
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

    def align_index(self, constraints: list["Constraint"]) -> None:
        # Local import to break circular dep.
        from ..wave.utils import align_index_vars

        self.index = align_index_vars(self.index, constraints)

    @property
    def reduction_dim(self) -> IndexSymbol:
        if hasattr(self.fx_node, "reduction_dim"):
            return self.fx_node.reduction_dim

    @reduction_dim.setter
    def reduction_dim(self, value: IndexSymbol):
        self.fx_node.reduction_dim = value


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
        return list(self.memory_type.symbolic_shape)

    def infer_type(self):
        dtype = self.memory_type.dtype
        self.type = Register[*self.indexing_dims, dtype]

    @property
    def memory_type(self) -> "Memory":
        return get_custom(self.memory).type

    @property
    def write_dependency(self) -> fx.Node:
        return self._write_dependency

    @write_dependency.setter
    def write_dependency(self, value: fx.Node):
        self.update_arg(len(self.fx_node.args) - 1, value)

    def align_index(self, constraints: list["Constraint"]) -> None:
        # Local import to break circular dep.
        from ..wave.utils import align_index_vars, is_shared_mem_access

        if is_shared_mem_access(self):
            self.index = align_index_vars(self.index, constraints)


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
            graph.subgraphs[subgraph_name].parent_op = node.fx_node.node
            return node.fx_node

        return wrapper

    def get_root_graph(self):
        """
        Return the "root"/outermost layer of our computation graph.
        This is done by iteratively accessing parent_graph of current
        graph. This is done until we find the "root" graph who
        will have "subgraph" attribute.
        """
        cur_graph = self.graph
        while not hasattr(cur_graph, "subgraphs"):
            if not hasattr(cur_graph, "parent_op"):
                raise ValueError("All subgraphs should have parent_op")
            cur_graph = cur_graph.parent_op.graph
        return cur_graph

    @property
    def indexing_dims(self) -> list[IndexSymbol] | list[list[IndexSymbol]]:
        expand_dims: list[IndexSymbol] = []
        return_node = [
            nested_node
            for nested_node in self.graph.subgraphs[self.subgraph_name].nodes
            if isinstance(get_custom(nested_node), Output)
        ]
        assert len(return_node) == 1
        return_vals = get_custom(return_node[0]).return_vals[0]
        if not isinstance(return_vals, Sequence):
            return_vals = [return_vals]
        for return_val in return_vals:
            return_dims = get_custom(return_val).indexing_dims
            reduced_dims = [dims for dims in return_dims if dims != self.axis]
            expand_dims.append(reduced_dims)
        if len(expand_dims) == 1:
            expand_dims = expand_dims[0]
        return expand_dims

    def iter_args(self, graph: fx.Graph) -> list[fx.Node]:
        iter_args = []
        for nested_node in graph.nodes:
            custom = get_custom(nested_node)
            if isinstance(custom, IterArg):
                iter_args.append(nested_node)
        return iter_args

    def captured_vars(self, graph: fx.Graph) -> list[fx.Node]:
        """
        Nodes that are placeholders and are not iter args are captured vars.
        """
        captured_vars = []
        for nested_node in graph.nodes:
            custom = get_custom(nested_node)
            if isinstance(custom, Placeholder) and not isinstance(custom, IterArg):
                captured_vars.append(nested_node)
        return captured_vars

    def infer_type(self):
        res_types = [get_custom(x).type for x in self.init_args]
        if len(res_types) == 1:
            res_types = res_types[0]
        self.type = res_types

    def outputs(self, graph: fx.Graph) -> list[fx.Node]:
        for node in graph.nodes:
            if isinstance(get_custom(node), Output):
                return get_custom(node).return_vals[0]

    @property
    def index(self) -> list[dict[IndexSymbol, IndexSequence]]:
        for node in self.get_root_graph().subgraphs[self.subgraph_name].nodes:
            if isinstance(output := get_custom(node), Output):
                return_vals = output.return_vals[0]
                return (
                    [
                        (
                            get_custom(val).acc_index
                            if isinstance(get_custom(val), MMA)
                            else val.index
                        )
                        for val in return_vals
                    ]
                    if isinstance(return_vals, (Sequence))
                    else return_vals.index
                )

    @index.setter
    def index(self, value: Any):
        CustomOp.index.fset(self, value)

    @property
    def count(self) -> int:
        if hasattr(self.fx_node, "count"):
            return self.fx_node.count
        return None

    @count.setter
    def count(self, value: int):
        self.fx_node.count = value


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
        return list(self.memory_type.symbolic_shape)

    def infer_type(self):
        address_space = self.memory_type.address_space
        dtype = self.memory_type.dtype
        self.type = Memory[*self.indexing_dims, address_space, dtype]

    @property
    def memory_type(self) -> "Memory":
        return get_custom(self.memory).type

    @property
    def register_type(self) -> "Register":
        return get_custom(self.register_).type

    @property
    def register_index(self) -> dict[IndexSymbol, IndexSequence]:
        custom = get_custom(self.register_)
        return custom.index

    def align_index(self, constraints: list["Constraint"]) -> None:
        # Local import to break circular dep.
        from ..wave.utils import align_index_vars, is_shared_mem_access

        if is_shared_mem_access(self):
            self.index = align_index_vars(self.index, constraints)


@define_py_op(operator.getitem)
@define_op("get_result")
@dataclass
class GetResult(CustomOp):
    value: fx.Node
    res_idx: int

    def infer_type(self):
        src_type = get_custom(self.value).type
        if isinstance(src_type, list):
            self.type = src_type[self.res_idx]
        else:
            self.type = src_type

    @property
    def indexing_dims(self) -> list[IndexExpr]:
        has_multiple_value = lambda x: all(isinstance(el, list) for el in x)
        is_valid_indexing_dim = lambda x: isinstance(src_indexing, list) and all(
            isinstance(el, IndexExpr) for el in x
        )
        src_indexing = get_custom(self.value).indexing_dims
        if has_multiple_value(src_indexing):
            assert self.res_idx <= len(src_indexing) - 1
            src_indexing = src_indexing[self.res_idx]
        assert is_valid_indexing_dim(src_indexing)
        return src_indexing

    @property
    def index(self) -> dict[IndexSymbol, IndexSequence]:
        custom = get_custom(self.value)
        custom_index = custom.index
        if custom_index is None:
            return None
        if not isinstance(custom, Reduction):
            return custom.index
        assert isinstance(custom_index, Sequence) and self.res_idx < len(
            custom.indexing_dims
        )
        return custom.index[self.res_idx]

    @index.setter
    def index(self, value: dict[IndexSymbol, IndexSequence]):
        CustomOp.index.fset(self, value)


@define_op("extract")
@dataclass
class Extract(CustomOp):
    """
    Op Rationale:

    Extract is an op used to represent extracting of
    a scalar from TKW's 1-D vector on the specified index.

    This can also be viewed as indexing/slicing on the fastest
    dimension. Hence, the semantic of this op is designed to
    see itself as a reduction on the indexed/fastest dimension.
    """

    register_: fx.Proxy
    offset: IndexExpr | int

    def infer_type(self):
        # Intuition here is we are trying to extract an element
        # from fastest dim => we reduce the fastest dim.
        src_type = get_custom(self.register_).type
        # Return itself if just 0-D/1-D symbolic.
        if len(src_type.symbolic_shape) <= 1:
            self.type = src_type
            return

        # Typically fastest dim is the last dimension,
        # If non-unit dim exists => non-unit dim is fastest dim.
        non_unit_dim = [k for k, v in self.register_.index.items() if v.size != 1]
        if len(non_unit_dim) > 1:
            raise NotImplementedError(
                f"NYI: Extract only support 1 non-unit dim, but found: {len(non_unit_dim)}"
            )
        dst_shape = list(src_type.symbolic_shape)
        dim_to_remove = dst_shape[-1] if not non_unit_dim else non_unit_dim[0]
        dst_shape.remove(dim_to_remove)
        dst_type = Register[*dst_shape, src_type.dtype]
        self.type = dst_type


@define_op("extract_slice")
@dataclass
class ExtractSlice(CustomOp):
    register_: fx.Proxy
    offset: tuple[IndexExpr]
    size: tuple[IndexExpr]
    stride: tuple[IndexExpr]

    @property
    def type(self) -> "Register":
        return get_custom(self.register_).type


@define_op("broadcast")
@dataclass
class Broadcast(CustomOp, ABC):
    """
    Represents a Broadcast operation.

    arg: Source tensor/value to broadcast
    target_shape: symbolic target broadcast shape.
    """

    arg: fx.Node
    target_type: Sequence[IndexSymbol] = None

    @property
    def target_shape(self):
        return self.target_type.symbolic_shape

    @property
    def indexing_dims(self) -> list[IndexSymbol]:
        return self.target_shape

    @property
    def type(self) -> Memory:
        src_dtype = get_custom(self.arg).type.dtype
        dst_type = Register[*self.target_shape, src_dtype]
        return dst_type


@define_interface_op("max")
@define_interface_op("sum")
@dataclass
class ReduceOp(CustomOp, ABC):
    """
    Represents a Reduce computation.

    arg: Source tensor/value to reduce
    init: init/accumulator for reducte
    dim: which symbolic dim to reduce.
    """

    arg: fx.Node | list[fx.Node]
    init: fx.Node = None
    dim: Optional[Any] = None

    @property
    def indexing_dims(self) -> list[IndexSymbol]:
        # Local import to break circular dep.
        from ..wave.utils import all_equal

        if isinstance(self.arg, Sequence):
            src_indexings = [get_custom(arg).indexing_dims for arg in self.arg]
            if not all_equal(src_indexings):
                raise NotImplementedError(
                    "NYI: Only support case where all inputs to ReduceOp to have same indexing dim."
                )
            src_indexing = src_indexings[0]
        else:
            src_indexing = get_custom(self.arg).indexing_dims
        dst_indexing = [dim for dim in src_indexing if dim != self.dim]
        return dst_indexing

    def infer_type(self):
        if isinstance(self.arg, Sequence):
            src_types = [get_custom(arg).type for arg in self.arg]
            ref_shape = src_types[0].symbolic_shape
            ref_dtype = src_types[0].dtype
            if not all(
                src_type.symbolic_shape == ref_shape and src_type.dtype == ref_dtype
                for src_type in src_types
            ):
                raise NotImplementedError(
                    "NYI: Only support case where all inputs to ReduceOp to have same type."
                )
            src_type = src_types[0]
        else:
            src_type = get_custom(self.arg).type
        reduced_dims = [dims for dims in src_type.symbolic_shape if dims != self.dim]
        dst_type = Register[*reduced_dims, src_type.dtype]
        self.type = dst_type

    @property
    def num_reduction_dims(self) -> int:
        if self.dim is None:
            raise NotImplementedError(
                "Currently do not support ReduceOp with no dims specified."
            )
        if isinstance(self.dim, Sequence):
            return len(self.dim)
        else:
            return 1

    @property
    def reduction_dim(self) -> IndexSymbol:
        return self.dim


# TODO: Add support for more shuffle types.
@define_op("shuffle")
@dataclass
class ShuffleOp(CustomOp):
    """
    Represents a shuffle.xor op.

    arg: value/vector to shuffle.
    offset: xor offset.
    width: xor width.
    """

    arg: fx.Node
    offset: int
    width: int

    @property
    def indexing_dims(self) -> list[IndexSymbol]:
        return get_custom(self.arg).indexing_dims

    @property
    def type(self) -> Register:
        src_type = get_custom(self.arg).type
        return src_type


@define_op("cast")
@dataclass
class CastOp(CustomOp, ABC):
    """
    Represents a cast operation.
    """

    arg: fx.Node
    dtype: DataType

    @property
    def indexing_dims(self) -> list[IndexSymbol]:
        return get_custom(self.arg).indexing_dims

    def infer_type(self):
        src_shape = get_custom(self.arg).type.symbolic_shape
        self.type = Register[*src_shape, self.dtype]


@define_op("permute")
@dataclass
class Permute(CustomOp, ABC):
    """
    Represents a permute operation that
    permutes arg into the target shape.
    """

    arg: fx.Node
    target_shape: Sequence[IndexExpr]

    @property
    def indexing_dims(self) -> list[IndexExpr]:
        return self.target_shape

    def infer_type(self):
        src_type = get_custom(self.arg).type
        assert set(src_type.symbolic_shape) == set(
            self.target_shape
        ), f"Target shape {self.target_shape} must be a permutation of source shape {src_type.symbolic_shape}"
        self.type = Register[*self.target_shape, src_type.dtype]


def _to_sequence(input: Any | Sequence[Any]) -> Sequence[Any]:
    return input if isinstance(input, Sequence) else (input,)


@define_op("reshape")
@dataclass
class Reshape(CustomOp, ABC):
    """
    Represents a reshape operation that reshapes
    vectors along the same dimension.

    """

    args: fx.Node | Sequence[fx.Node]
    target_vector_shape: dict[IndexSymbol, int]

    @property
    def indexing_dims(self) -> list[IndexExpr]:
        return get_custom(_to_sequence(self.args)[0]).indexing_dims

    @property
    def type(self) -> Register:
        return get_custom(_to_sequence(self.args)[0]).type
