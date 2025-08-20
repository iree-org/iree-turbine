from __future__ import annotations
from abc import ABC
from dataclasses import dataclass, field, fields
import sys
import copy
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Optional,
    Sequence,
    Type,
    TypeVar,
    final,
)
from typing_extensions import Self
import torch.fx as fx

from ..lang.tkw_types import Memory, Register, IndexMapping
from ..lang.global_symbols import *
from .._support.indexing import IndexExpr, IndexSymbol, IndexSequence
from .._support.dtype import DataType
from .._support.regions import RegionGraph
from .._support.location import FileLineColInfo, StackTraceInfo, capture_location
from .base import OpDispatcher
import numpy as np


T = TypeVar("T", bound=Type[Any])
AccT = TypeVar("AccT")
CustomOpT = TypeVar("CustomOpT", bound="CustomOp")
PlaceholderT = TypeVar("PlaceholderT", bound="Placeholder")


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
        # Some py operator has trailing "_", which needs to be removed
        # before reformatting to torch.fx.Proxy formats.
        # i.e `and_` -> `and`, `or_` -> `or`.
        fx_op_name = op_name.replace("_", "")
        if hasattr(fx.Proxy, f"__{fx_op_name}__"):
            original_handler = getattr(fx.Proxy, f"__{fx_op_name}__")

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
            setattr(fx.Proxy, f"__{fx_op_name}__", new_function)

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
        raise ValueError(f"fx.Node required but got custom op {node}")
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

    @property
    def location(self) -> Optional[FileLineColInfo | StackTraceInfo]:
        return getattr(self.fx_node, "location", None)

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
        if hasattr(self.fx_node, "subgraph_name") and self.fx_node.subgraph_name:
            vars_list.append(f"subgraph_name={self.fx_node.subgraph_name}")
        vars_str = ", ".join(vars_list)
        return f"{self.tkw_op_name}({vars_str}) type({self.fx_node.type})"

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

    def update_arg(self, idx_or_name: int | str | fx.Node, value: CustomOp | fx.Node):
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
        elif isinstance(idx_or_name, fx.Node):
            idx = self.fx_node.args.index(idx_or_name)
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

    def copy_core_attributes(self, new_node: fx.Node):
        """
        Copy core attributes from the current node to the new node.
        """
        core_attributes = [
            "index",
            "vector_shapes",
            "reduction_dim",
            "iter_idx",
            "location",
            "expanded_dims",
            "scheduling_parameters",
        ]
        for attr_name in core_attributes:
            if hasattr(self.fx_node, attr_name):
                attr = getattr(self.fx_node, attr_name)
                if attr_name == "index":
                    attr = copy.deepcopy(attr)
                setattr(new_node, attr_name, attr)

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
        self.copy_core_attributes(new_node)
        if new_name:
            new_node.name = new_name
        return get_custom(new_node)

    def replace_all_uses_with(self, new_node: CustomOp | fx.Node):
        """Replace all uses of the current node with the new node."""
        if isinstance(new_node, CustomOp):
            new_node = new_node.fx_node
        self.fx_node.replace_all_uses_with(new_node)

    def replace_all_uses_with_except(
        self, new_node: CustomOp | fx.Node, except_nodes: list[CustomOp]
    ):
        """Replace all uses of the current node with the new node except for the nodes in except_nodes."""
        for user in self.users:
            if user in except_nodes:
                continue
            indices = user.get_node_arg_index(self)
            if not isinstance(indices, Sequence):
                indices = [indices]
            for idx in indices:
                if isinstance(user.node_args[idx], Sequence):
                    sub_idx = user.node_args[idx].index(self)
                    new_nodes = [
                        (
                            user.node_args[idx][x].fx_node
                            if x != sub_idx
                            else new_node.fx_node
                        )
                        for x in range(len(user.node_args[idx]))
                    ]
                    user.update_arg(idx, new_nodes)
                else:
                    user.update_arg(idx, new_node.fx_node)

    def erase(self):
        """Erase the current node from the graph where it exists."""
        assert (
            not self.fx_node.users
        ), f"Attempting to erase {self.fx_node} which has {len(self.fx.users)} users!"
        self.graph.erase_node(self.fx_node)

    @classmethod
    def handle(cls, graph: RegionGraph, *args, **kwargs) -> fx.Node:
        node = cls(*args, **kwargs)
        node._add_proxy_to_graph(graph)
        node.fx_node.node.tkw_op = cls
        node.fx_node.node.tkw_op_name = cls.tkw_op_name
        node.fx_node.node.location = capture_location(graph.location_capture_config)
        return node.fx_node

    @property
    def name(self) -> str:
        if hasattr(self, "_name"):
            return self._name
        return self.fx_node.name

    @property
    def node_args(self) -> dict[int, Any]:
        """Returns the args to this custom op using subclasses of CustomOp if possible."""

        def propagate(n):
            custom = get_custom(n)
            if isinstance(custom, Placeholder):
                if prev := custom.get_captured_fx_node():
                    return propagate(prev)

            return custom

        custom_args = {}
        for i, arg in enumerate(self.fx_node.args):
            if isinstance(arg, fx.Node):
                custom_args[i] = propagate(arg)
            if isinstance(arg, Sequence) and all(isinstance(x, fx.Node) for x in arg):
                custom_args[i] = [propagate(x) for x in arg]
        return custom_args

    def get_node_arg_index(self, arg: CustomOp) -> Optional[CustomOp | list[CustomOp]]:
        keys = []
        for key, value in self.node_args.items():
            if isinstance(value, Sequence):
                if arg in value:
                    keys.append(key)
            elif value == arg:
                keys.append(key)
        if not keys:
            return None
        if len(keys) == 1:
            return keys[0]
        return keys

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
            ), f"Index must be a dict with values of type IndexSequence but got {value}"
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

    @property
    def has_side_effects(self) -> bool:
        return False

    @property
    def pre_expansion_id(self) -> int:
        if hasattr(self.fx_node, "pre_expansion_id"):
            return self.fx_node.pre_expansion_id
        return None

    @pre_expansion_id.setter
    def pre_expansion_id(self, value: int):
        self.fx_node.pre_expansion_id = value

    def infer_type(self):
        """
        Infer the type of this operator using the types
        of its arguments.
        """
        pass

    def transform_index_backwards(
        self, index: dict[IndexSymbol, IndexSequence], arg: fx.Node
    ) -> dict[IndexSymbol, IndexSequence]:
        """
        Transform the index of the node when propagating index backwards, i.e.
        from node to its arguments.
        """
        return index

    def transform_index(
        self, index: dict[IndexSymbol, IndexSequence]
    ) -> dict[IndexSymbol, IndexSequence]:
        """
        Transform the index of the node based on the provided mapping.
        """
        return index


@dataclass
class BinaryOpBase(CustomOp, ABC):
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

    def infer_shape(self) -> Any:
        lhs_type = get_custom(self.lhs).type
        rhs_type = get_custom(self.rhs).type
        if isinstance(lhs_type, DataType) and isinstance(rhs_type, DataType):
            has_same_type = True
        else:
            has_same_type = has_same_custom_type(lhs_type, rhs_type)
        if has_same_type:
            return lhs_type.symbolic_shape

        lhs_dim_set = set(lhs_type.symbolic_shape)
        rhs_dim_set = set(rhs_type.symbolic_shape)
        if lhs_dim_set.isdisjoint(rhs_dim_set):
            raise ValueError(
                "BinaryPyOp requires lhs and rhs shape to be at least broadcastable."
                f" got {lhs_type.symbolic_shape} vs {rhs_type.symbolic_shape}"
            )

        # TODO: this logic looks suspicious. Specifically, there's no check that
        # rhs_dim_set subsumes lhs_dim_set, they may partially overlap.
        broadcasted_type = lhs_type if lhs_dim_set > rhs_dim_set else rhs_type
        return broadcasted_type.symbolic_shape


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
        return f"unknown: {self.fx_node.name}({vars_str}) type({self.fx_node.type})"


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

    @property
    def has_side_effects(self) -> bool:
        return True


@define_op("register")
@dataclass
class NewRegister(CustomOp):
    shape: tuple[IndexExpr, ...]
    dtype: DataType
    value: float

    @property
    def indexing_dims(self) -> list[IndexSymbol]:
        return list(self.shape)

    def infer_type(self):
        self.type = Register[(*self.shape, self.dtype)]


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
        return f"{self.tkw_op_name}({vars_str}) type({self.fx_node.type})"

    def erase(self):
        """Erase the current node from the graph where it exists."""

        super().erase()
        if not hasattr(self.graph, "parent_op"):
            return

        parent = self.graph.parent_op
        custom = get_custom(parent)
        if not isinstance(custom, NestedRegionOp):
            return

        # Cleanup dead captures
        subgraph = custom.get_root_graph().subgraphs[custom.subgraph_name]
        live_captures = []
        for var in custom.implicit_captures:
            if custom.get_captured_fx_node(subgraph, var):
                live_captures.append(var)

        custom.update_arg("implicit_captures", live_captures)

    @property
    def indexing_dims(self) -> list[IndexSymbol]:
        if not hasattr(self._type, "symbolic_shape"):
            return []
        return list(self._type.symbolic_shape) if self._type else []

    def get_captured_fx_node(self) -> Optional[fx.Node]:
        return self.fx_node.meta.get("lifted", None)

    def infer_type(self):
        self.fx_node.type = self._type

    @property
    def index(self) -> list[dict[IndexSymbol, IndexSequence]]:
        var = self.get_captured_fx_node()
        if var is not None:
            return get_custom(var).index

        if hasattr(self.fx_node, "index"):
            return self.fx_node.index

        return None

    @index.setter
    def index(self, value: Any):
        var = self.get_captured_fx_node()
        if var is None:
            CustomOp.index.fset(self, value)
            return

        get_custom(var).index = value


class IterArg(Placeholder):
    """
    Represents a specific placeholder node in the graph that is an iter arg of
    a reduction node.
    """

    def parent_op(self):
        return get_custom(self.graph.parent_op)

    @property
    def iter_idx(self):
        if hasattr(self.fx_node, "iter_idx"):
            return self.fx_node.iter_idx
        return None

    @iter_idx.setter
    def iter_idx(self, value):
        self.fx_node.iter_idx = value

    def infer_type(self):
        parent_op = self.parent_op()
        init_args = parent_op.init_args
        self.type = init_args[self.iter_idx].type


@dataclass
class Read(CustomOp):

    memory: fx.Proxy
    elements_per_thread: Optional[Any] = None
    mapping: Optional[IndexMapping] = None
    mapping_dynamic_vals: tuple[fx.Node, ...] = ()
    bounds: Optional[dict[IndexSymbol, IndexExpr]] = None
    _write_dependency: Optional[list[fx.Node]] = None

    @property
    def memory_type(self) -> "Memory":
        return get_custom(self.memory).type

    @property
    def write_dependency(self) -> fx.Node:
        return self._write_dependency

    @write_dependency.setter
    def write_dependency(self, value: fx.Node):
        self.update_arg(len(self.fx_node.args) - 1, value)

    def transform_index_backwards(
        self, index: dict[IndexSymbol, IndexSequence], arg: fx.Node
    ) -> dict[IndexSymbol, IndexSequence]:
        """
        Propagate index backwards.

        Dynamic values potentially can have non-identity mapping, so we need
        to update index when walking from the node to dyn val arguments.

        E.g. if `index` is $idx and dynamic_val_mappings={N: j // ELEMS_PER_THREAD}
        resulted arg index will be $idx // ELEMS_PER_THREAD.
        """
        if arg in self.mapping_dynamic_vals:
            assert self.mapping.is_output_identity()
            i = self.mapping_dynamic_vals.index(arg)
            iters = self.mapping.iters
            mapping = self.mapping.dynamic_val_mappings[i]

            # This logic assumes that the output mapping is identity.
            subs = {
                k: index[v] for k, v in zip(iters, self.mapping.output_mapping.keys())
            }
            return {
                k: IndexSequence.from_expr(mapping[k], subs)
                for k in get_custom(arg).type.symbolic_shape
                if k in mapping
            }

        return index

    def get_derived_indices(
        self,
    ) -> list[tuple[dict[IndexSymbol, IndexSequence], fx.Node]]:
        def transform_idx(arg):
            # Treat zero index as 'not-set' and does't propagate it.
            # TODO: `set_thread_independent_index` currently blindly sets zero
            # index to all dims which are not participating in constraints, we
            # need to refactor `index_sequence_analysis` into proper dataflow
            # analysis.
            return {
                k: v
                for k, v in self.transform_index_backwards(self.index, arg).items()
                if v.start != 0
            }

        return [(arg, transform_idx(arg)) for arg in self.mapping_dynamic_vals]

    def has_identity_mapping(self) -> bool:
        """Check if mapping between input memory and output register is identity."""
        mapping = self.mapping
        if mapping is None:
            return True

        mem_shape = get_custom(self.memory).type.symbolic_shape
        if mapping.is_identity() and mapping.input_shape == mem_shape:
            return True

        return False


class NestedRegionOp(CustomOp):
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

    def get_outer_node(self, outer_node: fx.Node) -> fx.Node:
        while "lifted" in outer_node.meta:
            outer_node = outer_node.meta["lifted"]
        return outer_node

    def get_captured_fx_node(
        self, graph: fx.Graph, outer_node: fx.Node
    ) -> Optional[fx.Node]:
        outer_node = self.get_outer_node(outer_node)
        for var in self.captured_vars(graph):
            custom = get_custom(var)
            if custom.get_captured_fx_node() == outer_node:
                return var

        return None

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


@dataclass
class Write(CustomOp):
    register_: fx.Proxy
    memory: fx.Proxy
    elements_per_thread: Optional[Any] = None
    mapping: Optional[IndexMapping] = None
    mapping_dynamic_vals: tuple[fx.Node, ...] = ()
    bounds: Optional[dict[IndexSymbol, IndexExpr]] = None

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

    def transform_index_backwards(
        self, index: dict[IndexSymbol, IndexSequence], arg: fx.Node
    ) -> dict[IndexSymbol, IndexSequence]:
        """
        Propagate index backwards.

        Dynamic values potentially can have non-identity mapping, so we need
        to update index when walking from the node to dyn val arguments.

        E.g. if `index` is $idx and dynamic_val_mappings={N: j // ELEMS_PER_THREAD}
        resulted arg index will be $idx // ELEMS_PER_THREAD.
        """
        if arg in self.mapping_dynamic_vals:
            assert self.mapping.is_input_identity()
            i = self.mapping_dynamic_vals.index(arg)
            iters = self.mapping.iters
            mapping = self.mapping.dynamic_val_mappings[i]

            # This logic assumes that the input mapping is identity.
            subs = {
                k: index[v] for k, v in zip(iters, self.mapping.input_mapping.keys())
            }
            return {
                k: IndexSequence.from_expr(mapping[k], subs)
                for k in arg.type.symbolic_shape
                if k in mapping
            }

        return index

    def get_derived_indices(
        self,
    ) -> list[tuple[dict[IndexSymbol, IndexSequence], fx.Node]]:
        def transform_idx(arg):
            return {
                k: v
                for k, v in self.transform_index_backwards(self.index, arg).items()
                if v.start != 0
            }

        return [(arg, transform_idx(arg)) for arg in self.mapping_dynamic_vals]

    def has_identity_mapping(self) -> bool:
        """Check if mapping between input register and output memory is identity."""
        mapping = self.mapping
        if mapping is None:
            return True

        mem_shape = get_custom(self.memory).type.symbolic_shape
        if mapping.is_identity() and mapping.output_shape == mem_shape:
            return True

        return False
