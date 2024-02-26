from typing import (
    Optional,
    TypeVar,
    Callable,
    Type,
    cast,
    List,
    Dict,
    Tuple,
)
import random
import contextlib

import torch.fx as fx
import torch.utils._pytree as pytree


class RegionGraph:
    def __init__(self):
        self.tracers: List["SubgraphTracer"] = []
        self.subgraphs: Dict[str, fx.Graph] = dict()
        self.inner_freevars: Dict[fx.Graph, List[fx.Proxy]] = dict()

    @property
    def root_tracer(self) -> "SubgraphTracer":
        return self.tracers[0]

    @property
    def current_tracer(self) -> "SubgraphTracer":
        return self.tracers[-1]

    def create_proxy(self, *args, **kwargs):
        return self.current_tracer.create_proxy(*args, **kwargs)

    def create_node(self, *args, **kwargs):
        return self.current_tracer.create_node(*args, **kwargs)

    def create_arg(self, *args, **kwargs):
        return self.current_tracer.create_arg(*args, **kwargs)

    def new_subtracer(
        self, region_graph: "RegionGraph", parent: Optional["SubgraphTracer"] = None
    ) -> "SubgraphTracer":
        ...

    ### ========================================================================
    ### Subgraph Tracing
    ### ========================================================================
    def add_subgraph(
        self, name: str, graph: fx.Graph, inner_freevars: List[fx.Proxy]
    ) -> str:
        i = 0
        while True:
            candidate_name = f"{name}_{i}"
            i += 1
            if candidate_name not in self.subgraphs:
                self.subgraphs[candidate_name] = graph
                self.inner_freevars[graph] = inner_freevars
                return candidate_name

    @contextlib.contextmanager
    def subtracer(self):
        if self.tracers:
            new_tracer = self.new_subtracer(self, self.current_tracer)
        else:
            new_tracer = self.new_subtracer(self)
        self.tracers.append(new_tracer)
        yield new_tracer
        self.tracers.pop()

    def __str__(self):
        out = ""
        for name, subgraph in self.subgraphs.items():
            out += f"{name}:"
            out += str(subgraph)
            out += "\n"
        return out


class SubgraphTracer(fx.Tracer):
    def __init__(
        self, region_graph: RegionGraph, parent: Optional["SubgraphTracer"] = None
    ):
        super().__init__()
        self.graph = fx.Graph()
        self.region_graph = region_graph
        self.parent = parent
        self.lifted_freevars: Dict[fx.Proxy, fx.Proxy] = {}

    def trace(self, *args, **kwargs) -> Tuple[str, List[fx.Proxy]]:
        traced = super().trace(*args, **kwargs)
        inner_freevars = list(self.lifted_freevars.values())
        implicit_capture = list(self.lifted_freevars.keys())
        subgraph_name = self.region_graph.add_subgraph("region", traced, inner_freevars)
        return subgraph_name, implicit_capture

    def _create_graph_input(self, name: str, type_expr=None) -> fx.Proxy:
        proxy = self.create_proxy("placeholder", name, (), {}, type_expr=type_expr)
        # Can use this to check where the freevar has been lifted from.
        proxy.node.meta["lifted"] = None
        return proxy

    def _lift_tracked_freevar_to_input(self, proxy: fx.Proxy):
        # It makes no sense for the root graph to have free variables
        assert self.parent is not None, "Cannot lift freevars to input in root tracer"

        # If the freevar has already been lifted, return the lifted version.
        if proxy in self.lifted_freevars:
            return self.lifted_freevars[proxy]

        # Otherwise, create a new input and store it.
        new_proxy = self._create_graph_input(proxy.node.name, proxy.node.type)
        self.lifted_freevars[proxy] = new_proxy

        # Propagate freevar usage upwards.
        if self.parent is not None and proxy.tracer != self.parent:
            self.parent._lift_tracked_freevar_to_input(proxy)
        return new_proxy

    def _maybe_lift_tracked_freevar_to_input(self, arg):
        """
        If arg is a free variable, then lift it to be an input.
        Returns the new lifted arg (if lifted), else the original arg.
        """
        if not isinstance(arg, fx.Proxy):
            return arg
        elif arg.tracer == self:
            return arg
        else:
            return self._lift_tracked_freevar_to_input(arg)

    def create_proxy(
        self,
        kind,
        target,
        args,
        kwargs,
        name=None,
        type_expr=None,
        proxy_factory_fn=None,
    ):
        if self.parent is not None:
            flat_args, tree_spec = pytree.tree_flatten((args, kwargs))
            new_flat_args = []
            for arg in flat_args:
                maybe_new_arg = self._maybe_lift_tracked_freevar_to_input(arg)
                new_flat_args.append(maybe_new_arg)
            args, kwargs = pytree.tree_unflatten(new_flat_args, tree_spec)

        rv = super().create_proxy(
            kind,
            target,
            args,
            kwargs,
            name,
            type_expr,
            proxy_factory_fn,
        )

        return rv
