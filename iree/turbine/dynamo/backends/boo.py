from typing import Mapping, Callable
from functorch.compile import make_boxed_func
import torch
from torch import fx
from torch.fx.node import map_aggregate
from torch._guards import detect_fake_mode
from torch.fx.passes.shape_prop import _extract_tensor_metadata
from torch.fx.passes.fake_tensor_prop import FakeTensorProp
from torch._dynamo.backends.common import aot_autograd
from torch._dynamo.backends.registry import CompilerFn

from iree.turbine.kernel.boo.fusion.apply import fusion_transform
from iree.turbine.kernel.boo.fusion.schema import (
    FusionSchema,
    DEFAULT_SUPPORTED_BOO_FUSIONS,
    EXPERIMENTAL_SUPPORTED_BOO_FUSIONS,
)
from iree.turbine.kernel.boo.fusion.replacement import DEFAULT_BOO_OP_DECOMPOSITIONS


class ValAndTensorMetaProp(FakeTensorProp):
    """This fx.Interpreter applies `FakeTensorProp`, which updates `node.meta['val']`.

    We extend this interpreter to additionally update `node.meta['tensor_meta']`.
    """

    def run_node(self, n: fx.Node):
        result = super().run_node(n)

        found_tensor = False

        def extract_tensor_meta(obj):
            if isinstance(obj, torch.Tensor):
                nonlocal found_tensor
                found_tensor = True
                return _extract_tensor_metadata(obj)
            else:
                return obj

        tensor_meta = map_aggregate(result, extract_tensor_meta)
        if found_tensor:
            n.meta["tensor_meta"] = tensor_meta

        return result


def backend(
    *,
    nested_backend: str | CompilerFn = "eager",
    fusion_schema: FusionSchema = DEFAULT_SUPPORTED_BOO_FUSIONS,
    decomposition_table: Mapping[
        torch._ops.OperatorBase, Callable
    ] = DEFAULT_BOO_OP_DECOMPOSITIONS,
):
    """
    A 'torch.compile' backend that selectively offloads operations to IREE.
    Alternatively, 'torch.compile(..., backend="iree_boo")' can be used.

    After offloading is performed, the graph is further optimized using
    'nested_backend'. Any valid 'torch.compile' backend can be specified.

    'fusion_schema' may be used to control which operations are fused and
    offloaded. The `decomposition_table` controls which decompositons are
    applied to the IREE handled subgraphs.
    """
    nested_backend_fn = torch._dynamo.lookup_backend(nested_backend)

    def compiler_fn(model: fx.GraphModule, example_args):
        # Apply BOO fused subgraphs through IREE compiler/runtime.
        fusion_transform(
            model,
            fusion_schema=fusion_schema,
            decomposition_table=decomposition_table,
        )
        # torch.compile should provide us with fake example_args.
        fake_mode = detect_fake_mode(example_args)
        assert fake_mode is not None
        # Re-run fake prop to populate any stride metadata we modify during replacement.
        ValAndTensorMetaProp(model, fake_mode).propagate_dont_convert_inputs(
            *example_args
        )
        return make_boxed_func(nested_backend_fn(model, example_args))

    return aot_autograd(fw_compiler=compiler_fn)


default_backend = backend()
inductor_backend = backend(nested_backend="inductor")
experimental_backend = backend(
    fusion_schema=EXPERIMENTAL_SUPPORTED_BOO_FUSIONS,
)
