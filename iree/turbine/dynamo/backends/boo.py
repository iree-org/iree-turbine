from functorch.compile import make_boxed_compiler
import torch
from torch import fx
from torch._dynamo.backends.common import aot_autograd
from torch._dynamo.backends.registry import CompilerFn

from iree.turbine.kernel.boo.fusion.apply import fusion_transform
from iree.turbine.kernel.boo.fusion.schema import (
    FusionSchema,
    DEFAULT_SUPPORTED_BOO_FUSIONS,
)


def backend(
    *,
    nested_backend: str | CompilerFn = "eager",
    fusion_schema: FusionSchema = DEFAULT_SUPPORTED_BOO_FUSIONS,
):
    """
    A 'torch.compile' backend that selectively offloads operations to IREE.
    Alternatively, 'torch.compile(..., backend="iree_boo")' can be used.

    After offloading is performed, the graph is further optimized using
    'nested_backend'. Any valid 'torch.compile' backend can be specified.

    'fusion_schema' may be used to control which operations are fused and
    offloaded.
    """
    nested_backend_fn = make_boxed_compiler(
        torch._dynamo.lookup_backend(nested_backend)
    )

    def fw_compiler(model: fx.GraphModule, example_args):
        fusion_transform(model, fusion_schema=fusion_schema)
        return nested_backend_fn(model, example_args)

    return aot_autograd(
        fw_compiler=fw_compiler,
        # Skip boo compilation for the backwards pass for now.
        bw_compiler=nested_backend_fn,
    )


default_backend = backend()
inductor_backend = backend(nested_backend="inductor")
