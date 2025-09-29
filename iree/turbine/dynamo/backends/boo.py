from functorch.compile import make_boxed_func
import torch
from torch import fx
from torch._dynamo.backends.common import aot_autograd
from torch._dynamo.backends.registry import CompilerFn

from iree.turbine.kernel.boo.fusion.apply import fusion_transform
from iree.turbine.kernel.boo.fusion.schema import (
    FusionSchema,
    ReplacementSchema,
    DEFAULT_SUPPORTED_BOO_FUSIONS,
    DEFAULT_POST_FUSION_REPLACEMENTS,
    EXPERIMENTAL_SUPPORTED_BOO_FUSIONS,
    EXPERIMENTAL_POST_FUSION_REPLACEMENTS,
)


def backend(
    *,
    nested_backend: str | CompilerFn = "eager",
    fusion_schema: FusionSchema = DEFAULT_SUPPORTED_BOO_FUSIONS,
    post_fusion_replacements: ReplacementSchema = DEFAULT_POST_FUSION_REPLACEMENTS,
):
    """
    A 'torch.compile' backend that selectively offloads operations to IREE.
    Alternatively, 'torch.compile(..., backend="iree_boo")' can be used.

    After offloading is performed, the graph is further optimized using
    'nested_backend'. Any valid 'torch.compile' backend can be specified.

    'fusion_schema' may be used to control which operations are fused and
    offloaded. 'post_fusion_replacements' is applied to any fused subgraphs that
    are created.
    """
    nested_backend_fn = torch._dynamo.lookup_backend(nested_backend)

    def compiler_fn(model: fx.GraphModule, example_args):
        fusion_transform(
            model,
            fusion_schema=fusion_schema,
            post_fusion_replacements=post_fusion_replacements,
        )
        return make_boxed_func(nested_backend_fn(model, example_args))

    return aot_autograd(fw_compiler=compiler_fn)


default_backend = backend()
inductor_backend = backend(nested_backend="inductor")
experimental_backend = backend(
    fusion_schema=EXPERIMENTAL_SUPPORTED_BOO_FUSIONS,
    post_fusion_replacements=EXPERIMENTAL_POST_FUSION_REPLACEMENTS,
)
