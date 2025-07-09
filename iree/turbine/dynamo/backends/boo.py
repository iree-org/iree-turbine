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
    fusion_schema: FusionSchema = DEFAULT_SUPPORTED_BOO_FUSIONS,
    backend: str | CompilerFn = "eager",
):
    backend_fn = make_boxed_compiler(torch._dynamo.lookup_backend(backend))

    def fw_compiler(model: fx.GraphModule, example_args):
        fusion_transform(model, fusion_schema=fusion_schema)
        return backend_fn(model, example_args)

    return aot_autograd(
        fw_compiler=fw_compiler,
        # Skip boo compilation for the backwards pass for now.
        bw_compiler=backend_fn,
    )


default_backend = backend()
inductor_backend = backend(backend="inductor")
