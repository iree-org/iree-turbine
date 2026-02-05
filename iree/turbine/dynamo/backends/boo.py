import warnings
from functorch.compile import make_boxed_func
import torch
from torch import fx
from torch._dynamo.backends.common import aot_autograd
from torch._dynamo.backends.registry import CompilerFn
from torch._dynamo.testing import EagerAndRecordGraphs

from iree.turbine.kernel.boo.fusion.apply import fusion_transform
from iree.turbine.kernel.boo.fusion.schema import (
    FusionSchema,
    ReplacementSchema,
    DEFAULT_SUPPORTED_BOO_FUSIONS,
    DEFAULT_POST_FUSION_REPLACEMENTS,
    DEFAULT_POST_DECOMPOSITION_REPLACEMENTS,
    EXPERIMENTAL_SUPPORTED_BOO_FUSIONS,
    EXPERIMENTAL_POST_FUSION_REPLACEMENTS,
)


def _make_inductor_compiler(
    fusion_schema,
    post_fusion_replacements,
    post_decomposition_replacements,
    is_backward=False,
):
    """Create a compiler function that uses inductor's compile_fx_inner directly.

    This avoids double aot_autograd wrapping when BOO is used with the inductor
    nested backend, which caused DDP compatibility issues (fakify_first_call
    conflicts, metadata overwrites, stride info loss).
    """
    from torch._inductor.compile_fx import compile_fx_inner

    def compiler_fn(model: fx.GraphModule, example_inputs):
        fusion_transform(
            model,
            fusion_schema=fusion_schema,
            post_fusion_replacements=post_fusion_replacements,
            post_decomposition_replacements=post_decomposition_replacements,
        )
        return compile_fx_inner(model, example_inputs, is_backward=is_backward)

    return compiler_fn


def backend(
    *,
    nested_backend: str | CompilerFn = "eager",
    fusion_schema: FusionSchema = DEFAULT_SUPPORTED_BOO_FUSIONS,
    post_fusion_replacements: ReplacementSchema = DEFAULT_POST_FUSION_REPLACEMENTS,
    post_decomposition_replacements: ReplacementSchema = DEFAULT_POST_DECOMPOSITION_REPLACEMENTS,
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
    use_inductor_inner = nested_backend == "inductor"

    if not use_inductor_inner:
        if nested_backend not in {"eager"} and not isinstance(
            nested_backend, EagerAndRecordGraphs
        ):
            warnings.warn(
                f"Provided nested_backend is not guaranteed to work properly "
                f"with BOO (e.g., in distributed contexts): {nested_backend}."
            )
        nested_backend_fn = torch._dynamo.lookup_backend(nested_backend)

        def compiler_fn(model: fx.GraphModule, example_args):
            fusion_transform(
                model,
                fusion_schema=fusion_schema,
                post_fusion_replacements=post_fusion_replacements,
                post_decomposition_replacements=post_decomposition_replacements,
            )
            backend_func = nested_backend_fn(model, example_args)
            return make_boxed_func(backend_func)

        return aot_autograd(fw_compiler=compiler_fn)

    # Inductor path: use compile_fx_inner directly to avoid double aot_autograd.
    # Pass inductor's decomposition table so that ops like aten.t get decomposed
    # before reaching compile_fx_inner's GraphLowering.
    from torch._inductor.decomposition import select_decomp_table

    fw_compiler = _make_inductor_compiler(
        fusion_schema,
        post_fusion_replacements,
        post_decomposition_replacements,
        is_backward=False,
    )
    bw_compiler = _make_inductor_compiler(
        fusion_schema,
        post_fusion_replacements,
        post_decomposition_replacements,
        is_backward=True,
    )
    return aot_autograd(
        fw_compiler=fw_compiler,
        bw_compiler=bw_compiler,
        decompositions=select_decomp_table(),
    )


default_backend = backend()
inductor_backend = backend(nested_backend="inductor")
experimental_backend = backend(
    fusion_schema=EXPERIMENTAL_SUPPORTED_BOO_FUSIONS,
    post_fusion_replacements=EXPERIMENTAL_POST_FUSION_REPLACEMENTS,
)
