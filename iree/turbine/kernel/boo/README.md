The BOO (Bag of Ops) project is intended to provide a python interface for fusing and launching standalone IREE kernels in a pytorch model.

Currently, convolution kernels are supported with bias and activation fusions.

## Environment Variables

There are a few environment variables that control BOO behavior:

- `BOO_CACHE_ON=<0 or 1>`: whether to cache kernel artifacts for boo kernels (default = 1).
- `BOO_CACHE_DIR=<path to cache dir>`: where to store kernel artifacts (default = `~/.cache/turbine_kernels/boo/`).
- `BOO_USE_BACKWARD_KERNELS=<0 or 1>`: whether to use our backward kernels for boo ops (default = 0).

And several other variables are useful for debugging:

- `TURBINE_LOG_LEVEL=DEBUG`: enable debug output from Turbine.

## Usage:

### Enabling supported fusions in a model:

The best way to enable boo kernels in a model is to use one of our `torch.compile` backends:

- `"iree_boo"` : Uses IREE to identify and generate fused kernels based on `DEFAULT_SUPPORTED_BOO_FUSIONS` in [schema.py](./fusion/schema.py)
- `"iree_boo_inductor"` : Applies the same fusions as `iree_boo`, but additionally applies the `inductor` backend to the rest of the model.

```python
import torch

model : torch.nn.Module = ...

# To just use boo fusions:
boo_model = torch.compile(model, dynamic=False, backend="iree_boo")
# To use boo fusions and inductor:
boo_inductor_model = torch.compile(model, dynamic=False, backend="iree_boo_inductor")
```

To customize the backend further, the boo backend can be called directly.

```python
import torch

import iree.turbine.dynamo.backends.boo as boo
from iree.turbine.kernel.boo.fusion import FusionSchema, ReplacementSchema

custom_fusion_schema : FusionSchema = ...
custom_replacement_schema : ReplacementSchema = ...
customized_backend = boo.backend(nested_backend="some_other_backend", fusion_schema=custom_fusion_schema, post_fusion_replacements=custom_replacement_schema)

model : torch.nn.Module = ...

customized_boo_model = torch.compile(model, dynamic=False, backend=customized_backend)
```


### Launch a single convolution:

```python
import torch
from iree.turbine.kernel.boo.ops import boo_conv

input_shape = ...
weight_shape = ...
dtype = ...

x = torch.randn(input_shape, dtype=dtype, device="cuda:0")
w = torch.randn(weight_shape, dtype=dtype, device="cuda:0")
b = None # optionally fuse with a bias-add

y = boo_conv(x,w,b) # can also specify stride, dilation, padding, groups, and layouts (e.g., "NHWC")
```

### Use a `BooConv2d` module:

```python
from iree.turbine.kernel.boo.modeling import BooConv2d

conv2d = BooConv2d(...) # usual Conv2d args
```

### Benchmarking BOO ops:

See [driver documentation](./driver/README.md).
