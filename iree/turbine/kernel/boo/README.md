The BOO (Bag of Ops) project is intended to provide a python interface for fusing and launching standalone IREE kernels in a pytorch model.

Currently, convolution with bias and activation fusions and layernorm kernels are supported, in both forward and backward modes.

## Usage

Install `iree-turbine` following the instructions [here](/README.md#quick-start-for-users).
Installing a nightly version is recommended as BOO is under rapid development.

### Enabling supported fusions in a model

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

If a nested backend besided `inductor` is desired, a customized boo backend can be defined directly.

```python
import torch

import iree.turbine.dynamo.backends.boo as boo

customized_backend = boo.backend(nested_backend="some_other_backend")

model : torch.nn.Module = ...

customized_boo_model = torch.compile(model, dynamic=False, backend=customized_backend)
```

### Launch a single convolution

You can use `boo_conv` for user-friendly eager execution.

```python
import torch
from iree.turbine.kernel.boo.ops import boo_conv

input_shape = ...
weight_shape = ...
dtype = ...

# make some 3, 4, or 5d tensors
x = torch.randn(input_shape, dtype=dtype, device="cuda:0")
w = torch.randn(weight_shape, dtype=dtype, device="cuda:0")
b = None # optionally fuse with a bias-add

# perform convolution
y = boo_conv(x,w,b) # can also specify stride, dilation, padding, groups, and layouts (e.g., "NHWC")

# usual conv kwargs are also available:
y = boo_conv(x,w,stride=[1,2],dilation=[2,2], padding=[1,0])

# additional kwargs for alternative layouts also available:
y = boo_conv(x,w, shared_layout="NHWC")
y = boo_conv(x,w, input_layout="NCHW", kernel_layout="NHWC", output_layout="NCHW")
```

### Define signatures from explicit shapes

```python
import torch

from iree.turbine.kernel.boo.op_exports.conv import ConvSignature
from iree.turbine.kernel.boo.driver.launch import get_launchable

fwd_sig = ConvSignature(
    input_shape = [2,16,32,3],
    kernel_shape = [10,2,2,3],
    shared_layout="NHWC",
)

wrw_sig = ConvSignature(
    input_shape = [2,16,32,3],
    kernel_shape = [10,2,2,3],
    shared_layout="NHWC",
    # Can specify a mode "fwd" "bwd" "wrw" with:
    mode="wrw",
)

conv_fwd = get_launchable(fwd_sig)
conv_wrw = get_launchable(wrw_sig)

torch_device = torch.device("cuda:0") if torch.cuda.is_available() else None

x, w = fwd_sig.get_sample_args(device=torch_device, seed=10)

y = conv_fwd(x, w)

# get a random dLdy to back-prop.
dLdy, _ = wrw_sig.get_sample_args(device=torch_device, seed=2)
dLdw = conv_wrw(dLdy, x)
```

### Use a `BooConv2d` module directly

```python
from iree.turbine.kernel.boo.modeling import BooConv2d

conv2d = BooConv2d(...) # usual Conv2d args
```

### Get signatures from MiOpen driver commands:

The op registry is capable of locating MiOpen-compatible parsers for BOO-supported operations based on the presence of keywords such as `conv` or `layernorm` in the command.

```python

from iree.turbine.kernel.boo.driver.registry import BooOpRegistry
from iree.turbine.kernel.boo.driver.launch import get_launchable

miopen_driver_command = "convbfp16 -n 128 -c 128 -H 24 -W 48 -k 384 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1 --iter 100 --in_layout NHWC --out_layout NHWC --fil_layout NHWC"
signature = BooOpRegistry.parse_command(miopen_driver_command)

conv = get_launchable(signature)
```

## Environment Variables

There are a few environment variables that control BOO behavior:

- `BOO_CACHE_ON=<0 or 1>`: whether to cache kernel artifacts for boo kernels (default = 0).
- `BOO_CACHE_DIR=<path to cache dir>`: where to store kernel artifacts (default = `~/.cache/turbine_kernels/boo/`).
- `BOO_USE_BACKWARD_KERNELS=<0 or 1>`: whether to use our backward kernels for boo ops (default = 0).
- `BOO_TUNING_SPEC_PATH=<absolute file path>` : Indicates where to load a tuning spec for conv launchables. Some tuning specs are already included in `tuning_specs.mlir`, and the default behavior of `get_launchable` will use this included spec. You can disable using tuning specs via `export BOO_TUNING_SPEC_PATH=""`.

And several other variables are useful for debugging:

- `TURBINE_LOG_LEVEL=DEBUG`: enable debug output from Turbine.
- `TURBINE_DEBUG` : See `iree.turbine.runtime.logging` for more details. It is useful to set `TURBINE_DEBUG="log_level=DEBUG"` to see debug prints when something goes wrong.


## Advanced Usage

For benchmarking, and numerics debugging see [driver documentation](driver/README.md).
