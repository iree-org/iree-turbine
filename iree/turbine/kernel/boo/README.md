The BOO (Bag of Ops) project is intended to provide a python interface for launching individual IREE kernels within a pytorch model.

Currently, forward and backward convolution kernels are supported.

## Environment Variables

There are a few environment variables that control BOO behavior:

- `BOO_CACHE_ON=<0 or 1>`: whether to cache kernel artifacts for boo kernels (default = 1).
- `BOO_CACHE_DIR=<path to cache dir>`: where to store kernel artifacts (default = `~/.cache/turbine_kernels/boo/`).
- `BOO_USE_BACKWARD_KERNELS=<0 or 1>`: whether to use our backward kernels (default = 0).

## Usage:

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

### Replace `Conv2d` in a model with `BooConv2d`:

For a `resnet_18` boo convolution example with sample training, see [`examples/resnet_18_backward.py`](examples/resnet_18_backward.py).

```python
from iree.turbine.kernel.boo.modeling import replace_conv2d_with_boo_conv

model = ... # torch.nn.Module

replacement_kwargs = {"stride" : (1,1)} # controls which types of Conv2d are replaced.
model = replace_conv2d_with_boo_conv(model, **replacement_kwargs)
outputs = model(...)
```

### Use a `BooConv2d` module directly:

```python
from iree.turbine.kernel.boo.modeling import BooConv2d

conv2d = BooConv2d(...) # usual Conv2d args
```
