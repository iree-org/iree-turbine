## Contents

This directory contains some code for generating and launching convolution kernels.

- `conv.py` : Contains `ConvSignature` class for initializing a specific convolution in python.
- `generate.py` : (for IREE developers) A script for generating some MLIR.
- `miopen_parser.py` : Contains a parser for converting an MiOpen driver command to a `ConvSignature`.
- `preload.py` : Contains a `CachePrepopulator` class for pre-populating the launchable file cache. Can be run as a script with a specified commands `.txt` file.
- `numerics_runner.py` : A script for comparing boo gpu numerics against pytorch for convolutions in a specified `.txt` file.

## Useful Environment Variables:

- `BOO_CACHE_ON=<0 or 1>` : Whether to store launchable artifacts to a file cache. If unset, this will default to 1 (cache on).
- `BOO_CACHE_DIR=<absolute dir path>` : Indicates where to store launchable artifacts. If unset, will default to `~/.cache/turbine_kernels/boo/`.
- `BOO_TUNING_SPEC_PATH=<absolute file path>` : Indicates where to load a tuning spec for conv launchables. Some tuning specs are already included in `tuning_specs.mlir`, and the default behavior of `get_launchable` will use this included spec. You can disable using tuning specs via `export BOO_TUNING_SPEC_PATH=""`.
- `TURBINE_DEBUG` : See `iree.turbine.runtime.logging` for more details. It is useful to set `TURBINE_DEBUG="log_level=DEBUG"` to see debug prints when something goes wrong.

## Usage from python

Some examples of creating and launching conv kernels from python.

### Basic Use:

You can use `boo_conv` for user-friendly eager execution.

```python
import torch

from iree.turbine.kernel.boo.ops import boo_conv

# make some 3, 4, or 5d tensors (e.g. randn or splats)
x = torch.tensor(...)
w = torch.tensor(...)

# perform convolution
y = boo_conv(x,w)

# usual conv kwargs are also available:
y = boo_conv(x,w,stride=[1,2],dilation=[2,2], padding=[1,0])

# additional kwargs for alternative layouts also available:
y = boo_conv(x,w, shared_layout="NHWC")
y = boo_conv(x,w, input_layout="NCHW", kernel_layout="NHWC", output_layout="NCHW")
```

### Basic Use: define signatures from explicit shapes

```python
import torch

from iree.turbine.kernel.boo.conv_exports import ConvSignature
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

### Get signatures from MiOpen driver commands:

```python

from iree.turbine.kernel.boo.conv_exports.miopen_parser import ConvParser
from iree.turbine.kernel.boo.driver.launch import get_launchable

miopen_driver_command = "convbfp16 -n 128 -c 128 -H 24 -W 48 -k 384 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1 --iter 100 --in_layout NHWC --out_layout NHWC --fil_layout NHWC"
signature = ConvParser.command_to_signature(miopen_driver_command)

conv = get_launchable(signature)

```

### Prepopulate the launchable cache

#### From the command line

```
cd iree/turbine/kernel/boo/conv_exports
python preload.py "sample_commands.txt"
```

Will preload the launchable cache for all available devices and for all miopen driver commands in "sample_commands.txt".

To see other options, run:

```
python preload.py --help
```

If you just want to generate mlir files, you can pass `--device ""` or `-d ""`.

#### From python

```python
from iree.turbine.kernel.boo.conv_exports import CachePopulator
from iree.turbine.kernel.boo.driver.launch import get_launchable

populator = CachePopulator(commands_file="path/to/miopen/commands_list.txt")

populator.run()

# You can grab an example signature from the populator:
sample_signature = populator.signatures[0]

# One can also check the cache for this signature
cache_status = populator.get_cache_status(sample_signature.get_func_name())
print(cache_status)

# You can also get the list of failed signatures via:
failed_funcs = populator.get_failures()

conv = get_launchable(sample_signature)
```

## Benchmarking convolutions

See [driver documentation](../driver/README.md#benchmarking)
