## Contents

This directory contains some code for generating and launching convolution kernels.

- `boo_driver.py` : A script for benchmarking conv launchables.
- `conv.py` : Contains `ConvSignature` class for initializing a specific convolution in python.
- `generate.py` : (for IREE developers) A script for generating some MLIR.
- `launch.py` : Contains `get_launchable` function for converting a `ConvSignature` to a kernel launchable from python. This launchable interacts with a file cache the user can control with the environment variables `BOO_CACHE_ON=<0 or 1>` and  `BOO_CACHE_DIR=<absolute path>`.
- `miopen_parser.py` : Contains a parser for converting an MiOpen driver command to a `ConvSignature`.
- `preload.py` : Contains a `CachePrepopulator` class for prepopulating the launchable file cache.

## Usage from python

Some examples of creating and launching conv kernels from python.

### Basic Use: define signatures from explicit shapes

```python
import torch

from iree.turbine.kernel.boo.conv_exports import ConvSignature, get_launchable

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

x, w = fwd_sig.get_sample_conv_args(device=torch_device, seed=10)

y = conv_fwd(x, w)

# get a random dLdy to back-prop.
dLdy, _ = wrw_sig.get_sample_conv_args(device=torch_device, seed=2)
dLdw = conv_wrw(dLdy, x)
```

### Get signatures from MiOpen driver commands:

```python

from iree.turbine.kernel.boo.conv_exports import command_to_signature, get_launchable

miopen_driver_command = "convbfp16 -n 128 -c 128 -H 24 -W 48 -k 384 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1 --iter 100 --in_layout NHWC --out_layout NHWC --fil_layout NHWC"
signature = command_to_signature(miopen_driver_command)

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

#### From python

```python
from iree.turbine.kernel.boo.conv_exports import CachePopulator, get_launchable

populator = CachePopulator(commands_file="path/to/miopen/commands_list.txt")

populator.run()

# You can grab an example signature from the populator:
sample_signature = populator.signatures[0]

# One can also check the cache for this signature
cache_status = populator.get_cache_status()
print(cache_status[sample_signature.get_func_name()])

conv = get_launchable(sample_signature)
```

## generate.py script

This tool is an early iteration for generating some sample MLIR files for compiler triage.

To quickly generate some IR examples, install iree-turbine (e.g., `pip install -e .` from the base directory for iree-turbine), then

```
cd iree/turbine/boo/conv_exports
python generate.py -o all_convs --commands-file="sample_commands.txt"
```

This will populate a new directory `iree/turbine/kernel/boo/conv_exports/all_convs/` containing IR matching the MiOpen signatures in `sample_commands.txt`.

If you want to use a different pass pipeline on mlir import to lower to linalg or iree-input, the `generate.py` script allows running a user-specified pass pipeline from the initial torch-mlir IR. For example,

```
python generate.py -o all_convs -f "sample_commands.txt" --pipeline="builtin.module(torch-to-iree, iree-preprocessing-transpose-convolution-pipeline)"
```

Will sometimes fuse the transposes with the auto-generated `nchw` conv coming from pytorch.

There are also flags in `generate.py` for:

1. Running a single MiOpen driver signature. E.g.  `python generate.py -c "convbfp16 -n 16 -c 96 -H 48 -W 32 -k 288 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 1 -F 1 -t 1 --in_layout NHWC --out_layout NHWC --fil_layout NHWC"` will print the IR for that one convolution.
2. Filtering the provided signatures by number of spatial dims (`-N` / `--num-spatial-dims`)
3. Filtering the provided signatures by type (`-F`/ `--forw`) forward conv = "fwd", input backward = "bwd", weight backward = "wrw".

## Benchmarking convolutions

The `boo_driver.py` script allows for executing convolutions. It uses the same interface as `MIOpenDriver`:
```console
$ python boo_driver.py convbfp16 -n 128 -c 128 -H 24 -W 48 -k 384 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1 --iter 100 --in_layout NHWC --out_layout NHWC --fil_layout NHWC
convbfp16 -n 128 -c 128 -H 24 -W 48 -k 384 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 --iter 100 --in_layout NHWC --out_layout NHWC --fil_layout NHWC
>>> tensor([...], device='cuda:0', size=(128, 24, 48, 384), dtype=torch.bfloat16)
>>> min=85.24us max=371.23us mean=255.28us stddev=92.66us
```

The driver commands can also be supplied through a file, using `--commands-file`:
```console
$ python boo_driver.py --commands-file sample_commands.txt
convbfp16 -n 128 -c 128 -H 24 -W 48 -k 384 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 --iter 100 --in_layout NHWC --out_layout NHWC --fil_layout NHWC
>>> tensor([...], device='cuda:0', size=(128, 24, 48, 384), dtype=torch.bfloat16)
>>> min=77.76us max=115.29us mean=82.99us stddev=5.07us
convbfp16 -n 128 -c 128 -H 24 -W 48 -k 384 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 --iter 100 --in_layout NHWC --out_layout NHWC --fil_layout NHWC
>>> tensor([...], device='cuda:0', size=(384, 1, 1, 128), dtype=torch.bfloat16)
>>> min=4383.18us max=4713.31us mean=4407.70us stddev=31.58us
convbfp16 -n 128 -c 35 -H 48 -W 32 -k 35 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 --iter 100 --in_layout NHWC --out_layout NHWC --fil_layout NHWC
>>> tensor([...], device='cuda:0', size=(128, 48, 32, 35), dtype=torch.bfloat16)
>>> min=51.39us max=1989.03us mean=459.48us stddev=654.92us
...
```

The `-t 1` option to collect timing is implemented by launching the kernel in a subprocess, which is then traced using `tracy`. Using this requires:
- IREE runtime python bindings with tracy enabled. If using the pre-built `iree-base-runtime` package, this requires setting `IREE_PY_RUNTIME=tracy` in your environment
- `iree-tracy-capture`; see https://iree.dev/developers/performance/profiling-with-tracy/#building-the-tracy-capture-cli-tool
- `tracy-csvexport`; see https://iree.dev/developers/performance/profiling-with-tracy/#building-the-tracy-csvexport-tool
