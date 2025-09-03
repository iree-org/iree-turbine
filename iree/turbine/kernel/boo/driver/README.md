Facilities to launch BOO kernels in standalone processes.

## Contents

- `driver.py`: A script for benchmarking kernels with MIOpen-compatible options.
- `launch.py`: Contains the `get_launchable` for converting an `OpSignature` into a kernel launchable from Python. This launchable interacts with a file cache the user can control with the environment variables `BOO_CACHE_ON=<0 or 1>` and  `BOO_CACHE_DIR=<absolute path>`.
- `numerics.py`: A script for comparing BOO GPU numerics against PyTorch baselines for ops specified by commands in a `.txt` file.
- `preload.py`: Instantiates a `CachePopulator` class for pre-populating the launchable file cache. Can be run as a script with  specified commands `.txt` file.
- `registry.py`: A module for automated discovery of BOO ops in the `../op_exports` directory and dispatch to the appropriate signature and parser classes based on the textual command.


## Pre-populating the Launchable Cache

### From Command Line

```sh
# Assuming working directory is iree/turbine/kernel/boo/driver:
python preload.py <sample_commands.txt>
```

Will preload the launchable cache for all available devices and for all MIOpen driver commands in "sample_commands.txt".

To see other options, run:

```
python preload.py --help
```

### From Python

```python
from iree.turbine.kernel.boo.driver.preload import CachePopulator
from iree.turbine.kernel.boo.driver.launch import get_launchable

# Note that you will have to specify op-specific parser and signature classes.
populator = CachePopulator(commands_file="path/to/miopen/commands_list.txt")
populator.run()

# You can grab an example signature from the populator:
sample_signature = populator.signatures[0]

# One can also check the cache for this signature
cache_status = populator.get_cache_status(sample_signature.func_name)
print(cache_status)

# You can also get the list of failed signatures via:
failed_funcs = populator.get_failures()

conv = get_launchable(sample_signature)
```

## Benchmarking

The `driver.py` script allows for running kernels from the command line. It uses
an interface similar to that of `MIOpenDriver`: some additional flags or flag
values are added to support scenarios not supported by the driver such as
non-default layouts.

```console
$ python driver.py convbfp16 -n 128 -c 128 -H 24 -W 48 -k 384 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1 --iter 100 --in_layout NHWC --out_layout NHWC --fil_layout NHWC
convbfp16 -n 128 -c 128 -H 24 -W 48 -k 384 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 --iter 100 --in_layout NHWC --out_layout NHWC --fil_layout NHWC
>>> tensor([...], device='cuda:0', size=(128, 24, 48, 384), dtype=torch.bfloat16)
>>> min=85.24us max=371.23us mean=255.28us stddev=92.66us
```

The driver commands can also be supplied through a file, using `--commands-file`:

```console
$ python driver.py --commands-file sample_commands.txt
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

The `--time 1` (or `-t 1` for short) option to collect timing is implemented by
launching the kernel, which is then profiled using `torch.profiler`. Overall GPU
time is reported, including memory and other operations not necessarily included
in the kernel itself. Note: you can statistics to a csv file with
`--csv=results.csv`.

BOO operations can be compared against a set of reference backends by providing
one or more `--reference-backend` flags. Currently supported backends include:

- `torch`: Eager Pytorch.
- `inductor`: Pytorch Inductor (`torch.compile` default).
- `iree_boo_inductor`: BOO where applicable and Inductor otherwise.
- `iree_boo_legacy`: direct call of BOO kernel without `torch.compile`.

#### Misc requirements Q&A:

1. How to fix the error `ImportError: <path-to-sharedlib>/libIREECompiler.so: cannot allocate memory in static TLS block`

  **A:** Please add the following into your rc file and reload it:
  ```bash
  export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=4096
  ```

  Please note that this is an OS bug, for context of this bug please refer to [here](https://github.com/pytorch/pytorch/issues/2575#issuecomment-1640566350).
