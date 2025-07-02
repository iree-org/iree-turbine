Facilities to launch BOO kernels in standalone processes.

## Contents

- `driver.py`: A script for benchmarking kernels with MIOpen-compatible options.
- `launch.py`: Contains the `get_launchbale` for converting an `OpSignature` into a kernel launchable from Python. This launchable interacts with a file cache the user can control with the environment variables `BOO_CACHE_ON=<0 or 1>` and  `BOO_CACHE_DIR=<absolute path>`.

## Benchmarking

The `driver.py` script allows for running kernels from the command line. It uses the same interface as `MIOpenDriver`:

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

The `--time 1` (or `-t 1` for short) option to collect timing is implemented by launching the kernel in a subprocess, which is then traced using `tracy`. Note: you can output `min_time (us)` to a csv file with `--csv=results.csv`.

### Requirements for collecting timing info:

#### pip installed `iree-base-runtime`:

- set `IREE_PY_RUNTIME=tracy` in your environment.
- build `tracy-csvexport` and add it to your `PATH`:

```
git clone https://github.com/wolfpld/tracy.git
cd tracy/csvexport
cmake -B csvexport/build -S csvexport -DCMAKE_BUILD_TYPE=Release
cmake --build csvexport/build --parallel --config Release
export PATH="$PWD/build/:$PATH"
```

#### local iree build with python bindings:

- You may need to configure cmake with `-DTRACY_DELAYED_INIT=ON -DTRACY_MANUAL_LIFETIME=ON` to work around issues with the tracy client being closed before traces are captured.
- Make sure `PYTHONPATH` is set to your locally built compiler and runtime python bindings. E.g. `source <build-dir>/.env && export PYTHONPATH`.
- Build `iree-tracy-capture`. Use `-DIREE_BUILD_TRACY=ON` when building IREE, and include `<build-dir>/tracy` in your `PATH` environment variable.
- Build `tracy-csvexport`. See https://iree.dev/developers/performance/profiling-with-tracy/#building-the-tracy-csvexport-tool

#### Misc requirements Q&A:

1. How to fix the error `ImportError: <path-to-sharedlib>/libIREECompiler.so: cannot allocate memory in static TLS block`

  **A:** Please add the following into your rc file and reload it:
  ```bash
  export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=4096
  ```

  Please note that this is an OS bug, for context of this bug please refer to [here](https://github.com/pytorch/pytorch/issues/2575#issuecomment-1640566350).

2. I run into `buffer overflow detected: terminated`, what's wrong?

  **A:** Ensure that you consistently use the `iree-tracy-capture` built alongside the IREE compiler. You should either use both from `iree-base-runtime` or from local built setup, but do not combine different sources. This is a known issue when use the locally built iree compiler together with `tracy-capture` from upstream. To determine whether a failure is triggered by `tracy-capture`, try running Boo driver with `-t 0` and see if the problem still exists.
