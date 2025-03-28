This directory contains scripts for exporting/executing various forward and backward convolution configurations.

## Generating IR
To quickly generate some examples, install iree-turbine (e.g., `pip install -e .` from the base directory for iree-turbine), then

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

If you want to generate a convolution signature explicitly from python:

```python

from iree.turbine.kernel.boo.conv_exports.conv import ConvSignature
from iree.turbine.kernel.boo.conv_exports.generate import generate_mlir

# see the definition for default values, or customize stride, dilation, padding, groups, etc.
signature = ConvSignature(
    input_shape = [2,3,16,32],
    kernel_shape = [10,3,2,2],
    shared_layout="NCHW",
    # Can specify a mode "fwd" "bwd" "wrw" with:
    mode="bwd",
)

conv = signature.get_nn_module()

module = generate_mlir(
    signature,
    # output_path=...,
    import_pipeline=["torch-to-iree"],
    # print_ir=...,
)

# Instead of `generate_mlir` you could instead use iree-turbine's exporter:

from iree.turbine.aot import export

args = signature.get_sample_conv_args()

exported = export(conv, args=args)

# get iree-input IR
exported.import_to("full")

module = exported.mlir_module

# you can also directly compile the export output and save the vmfb to a file:

from pathlib import Path

vmfb_path = Path(__file__).parent / "sample.vmfb"

exported.compile(save_to=vmfb_path)

```

## Executing convolutions

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
