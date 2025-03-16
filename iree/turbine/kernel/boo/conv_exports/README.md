This directory contains scripts for exporting various forward and backward convolution configurations to MLIR.

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
