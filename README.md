<!--
SPDX-FileCopyrightText: 2024 The IREE Authors

SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

# IREE Turbine

![image](https://netl.doe.gov/sites/default/files/2020-11/Turbine-8412270026_83cfc8ee8f_c.jpg)

Turbine is IREE's frontend for PyTorch.

Turbine provides a collection of tools:

* *AOT Export*: For compiling one or more `nn.Module`s to compiled, deployment
  ready artifacts. This operates via both a simple one-shot export API (Already upstreamed to [torch-mlir](https://github.com/llvm/torch-mlir/blob/main/python/torch_mlir/extras/fx_importer.py))
  for simple models and an underlying [advanced API](shark_turbine/aot/compiled_module.py) for complicated models
  and accessing the full features of the runtime.
* *Eager Execution*: A `torch.compile` backend is provided and a Turbine Tensor/Device
  is available for more native, interactive use within a PyTorch session.
* *Custom Ops*: Integration for defining custom PyTorch ops and implementing them in
  terms of IREE's backend IR or a Pythonic kernel language.

## Contact Us

Turbine is under active development. Feel free to reach out on one of
[IREE's communication channels](https://github.com/iree-org/iree?tab=readme-ov-file#communication-channels) (specifically, we monitor the
#pytorch channel on the IREE Discord server).

## Quick Start for Users

1. Install from source:

```
pip install iree-turbine
# Or for editable: see instructions under developers
```

The above does install some unecessary cuda/cudnn packages for cpu use. To avoid this you
can specify pytorch-cpu and install via:
```
pip install -r pytorch-cpu-requirements.txt
pip install iree-turbine
```

(or follow the "Developers" instructions below for installing from head/nightly)

2. Try one of the samples:

Generally, we use Turbine to produce valid, dynamic shaped Torch IR (from the
[`torch-mlir torch` dialect](https://github.com/llvm/torch-mlir/tree/main/include/torch-mlir/Dialect/Torch/IR)
with various approaches to handling globals). Depending on the use-case and status of the
compiler, these should be compilable via IREE with `--iree-input-type=torch` for
end to end execution. Dynamic shape support in torch-mlir is a work in progress,
and not everything works at head with release binaries at present.

  * [AOT MLP With Static Shapes](examples/aot_mlp/mlp_export_simple.py)
  * [AOT MLP with a dynamic batch size](examples/aot_mlp/mlp_export_dynamic.py)
  * [AOT llama2](examples/llama2_inference/llama2.ipynb):
    Dynamic sequence length custom compiled module with state management internal to the model.
  * [Eager MNIST with `torch.compile`](examples/eager_mlp/mlp_eager_simple.py)

## Developers

Use this as a guide to get started developing the project using pinned,
pre-release dependencies. You are welcome to deviate as you see fit, but
these canonical directions mirror what the CI does.

### Setup a venv

We recommend setting up a virtual environment (venv). The project is configured
to ignore `.venv` directories, and editors like VSCode pick them up by default.

```
python -m venv --prompt iree-turbine .venv
source .venv/bin/activate
```

### Install PyTorch for Your System

If no explicit action is taken, the default PyTorch version will be installed.
This will give you a current CUDA-based version. Install a different variant
by doing so explicitly first:

*CPU:*

```
pip install -r pytorch-cpu-requirements.txt
```

*ROCM:*

```
pip install -r pytorch-rocm-requirements.txt
```

### Install Development Packages

```
# Install editable local projects.
pip install -r requirements.txt -e .
```

### Running Tests

```
pytest .
```

### Optional: Pre-commits and developer settings

This project is set up to use the `pre-commit` tooling. To install it in
your local repo, run: `pre-commit install`. After this point, when making
commits locally, hooks will run. See https://pre-commit.com/

### Using a development compiler

If doing native development of the compiler, it can be useful to switch to
source builds for iree-compiler and iree-runtime.

In order to do this, check out [IREE](https://github.com/openxla/iree) and
follow the instructions to [build from source](https://iree.dev/building-from-source/getting-started/), making
sure to specify [additional options for the Python bindings](https://iree.dev/building-from-source/getting-started/#building-with-cmake):

```
-DIREE_BUILD_PYTHON_BINDINGS=ON -DPython3_EXECUTABLE="$(which python)"
```

#### Configuring Python

Uninstall existing packages:

```
pip uninstall iree-compiler
pip uninstall iree-runtime
```

Copy the `.env` file from `iree/` to this source directory to get IDE
support and add to your path for use from your shell:

```
source .env && export PYTHONPATH
```
