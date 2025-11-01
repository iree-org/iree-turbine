# IREE Turbine

[![PyPI version](https://badge.fury.io/py/iree-turbine.svg)](https://badge.fury.io/py/iree-turbine)
[![Documentation status](https://readthedocs.org/projects/iree-turbine/badge/?version=latest)](https://app.readthedocs.org/projects/iree-turbine/builds/?version__slug=latest)

<img src="https://netl.doe.gov/sites/default/files/2020-11/Turbine-8412270026_83cfc8ee8f_c.jpg" height="300px" width="300px">

Turbine is [IREE's](https://iree.dev/) frontend for
[PyTorch](https://pytorch.org/).

Turbine provides a collection of tools:

* *AOT Export*: For compiling one or more `nn.Module`s to compiled, deployment
  ready artifacts. This operates via both a simple one-shot export API
  (also available via
  [torch-mlir](https://github.com/llvm/torch-mlir/blob/main/python/torch_mlir/extras/fx_importer.py))
  for simple models and an underlying
  [advanced API](https://github.com/iree-org/iree-turbine/blob/main/iree/turbine/aot/compiled_module.py)
  for complicated models and accessing the full features of the runtime.
* *Eager Execution*: A `torch.compile` backend is provided and a Turbine Tensor/Device
  is available for more native, interactive use within a PyTorch session.
* *Custom Ops*: Integration for defining custom PyTorch ops and implementing them in
  terms of IREE's backend IR or a Pythonic kernel language.

## Documentation

* API reference pages and project documentation: https://iree-turbine.readthedocs.io/
* Guides for using Turbine as a bridge between PyTorch and IREE:
  https://iree.dev/guides/ml-frameworks/pytorch/

## Contact us

Turbine is under active development. Feel free to reach out on one of
[IREE's communication channels](https://github.com/iree-org/iree?tab=readme-ov-file#communication-channels)
(specifically, we monitor the `#pytorch` and `#turbine` channels on the IREE
Discord server).

## Quick start for users

1. Install from PyPI:

    Install a torch version that fulfills your needs:

    ```bash
    # Fast installation of torch with just CPU support.
    # See other options at https://pytorch.org/get-started/locally/
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    ```

    Then install iree-turbine:

    ```bash
    # Stable releases
    pip install iree-turbine

    # Nightly releases
    pip install --find-links https://iree.dev/pip-release-links.html --upgrade --pre iree-turbine
    ```

    (or follow the "Developers" instructions below)

2. Try one of the [examples](https://github.com/iree-org/iree-turbine/blob/main/examples/):

    * [AOT MLP With Static Shapes](https://github.com/iree-org/iree-turbine/blob/main/examples/aot_mlp/mlp_export_simple.py)
    * [Eager MNIST with `torch.compile`](https://github.com/iree-org/iree-turbine/blob/main/examples/eager_mlp/mlp_eager_simple.py)
    * [Dynamic AOT resnet-18](https://github.com/iree-org/iree-turbine/blob/main/examples/resnet-18/)

    Generally, we use Turbine to produce valid, dynamic shaped Torch IR (from the
    [torch-mlir `torch` dialect](https://github.com/llvm/torch-mlir/tree/main/include/torch-mlir/Dialect/Torch/IR)
    with various approaches to handling globals). Depending on the use-case and status of the
    compiler, these should be compilable via IREE with `--iree-input-type=torch` for
    end to end execution. Dynamic shape support in torch-mlir is a work in progress,
    and not everything works at head with release binaries at present.

## Developers

Use this as a guide to get started developing the project using pinned,
pre-release dependencies. You are welcome to deviate as you see fit, but
these canonical directions mirror what the CI does.

### Setup a venv

We recommend setting up a
[virtual environment (venv)](https://docs.python.org/3/library/venv.html). The
project is configured to ignore `.venv` directories, and editors like VSCode
pick them up by default.

```bash
python -m venv --prompt iree-turbine .venv
source .venv/bin/activate
```

### Install PyTorch for your system

You need to explicit install a PyTorch version that fulfills your needs.
On Linux, install a variant by either following the
[official instructions](https://pytorch.org/get-started/locally/) or by using
one of our `requirements.txt` files:

* *CPU: [`pytorch-cpu-requirements.txt`](https://github.com/iree-org/iree-turbine/blob/main/pytorch-cpu-requirements.txt)*

  ```bash
  pip install -r pytorch-cpu-requirements.txt
  ```

* *ROCM: [`pytorch-rocm-requirements.txt`](https://github.com/iree-org/iree-turbine/blob/main/pytorch-rocm-requirements.txt)*

  ```bash
  pip install -r pytorch-rocm-requirements.txt
  ```

### Install development packages

```bash
# Install editable local projects.
pip install -r requirements.txt -e .
```

### Running tests

```bash
# Python unit tests
pytest .
```

### Optional: Pre-commits and developer settings

This project is set up to use the [`pre-commit`](https://pre-commit.com/)
tooling. To install it in your local repo, run: `pre-commit install`. After
this point, when making commits locally, hooks will run automatically.

### Using a development compiler

If doing native development of the compiler, it can be useful to switch to
source builds for the
[iree-base-compiler](https://pypi.org/project/iree-base-compiler/) and
[iree-base-runtime](https://pypi.org/project/iree-base-runtime/) packages.

In order to do this, check out [IREE](https://github.com/iree-org/iree) and
follow the instructions to
[build from source](https://iree.dev/building-from-source/getting-started/),
making sure to specify
[additional options for the Python bindings](https://iree.dev/building-from-source/getting-started/#building-with-cmake):

```bash
-DIREE_BUILD_PYTHON_BINDINGS=ON -DPython3_EXECUTABLE="$(which python)"
```

#### Configuring python

Uninstall existing packages (including any with the old package names):

```bash
pip uninstall iree-compiler iree-base-compiler iree-runtime iree-base-runtime
```

Copy the `.env` file from `iree/` to this source directory to get IDE
support and add to your path for use from your shell:

```bash
source .env && export PYTHONPATH
```
