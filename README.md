# SHARK Turbine
![image](https://netl.doe.gov/sites/default/files/2020-11/Turbine-8412270026_83cfc8ee8f_c.jpg)

Turbine is the set of development tools that the [SHARK Team](https://github.com/nod-ai/SHARK)
is building for deploying all of our models for deployment to the cloud and devices. We
are building it as we transition from our TorchScript-era 1-off export and compilation 
to a unified approach based on PyTorch 2 and Dynamo. While we use it heavily ourselves, it 
is intended to be a general purpose model compilation and execution tool.

Turbine provides three primary tools:

* *AOT Export*: For compiling one or more `nn.Module`s to compiled, deployment
  ready artifacts. This operates via both a [simple one-shot export API](https://github.com/nod-ai/SHARK-Turbine/blob/main/python/shark_turbine/aot/exporter.py)
  for simple models and an underlying [advanced API](https://github.com/nod-ai/SHARK-Turbine/blob/main/python/shark_turbine/aot/compiled_module.py) for complicated models
  and accessing the full features of the runtime.
* *Eager Execution*: A `torch.compile` backend is provided and a Turbine Tensor/Device
  is available for more native, interactive use within a PyTorch session.
* *Turbine Kernels*: (coming soon) A union of the [Triton](https://github.com/openai/triton) approach and
  [Pallas](https://jax.readthedocs.io/en/latest/pallas/index.html) but based on
  native PyTorch constructs and tracing. It is intended to complement for simple
  cases where direct emission to the underlying, cross platform, vector programming model
  is desirable.

Under the covers, Turbine is based heavily on [IREE](https://github.com/openxla/iree) and
[torch-mlir](https://github.com/llvm/torch-mlir) and we use it to drive evolution
of both, upstreaming infrastructure as it becomes timely to do so.

## Contact Us

Turbine is under active development. If you would like to participate as it comes online,
please reach out to us on the `#turbine` channel of the 
[nod-ai Discord server](https://discord.gg/QMmR6f8rGb).

## Quick Start for Users

1. Install from source:

```
pip install shark-turbine
# Or editable: pip install -e .
```

The above does install some unecessary cuda/cudnn packages for cpu use. To avoid this you
can specify pytorch-cpu and install via:
```
pip install --index-url https://download.pytorch.org/whl/cpu \
    -r pytorch-cpu-requirements.txt \
    -r torchvision-requirements.txt
pip install shark-turbine
```

(or follow the "Developers" instructions below for installing from head/nightly)

2. Try one of the samples:

Generally, we use Turbine to produce valid, dynamic shaped Torch IR (from the 
[`torch-mlir torch` dialect](https://github.com/llvm/torch-mlir/tree/main/include/torch-mlir/Dialect/Torch/IR)
with various approaches to handling globals). Depending on the use-case and status of the
compiler, these should be compilable via IREE with `--iree-input-type=torch` for
end to end execution. Dynamic shape support in torch-mlir is a work in progress,
and not everything works at head with release binaries at present.

  * [AOT MLP With Static Shapes](https://github.com/nod-ai/SHARK-Turbine/blob/main/examples/aot_mlp/mlp_export_simple.py)
  * [AOT MLP with a dynamic batch size](https://github.com/nod-ai/SHARK-Turbine/blob/main/examples/aot_mlp/mlp_export_dynamic.py)
  * [AOT llama2](https://github.com/nod-ai/SHARK-Turbine/blob/main/examples/llama2_inference/llama2.ipynb):
    Dynamic sequence length custom compiled module with state management internal to the model.

## Developers

### Getting Up and Running

If only looking to develop against this project, then you need to install Python
deps for the following:

* PyTorch
* iree-compiler (with Torch input support)
* iree-runtime

The pinned deps at HEAD require pre-release versions of all of the above, and
therefore require additional pip flags to install. Therefore, to satisfy
development, we provide a `requirements.txt` file which installs precise
versions and has all flags. This can be installed prior to the package:

Installing into a venv is highly recommended.

```
pip install --upgrade -r requirements.txt
pip install --upgrade -e .[torch-cpu-nightly,testing]
```

Run tests:

```
pytest
```

### Using a development compiler

If doing native development of the compiler, it can be useful to switch to
source builds for iree-compiler and iree-runtime.

In order to do this, check out [IREE](https://github.com/openxla/iree) and
follow the instructions to [build from source](https://openxla.github.io/iree/building-from-source/getting-started/#configuration-settings), making
sure to specify [additional options](https://openxla.github.io/iree/building-from-source/getting-started/#building-with-cmake):

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
