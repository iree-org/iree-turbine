# SHARK Turbine
![image](https://netl.doe.gov/sites/default/files/2020-11/Turbine-8412270026_83cfc8ee8f_c.jpg)

This project provides a unified build of [IREE](https://github.com/openxla/iree),
[torch-mlir](https://github.com/llvm/torch-mlir), and auxilliary support for
providing a tight integration with PyTorch and other related frameworks. It
presently uses IREE's compiler plugin API to achieve this coupling, allowing
us to build a specialized compiler with tight coupling to PyTorch concepts.

WARNING: This project is still under construction and is at an early phase.

As things progress, we will be building out:

* Native Dynamo support.
* Integration to allow use of the compiler flow as part of the eager flow.
* Compiler support for hallmark PyTorch features such as strided tensors,
  in-place semantics, dynamic shapes, etc (IREE mostly supports these
  features under the covers but they need adaptation for good interop with
  PyTorch).
* Custom op and type support for emerging low-precision numerics.
* Triton code generation and retargeting.
* Cleaned up APIs and options for AOT compiling and standalone deployment.

We would also like to engage with the community to continue to push the bounds
on what Dynamo can do, especially when it comes to tighter integration with
optimizers and collectives -- both of which we are eager to integrate with
PyTorch to a similar level as can be achieved with whole-graph frameworks like
Jax.

## Getting Up and Running

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
pip install --upgrade -e .[torch,testing]
```

Run tests:

```
pytest
```

## Using a development compiler

If doing native development of the compiler, it can be useful to switch to
source builds for iree-compiler and iree-runtime.

In order to do this, check out [IREE](https://github.com/openxla/iree) and
follow the instructions to [build from source](https://openxla.github.io/iree/building-from-source/getting-started/#configuration-settings), making
sure to specify [additional options](https://openxla.github.io/iree/building-from-source/getting-started/#building-with-cmake):

```
-DIREE_BUILD_PYTHON_BINDINGS=ON -DPython3_EXECUTABLE="$(which python)"
```

### Configuring Python

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
