# SHARK Turbine

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

## Initial Development

Currently, development is being done by checking out iree and torch-mlir
as siblings and then doing the following from shark-turbine:

```
cmake -GNinja -Bbuild -S. \
  -DIREE_BUILD_PYTHON_BINDINGS=ON \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DIREE_ENABLE_ASSERTIONS=ON \
  -DIREE_ENABLE_SPLIT_DWARF=ON \
  -DIREE_ENABLE_THIN_ARCHIVES=ON \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DIREE_ENABLE_LLD=ON \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
```
