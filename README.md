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

