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

Note that you will need a compatible side-by-side checkout of the following
projects:

* [IREE](https://github.com/openxla/iree.git)
* [torch-mlir](https://github.com/llvm/torch-mlir.git)

Run `python sync_deps.py` to fetch both and bring them to the last known
good commit. If you already have them checked out, running this script will
update them to the correct commit. If doing active development on either,
you may want to manage this yourself (see the top of the script for the
commit hashes).

### Building for development

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

# Python projects.
pip install -e frontend
source build/iree/.env
```

## Project Maintenance

This section is a work in progress describing various project maintenance
tasks.

### Pre-requisite: Install SHARK-devtools

```
pip install git+https://github.com/nod-ai/SHARK-devtools.git
```

### Sync all deps to pinned versions

```
shark-ws sync
```

### Update IREE to head

This updates the pinned IREE revision to the HEAD revision at the remote.

```
# Updates the sync_deps.py metadata.
shark-ws roll iree
# Brings all dependencies to pinned versions.
shark-ws sync
```

### Full update of all deps

This updates the pinned revisions of all dependencies. This is presently done
by updating `iree` and `torch-mlir` to remote HEAD.

```
# Updates the sync_deps.py metadata.
shark-ws roll nightly
# Brings all dependencies to pinned versions.
shark-ws sync
```

### Pin current versions of all deps

This can be done if local, cross project changes have been made and landed.
It snapshots the state of all deps as actually checked out and updates
the metadata.

```
shark-ws pin
```
