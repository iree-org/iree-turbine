boo = bag of ops

The heirarchy for kernel implementations is something like:

1. namespace - has a cache_manager and contains a registry of templates
2. kernel template - python callable which can specialize to certain args (similar to CustomOp)
3. template specialization - essentially 1-1 equivalent to device-agnostic mlir specialized from a template for specific args and kwargs. Can be compiled to a kernel.
4. kernel - essentially 1-1 equivalent to a pre-compiled dispatch for a template specialization

The cache_manager should be able to check for generated specializations and kernels and retrieve them as needed. Kernel Templates should be able to build/compile specializations as-needed if they aren't available from the cache.

User experience should be as easy as:

1. set up some options for iree
2. choose which kernel libraries to allow
3. can either call custom ops from these libraries directly, or..
4. can wrap a function call in a decorator to auto generate kernel impls?

E.g.,

```python
import iree.turbine.boo as boo

iree_options = {"target_backends" : "rocm", "flags" : ["--iree-hip-target=gfx942"]}
km = boo.KernelManager(iree_options)

km.load_namespace("tkw")

# use the tkw namespace to search for a kernel template by name
# calling this mmt function should use the tkw cache_manager to find/deploy/generate
mmt = km.find_template_op("tkw", "mmt")

LHS = torch.randn(16, 32)
RHS = torch.randn(32, 128)

# calling this template op should automatically check the cache for a kernel matching the input signature
# if no kernel exists for this signature, specialize and compile - adding those to the cache
result = mmt(LHS, RHS)

## example flow of using auto-generated kernels ##

# the boo cache_manager should also allow saving auto-generated kernels to disk for loading in a new session.

# allows saving the template and kernels for future use
@km.auto_kernel(op_name = "mmt", save_kernels=True)
def my_auto_mmt(LHS, RHS):
    return torch.matmul(LHS, RHS.transpose(-1,-2))

result = mmt(LHS, RHS)

# should be available from a new python session
same_mmt = km.find_template_op("auto_op", "my_auto_mmt")

same_result = same_mmt(LHS, RHS)

# find fused kernel templates?
fused_conv_relu = km.find_template_op("tkw", "fused_conv_relu")

args = (...)
kwargs = {...}

fused_conv_relu(*args, **kwargs)

```
