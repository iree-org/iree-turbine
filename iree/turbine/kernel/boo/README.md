BOO = Bag of ops

This subdirectory contains some simple API for generating and calling into a larger corpus of kernels.

The general idea is:

1. Write a pytorch function that should ideally be a single kernel.
2. Import the function to MLIR.
3. Try to force the IR function into a single dispatch (TODO: add a func op attribute for this on import)

You can define a boo kernel as easily as:

```python
from iree.turbine.kernel.boo import boo_kernel
from iree.turbine.kernel._support.tracing import TestLaunchContext


@boo_kernel
def matmul_relu(lhs, rhs):
    mm = torch.matmul(lhs, rhs)
    return torch.nn.functional.relu(mm)
```

To compile/run this kernel, you need to set some configurations for the compiler and runtime.

```python
compile_config = {
    "target_backends": ("llvm-cpu",),
    "flags": ("--iree-llvmcpu-target-cpu=host",),
    "print_mlir": True,
}

run_config = {
    "device": "local-task",
}

with TestLaunchContext(compile_config=compile_config, run_config=run_config):
    # this compiles for m,n,k = 16,16,32 bfloat16 example
    x = torch.randn([16,32], dtype=torch.bfloat16)
    w = torch.randn([32,16], dtype=torch.bfloat16)
    y = matmul_relu(x,w)
    # this will now pull from the cached files in ~/.cache/turbine_kernels/
    x = torch.randn([16,32], dtype=torch.bfloat16)
    w = torch.randn([32,16], dtype=torch.bfloat16)
    y = matmul_relu(x,w)
    # this is a new configuration, so it will need to re-compile a new kernel
    x = torch.randn([128,128], dtype=torch.float32)
    w = torch.randn([128,128], dtype=torch.float32)
    y = matmul_relu(x,w)
```
