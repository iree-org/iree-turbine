# Matrix Multiplication

Let's assume we want to compute the following matrix multiplication.
 ```math
 C = A \times B^{T}
 ```
where $A$ is a $M \times K$ matrix of type `fp16`, $B$ is a $N \times K$ matrix of type `fp16` and $C$ is a $M \times N$ matrix of type `fp32`. This is a mixed-precision transposed-B matrix multiplication operation that is common in machine learning applications.

In `tkw`, the kernels are written using a wave/warp programming model. This means that when we author the kernel we write it from the perspective of the work that a single wave would do. This also means that we need to know what operations a single wave can do.

For this example, we can assume that a wave can read and write data to different parts of the memory hierarchy. In addition, several graphical programming units (GPU) have instructions that perform matrix multiplication. For example, on AMD Instinct GPUs, there is a MFMA instruction that operates on tiles of the $A, B$ and $C$ matrices of size $16\times16$. We can assume that the wave can issue those specialized instructions.

The operations described above are exposed as primitives in `tkw` such as `tkw.read`, `tkw.write` and `tkw.mma`. We can put these together to write our kernel shown below.

```python
@tkw.wave(constraints)
def gemm(
    a: tkl.Memory[M, K, ADDRESS_SPACE, tkl.f16],
    b: tkl.Memory[N, K, ADDRESS_SPACE, tkl.f16],
    c: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, tkl.f32],
):
    c_reg = tkl.Register[M, N, tkl.f32](0.0)

    @tkw.iterate(K, init_args=[c_reg])
    def repeat(acc: tkl.Register[M, N, tkl.f32]) -> tkl.Register[M, N, tkl.f32]:
        a_reg = tkw.read(a, elements_per_thread=LOAD_ELEMS_PER_THREAD)
        b_reg = tkw.read(b, elements_per_thread=LOAD_ELEMS_PER_THREAD)
        acc = tkw.mma(a_reg, b_reg, acc)
        return acc

    tkw.write(repeat, c, elements_per_thread=STORE_ELEMS_PER_THREAD)
```

We define our kernel (named `gemm`) just like a regular function in Python, with the only difference being that we need to annotate it with a `tkw.wave(constraints)` decorator. This decorator indicates that the function is a `tkw` kernel. We will describe the `constraints` argument to the decorator later, but for now think of these as constraints that express how we want to partition the operator.

We note that there are 3 arguments to the kernel, `a, b, c`. Each one of these has a `tkl.Memory` type. This type represents n-dimensional data that is stored in the memory hierarchy. `a` has shape `[M, N]` and resides in `ADDRESS_SPACE` with type `tkl.f16`. In `tkw`, we express variable quantities using symbolic variables such as the shape and address space for `a`. An important point to note here is that `[M, N]` is not the shape of the original data. Since we are operating at the wave level, it is the shape of the data that a single wave would operate on. The exact shape is determined through the user-specified `constraints`, but not required during kernel authoring.

Next, we initialize a variable `c_reg` which is of type `tkl.Register`. This type represents data that is stored in virtual registers. We initialize the value of the register to 0.0 in the line below.

```python
c_reg = tkl.Register[M, N, tkl.f32](0.0)
```

Next, we define the core of the operation. A vanilla matrix multiplication implementation would look like below.

```python
for i in range(M):
    for j in range(N):
        c[i, j] = 0.0
        for k in range(K):
            c[i, j] += a[i, k] * b[j, k]
```

Here we do a reduction over the $K$ dimension in the inner-most loop. The equivalent of such for loops in `tkw` is the `tkw.iterate` operator shown below.

```python
@tkw.iterate(K, init_args=[c_reg])
def repeat(acc: tkl.Register[M, N, tkl.f32]) -> tkl.Register[M, N, tkl.f32]:
    a_reg = tkw.read(a, elements_per_thread=LOAD_ELEMS_PER_THREAD)
    b_reg = tkw.read(b, elements_per_thread=LOAD_ELEMS_PER_THREAD)
    acc = tkw.mma(a_reg, b_reg, acc)
    return acc
```

The iterate operator has two arguments: the dimension over which the iteration is taking place and any loop carried variables.

```python
@tkw.iterate(K, init_args=[c_reg])
```
In the example above, we are reducing over the `K` dimension and initializing our loop carried variable to `c_reg`. After this decorator, we specify the body of the loop.

```python
def repeat(acc: tkl.Register[M, N, tkl.f32]) -> tkl.Register[M, N, tkl.f32]:
    a_reg = tkw.read(a, elements_per_thread=LOAD_ELEMS_PER_THREAD)
    b_reg = tkw.read(b, elements_per_thread=LOAD_ELEMS_PER_THREAD)
    acc = tkw.mma(a_reg, b_reg, acc)
    return acc
```

Here the loop carried variable is passed in as the input argument `acc` and this function needs to return a single value which will be used in the next iteration of the reduction. Unlike for loops in imperative languages, here we don't need to specify the induction variable, loop bounds or step size as that is inferred by the compiler.

Now, let's take a look at the contents of the body of the reduction.

```python
a_reg = tkw.read(a, elements_per_thread=LOAD_ELEMS_PER_THREAD)
b_reg = tkw.read(b, elements_per_thread=LOAD_ELEMS_PER_THREAD)
acc = tkw.mma(a_reg, b_reg, acc)
return acc
```

First, we read from `a` into the `a_reg` register and from `b` into the `b_reg` register. During this read, each thread loads `LOAD_ELEMS_PER_THREAD` along the contiguous dimension. After reading the operands, we pass them into the `tkw.mma` operator which performs the matrix multiplication and returns the result.

After the loop, we write out the result to `c`.

```python
tkw.write(repeat, c, elements_per_thread=STORE_ELEMS_PER_THREAD)
```
Here the name of the function that is the body of the `tkw.iterate` represents the result of the reduction.
This is passed to `tkw.write` which writes out the result to `c`. The semantics of `elements_per_thread` are the same as those for `tkw.read`.

This concludes the kernel implementation. Now, we move on to the constraints.

## Constraints

The kernel above captures the computations that a single wave does, but does not provide information about how we want to distribute the computation among the different hardware elements of the GPU. In order to do that, we need to specify constraints. The constraints for this GEMM are shown below.

```python
constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
constraints += [tkw.TilingConstraint(K, BLOCK_K)]
constraints += [tkw.WaveConstraint(M, BLOCK_M / 2)]
constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]

constraints += [
    tkw.HardwareConstraint(threads_per_wave=64,
    mma_type=tkw.MMAType.F32_16x16x16_F16)
]
```

Since $M$ and $N$ are parallel dimensions in this problem, we can distribute them among workgroups and waves. We specify this as follows.

```python
constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
```

This reads as "distribute dimension `M` with a tile size of `BLOCK_M` along workgroup dimension 0". So if we have a tensor of shape `[M, N]`, then with this constraint, each workgroup would be operating on a tile of size `[BLOCK_M, N]`.

Once we have a workgroup constraint in place, we add further wave constraints on the same dimension.

```python
constraints += [tkw.WaveConstraint(M, BLOCK_M / 2)]
```
The above constraint states that "distribute dimension `M` with a tile size of `BLOCK_M/2` among waves". With the above and this constraint, each wave would see a tile with shape `[BLOCK_M/2, N]`.

We can also specify constraints for iterations. We do this through the use of `TilingConstraint`.

```python
constraints += [tkw.TilingConstraint(K, BLOCK_K)]
```

The tiling constraint specifies the tile size to use for iterations.

Finally, we have the hardware constraint which specifies hardware constraints.

```python
constraints += [
    tkw.HardwareConstraint(threads_per_wave=64,
    mma_type=tkw.MMAType.F32_16x16x16_F16)
]
```

This constraint specifies how many threads each wave has as well as the canonical shape for the program which can be specified either by specifying the MMA type or user-specified vector shapes. The canonical shape is used to specify the minimum granularity of
the operations. So for an MMA of shape 16x16x16, we
will not be computing matrix multiplications of shapes
smaller than what is specified.
