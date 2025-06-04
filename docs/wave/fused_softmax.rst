Fused Softmax Tutorial
======================

This tutorial demonstrates how to implement Fused Softmax using Wave.

Softmax Function
----------------

The softmax function is defined as:

.. math::

   \mathrm{softmax}(\mathbf{z})_i = \frac{e^{z_i}}{\sum_{j=1}^{N} e^{z_j}}

Here, :math:`\mathbf{z}` is a vector where each value :math:`z_i` is the output of a neural network. The softmax function converts these outputs into probabilities by exponentiating each :math:`z_i` and dividing by the sum of all exponentiated outputs.

This function is typically used as the final step of a classification neural network to represent the probability distribution over predicted classes.

Note:
**Rows = samples**, **Columns = classes**

Fused Softmax
-------------

A *fused* softmax implementation combines the exponentiation, summation, and normalization into a single kernel. This avoids intermediate memory reads/writes, reducing latency and improving performance.

The function:

.. code-block:: python

   test_fused_softmax(...)

is defined in `wave_e2e_test.py`. It loads the softmax kernel with the correct constraints and compares the result with PyTorch’s built-in `softmax` function. Input shapes are defined in `test_param.json`.

Key GPU Programming Terms in Wave
---------------------------------

- **Thread**: Smallest unit of execution in GPU.
- **Wave**: A group of threads (typically 32 or 64) that processes a tile. Equivalent to a CUDA warp.
- **Workgroup**: A group of waves. The grid is divided into workgroups, each of which may include one or more waves.

Constraint Creation
-------------------

.. code-block:: python

   M = tkl.sym.M
   N = tkl.sym.N
   ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

   wave_size = 64
   BLOCK_M = 1

   constraints: list[tkw.Constraint] = [
       tkw.HardwareConstraint(
           threads_per_wave=wave_size,
           waves_per_block=(1, 1, 1),
           vector_shapes={M: BLOCK_M, N: N},
       )
   ]

   constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 1)]
   constraints += [tkw.WaveConstraint(M, BLOCK_M)]

Explanation:

- :math:`M` and :math:`N` are symbolic values representing rows and columns of the grid.
- `HardwareConstraint` specifies:
  - 64 threads per wave
  - 1 wave per block in each dimension
  - each wave handles a tile of shape [1, N] (or 1 row)
- `WorkgroupConstraint(M, BLOCK_M, 1)` officially maps the M dimension to the Y-axis (the rows) and partitions the grid into workgroups of size `BLOCK_M`.
- `WaveConstraint(M, BLOCK_M)` once again ensures that each wave covers a tile of 1 row.

Note that here each workgroup contains one wave and both operate on the same partition of data, but that will not always be the case.

Kernel Creation
---------------

Now that constraints are defined, we implement the kernel to match them.

Step 1: Define the kernel

.. code-block:: python

   @tkw.wave(constraints)
   def test(
       a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f32],
       b: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f32]
   ):

Each wave executes this function independently on its designated tile.

.. note::

    Note 1: Kernels written in Wave are generally executed at the wave level, meaning each wave processes its own tile independently. However, Wave provides certain functionalities that allow communication between waves within the same workgroup when needed.

.. note::

    Note 2: The a and b tile inputs are typed as [M, N], but these M and N symbols represent the tile dimensions that each wave is assigned, not the full input grid dimensions (even if the symbols are named the same). For example, suppose the original input grid has size M = 256, and you set a WaveConstraint with BLOCK_M = 64. Then, each wave receives a tile with M = 64, and the value passed into the kernel for M will be 64 — not the full grid size of 256.

In this particular case, the N dimension remains unchanged between the grid and the tile, meaning each wave processes all columns.

Step 2: Write the kernel body

.. code-block:: python

   val = tkw.read(a)
   row_max = tkw.max(val, dim=N)
   row_max_bcast = tkw.broadcast(row_max, [M, N])
   val -= row_max_bcast
   val = tkw.exp(val)
   denominator = tkw.sum(val, dim=N)
   denom_broadcast = tkw.broadcast(denominator, [M, N])
   val = val / denom_broadcast
   tkw.write(val, b)

Explanation:

- `read`: loads a row from memory.
- `max`: computes max value across the row - which we then subtract from each value in row. This is a slight addition to the original softmax equation to improve numerical stability.
- `broadcast`: replicates max or sum values across the row.
- `exp`: applies exponentiation.
- `sum`: computes denominator.
- `write`: stores the result to output buffer.

Each wave performs softmax on its assigned row.
