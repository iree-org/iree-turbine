Runtime
===========================

The runtime is the component of Wave that is responsible for launching kernels.
Wave compiles the kernels to a vmfb file that is then passed on to
the iree-runtime [1]_.

iree-runtime accepts tensors using the widely adopted DLPack [2]_ format.
Pytorch tensors are converted to DLPack tensors and passed to the runtime
using the `torch.to_dlpack` function as shown below.

.. code-block:: python

    capsule = torch.to_dlpack(arg_tensor)

Wave Runtime
------------

In scenarios where the latencies associated with `torch.to_dlpack` are
unacceptable, the Wave runtime can be used to launch kernels directly.
The Wave runtime is a C++ extension that uses nanobind [3]_ to interface with
inputs from Wave. In order to enable the wave runtime, first install
the `wave-runtime` pip package.

.. code-block:: python

     pip install -r requirements-wave-runtime.txt

Then modify the `run_config` as shown below.

.. code-block:: python

    run_config["wave_runtime"] = True
    with tk.gen.TestLaunchContext(
        run_config=run_config,
        ...
    ) as context:
        call_kernel(...)



.. [1] https://iree.dev/reference/bindings/c-api/#runtime-api
.. [2] https://dmlc.github.io/dlpack/latest/
.. [3] https://nanobind.readthedocs.io/en/latest/
