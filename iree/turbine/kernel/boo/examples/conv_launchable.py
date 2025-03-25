import torch
from iree.turbine.kernel.boo.conv_exports import (
    ConvSignature,
    get_launchable,
    clear_cache_dir,
)
from time import time
from iree.turbine.support.logging import runtime_logger as logger

# run this script with the env variable: TURBINE_DEBUG="log_level=DEBUG"

devices = [torch.device("cpu")]
if torch.cuda.is_available():
    devices.append(torch.device("cuda:0"))
    if torch.cuda.device_count() > 1:
        devices.append(torch.device("cuda:1"))


def main(clear_cache: bool):
    """Gets a launchable conv kernel and applies it twice on each device in devices."""

    # get a conv signature
    sig = ConvSignature(
        input_shape=[1, 2, 16, 16], kernel_shape=[1, 2, 2, 2], dtype=torch.float32
    )

    # get a launchable kernel
    conv = get_launchable(sig)
    seed = 1
    for torch_device in devices:
        # get some random sample args on a specific device
        args = sig.get_sample_conv_args(seed=seed, device=torch_device)
        seed += 1

        # Call the launchable (log the time). If we have a vmfb file in the cache, this should just load it.
        t0 = time()
        y = conv(*args)
        run_time = (time() - t0) * 1000
        logger.debug("first launchable call time : %fms", run_time)

        # get some new inputs on the same device
        args = sig.get_sample_conv_args(seed=seed, device=torch_device)
        seed += 1

        # run a second time. This should launch from the cached (VmContext, VmFunction) used in the first call.
        t0 = time()
        y = conv(*args)
        run_time = (time() - t0) * 1000
        logger.debug("second launchable call time (same device) : %fms", run_time)

    # clear the mlir artifacts from the cache dir (cache is located at ~/.cache/turbine-kernels/boo/ by default)
    if clear_cache:
        clear_cache_dir()


if __name__ == "__main__":
    # run main twice. The second run should be much faster since it loads vmfb/mlir from file cache.
    main(clear_cache=False)
    main(clear_cache=True)
