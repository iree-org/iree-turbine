import torch
from iree.turbine.kernel.boo.conv_exports import (
    ConvSignature,
    get_launchable,
    clear_cache_dir,
)
from time import time
from iree.turbine.support.logging import runtime_logger as logger

# run this script with the env variable: TURBINE_DEBUG="log_level=DEBUG"


def main():
    # get a conv signature
    sig = ConvSignature(input_shape=[1, 2, 16, 16], kernel_shape=[1, 2, 2, 2])

    # get a launchable kernel
    conv = get_launchable(sig)

    # set a device
    torch_device = torch.device("cuda:0") if torch.cuda.is_available() else None

    # get some random sample args on a specific device
    args_0 = sig.get_sample_conv_args(seed=10, device=torch_device)

    # call the launchable (log the time)
    t0 = time()
    y = conv(*args_0)
    run_time_0 = (time() - t0) * 1000
    logger.debug("first launchable call time : %fms", run_time_0)

    # get some new inputs on the same device
    args_1 = sig.get_sample_conv_args(seed=9, device=torch_device)

    # run a second time. This should take significantly less time (no compilation)
    t0 = time()
    y = conv(*args_1)
    run_time_1 = (time() - t0) * 1000
    logger.debug("second launchable call time (same device) : %fms", run_time_1)

    # run a third time, but on a new GPU. This should take a bit longer as we need to assemble a context from the vmfb.
    torch_device = (
        torch.device("cuda:1")
        if torch.cuda.is_available() and torch.cuda.device_count() > 1
        else None
    )
    other_args_0 = sig.get_sample_conv_args(seed=8, device=torch_device)
    t0 = time()
    y = conv(*other_args_0)
    run_time_2 = (time() - t0) * 1000
    logger.debug(
        "third launchable call time (new gpu if multiple gpus available) : %fms",
        run_time_2,
    )

    # run a fourth time. This should take about as long as the second run time.
    other_args_1 = sig.get_sample_conv_args(seed=7, device=torch_device)
    t0 = time()
    y = conv(*other_args_1)
    run_time_3 = (time() - t0) * 1000
    logger.debug("fourth launchable call time : %fms", run_time_3)

    # clear the mlir artifacts from the cache dir (cache is located at ~/.cache/turbine-kernels/boo/ by default)
    clear_cache_dir()


if __name__ == "__main__":
    main()
