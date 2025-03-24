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

    # Interestingly, launching on a different GPU results in a memory access fault:
    # torch_device = torch.device("cuda:1") if torch.cuda.is_available() else None
    # other_args = sig.get_sample_conv_args(seed=8, device=torch_device)
    # y = conv(*other_args)

    # clear the cache dir (cache is located at ~/.cache/turbine-kernels/boo/ by default)
    # clear_cache_dir()


if __name__ == "__main__":
    main()
