import torch
from iree.turbine.kernel.boo.conv_exports import ConvSignature, get_launchable

# run this script with the env variable: TURBINE_DEBUG="log_level=DEBUG"

# get a conv signature
sig = ConvSignature(input_shape=[1, 2, 16, 16], kernel_shape=[1, 2, 2, 2])

# get a launchable kernel
conv = get_launchable(sig)

# get some sample args on a specific device
torch_device = torch.device("cuda:0") if torch.cuda.is_available() else None
args = sig.get_sample_conv_args(seed=10, device=torch_device)

# call the launchable
y = conv(*args)

# run on the same device:
args = sig.get_sample_conv_args(seed=9, device=torch_device)
y = conv(*args)


# Interestingly, launching on a different GPU results in a memory access fault:
# torch_device = torch.device("cuda:1") if torch.cuda.is_available() else None
# other_args = sig.get_sample_conv_args(seed=8, device=torch_device)
# y = conv(*other_args)
