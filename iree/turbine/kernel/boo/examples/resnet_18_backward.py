# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import contextlib
from functools import partial

try:
    import torchvision
except ImportError as e:
    raise ImportError(
        "resnet 18 example requires torchvision package. E.g. pip install torchvision."
    )

import torch
from torch.profiler import profile, ProfilerActivity

import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from iree.turbine.kernel.boo.modeling import replace_conv2d_with_boo_conv

# load and modify the model:


def get_model(
    use_direct_replacement: bool = False,
    compile: bool = False,
    compile_backend: str = "iree_boo",
    memory_format: torch.memory_format = torch.channels_last,
    **boo_conv_kwargs,
) -> torch.nn.Module:
    """Sets up a resnet 18 model on cuda device based on provided parameters."""
    # base model
    resnet_model = torchvision.models.resnet18(pretrained=False)
    # boo replacement
    resnet_model = (
        replace_conv2d_with_boo_conv(resnet_model, **boo_conv_kwargs)
        if use_direct_replacement
        else resnet_model
    )
    # move to gpu and apply memory format
    resnet_model = resnet_model.to(device="cuda", memory_format=memory_format)
    # compile model
    resnet_model = (
        torch.compile(resnet_model, dynamic=False, backend=compile_backend)
        if compile
        else resnet_model
    )
    return resnet_model


def get_train_loader(batch_size=128, image_size=256, num_workers=2):
    transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    train_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    return train_loader


def train_loop(
    trace_path,
    resnet_model,
    train_loader,
    memory_format=torch.channels_last,
    autocast_dtype=torch.bfloat16,
    do_warmup=True,
):
    """Runs a training loop and generates a profile to `trace_path`."""
    device = "cuda"
    # Define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(resnet_model.parameters(), lr=0.001, momentum=0.9)
    resnet_model.train()

    # grab some warmup args
    if do_warmup:
        inputs, _ = train_loader.dataset[0]
        new_shape = [train_loader.batch_size] + list(inputs.shape)
        inputs = inputs.expand(new_shape)
        labels = torch.zeros(
            [train_loader.batch_size], dtype=torch.int64, device=device
        )
        inputs = inputs.to(device=device, memory_format=memory_format)
        with torch.amp.autocast(
            device_type=torch.device(device).type, dtype=autocast_dtype
        ):
            outputs = resnet_model(inputs)
            loss = criterion(outputs, labels)

        loss.backward()

    def dump_profile(profiler: profile):
        profiler.export_chrome_trace(trace_path)

    set_num = 15

    def profiler_ctx(iter_num: int):
        if set_num == iter_num:
            return profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
                on_trace_ready=partial(dump_profile),
            )
        else:
            return contextlib.nullcontext()

    # Training loop
    epochs = 3
    for epoch in range(epochs):
        running_loss = 0.0
        for idx, (inputs, labels) in enumerate(train_loader):
            # For sake of time, only running a few data points
            if idx > 10:
                break
            with profiler_ctx(epoch * 10 + idx):
                inputs, labels = inputs.to(
                    device=device, memory_format=memory_format
                ), labels.to(device=device)

                # Do forward and loss calculation in autocast context.
                with torch.amp.autocast(
                    device_type=torch.device(device).type, dtype=autocast_dtype
                ):
                    outputs = resnet_model(inputs)  # Forward pass
                    loss = criterion(outputs, labels)  # Calculate loss

                optimizer.zero_grad()  # Zero the gradients
                loss.backward()  # Backpropagate
                optimizer.step()  # Update weights

                running_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")


def _get_argparse():
    parser = argparse.ArgumentParser(
        description="A script for generating profiles with boo convs."
    )
    parser.add_argument(
        "trace_path",
        type=str,
        help="Specify a path to save a resnet trace json file.",
    )
    parser.add_argument(
        "-c",
        "--compile",
        action="store_true",
        default=False,
        help="Set this to use torch.compile on the model",
    )
    parser.add_argument(
        "-b",
        "--compile_backend",
        type=str,
        help="Specify a torch.compile backend. E.g. 'iree_boo' or 'iree_boo_inductor'",
    )
    parser.add_argument(
        "-r",
        "--use_direct_replacement",
        action="store_true",
        default=False,
        help="Set this to replace Conv2d modules with BooConv2d",
    )
    parser.add_argument(
        "-p",
        "--pytorch_layout",
        action="store_true",
        default=False,
        help="Set this to use NCHW format. Default layout is torch.channels_last.",
    )
    parser.add_argument(
        "-B",
        "--batch_size",
        type=int,
        default=128,
        help="Specify a batch size for the train loader.",
    )
    parser.add_argument(
        "-s",
        "--image_size",
        type=int,
        default=256,
        help="Specify the spatial dim size (square) for the model inputs.",
    )
    return parser


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("GPU expected for running this script.")
        import sys

        sys.exit(1)
    args = _get_argparse().parse_args()
    memory_format = (
        torch.contiguous_format if args.pytorch_layout else torch.channels_last
    )
    resnet_model = get_model(
        args.use_direct_replacement,
        args.compile,
        args.compile_backend,
        memory_format,
        stride=(1, 1),
    )
    # Ensure model is in training mode. E.g. BatchNorm2d will update running stats.
    resnet_model.train()
    train_loader = get_train_loader(args.batch_size, args.image_size)
    train_loop(
        args.trace_path,
        resnet_model,
        train_loader,
        memory_format,
        autocast_dtype=torch.bfloat16,
    )
