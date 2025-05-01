# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# NOTE: This script does not currently work because some bwd conv configs in the model fail compilation

try:
    import torchvision
except ImportError as e:
    raise ImportError(
        "resnet 18 example requires torchvision package. E.g. pip install torchvision."
    )

import torch

from iree.turbine.kernel.boo.modeling import replace_conv2d_with_boo_conv

# load and modify the model:

resnet_model = torchvision.models.resnet18(pretrained=False)
resnet_model = replace_conv2d_with_boo_conv(resnet_model)

device = "cuda" if torch.cuda.is_available() else "cpu"
assert device == "cuda", f"device is {device}."

resnet_model = resnet_model.to(
    dtype=torch.bfloat16, device=device, memory_format=torch.channels_last
)

import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# get a training data loader
transform = transforms.Compose(
    [
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]  # Example normalization
)
train_dataset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)

# Define loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(resnet_model.parameters(), lr=0.001, momentum=0.9)

# Training loop
epochs = 10
for epoch in range(epochs):
    resnet_model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:  # Assuming you have a train_loader
        inputs, labels = inputs.to(device=device, dtype=torch.bfloat16), labels.to(
            device
        )

        optimizer.zero_grad()  # Zero the gradients

        outputs = resnet_model(inputs)  # Forward pass
        loss = criterion(outputs, labels)  # Calculate loss
        loss.backward()  # Backpropagate
        optimizer.step()  # Update weights

        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")
