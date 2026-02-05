# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for BOO backends with DistributedDataParallel (DDP).

Exercises the DDPOptimizer graph-splitting and fakify_first_call codepaths
that previously broke with double aot_autograd wrapping.
"""

import os
import socket
from unittest.mock import patch
import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP


def _find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


class SmallModel(nn.Module):
    """Small model that exercises BOO fusion patterns (conv + bn + relu)."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(8)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(8, 4)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x


_DDP_ENV_KEYS = ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE")


@pytest.fixture
def ddp_env(request):
    """Set up single-process DDP with gloo backend for testing.

    This is sufficient to exercise DDPOptimizer's graph splitting and
    fakify_first_call codepath without requiring multiple processes.

    Pass dist_backend via indirect parametrization to override the default
    (gloo). Gloo supports both CPU and CUDA tensors.
    """
    dist_backend = getattr(request, "param", "gloo")
    saved = {k: os.environ.get(k) for k in _DDP_ENV_KEYS}
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(_find_free_port())
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    dist.init_process_group(backend=dist_backend, rank=0, world_size=1)
    yield
    dist.destroy_process_group()
    torch.compiler.reset()
    for key, val in saved.items():
        if val is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = val


def _train_step(model, input_tensor):
    """Run a single forward + backward training step."""
    output = model(input_tensor)
    loss = output.sum()
    loss.backward()
    return output


@pytest.mark.parametrize("backend_name", ["iree_boo", "iree_boo_inductor"])
class TestDDPThenCompile:
    """Test DDP wrapping before torch.compile (triggers DDPOptimizer)."""

    def test_output_shape(self, ddp_env, backend_name):
        model = SmallModel()
        ddp_model = DDP(model)
        compiled = torch.compile(ddp_model, backend=backend_name)

        x = torch.randn(2, 3, 8, 8)
        output = compiled(x)
        assert output.shape == (2, 4)

    def test_gradients_exist(self, ddp_env, backend_name):
        model = SmallModel()
        ddp_model = DDP(model)
        compiled = torch.compile(ddp_model, backend=backend_name)

        x = torch.randn(2, 3, 8, 8)
        _train_step(compiled, x)

        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

    def test_multiple_training_steps(self, ddp_env, backend_name):
        """Multiple steps catch the '2nd invocation returns fakes' bug."""
        model = SmallModel()
        ddp_model = DDP(model)
        compiled = torch.compile(ddp_model, backend=backend_name)

        x = torch.randn(2, 3, 8, 8)
        for step in range(3):
            model.zero_grad()
            output = _train_step(compiled, x)
            assert output.shape == (2, 4), f"Bad shape at step {step}"
            assert not output.is_meta, f"Got fake/meta tensor at step {step}"
            assert output.device.type == "cpu", f"Wrong device at step {step}"


@pytest.mark.parametrize("backend_name", ["iree_boo", "iree_boo_inductor"])
class TestCompileThenDDP:
    """Test torch.compile before DDP wrapping (simpler path, no DDPOptimizer)."""

    def test_output_shape(self, ddp_env, backend_name):
        model = SmallModel()
        compiled = torch.compile(model, backend=backend_name)
        ddp_model = DDP(compiled)

        x = torch.randn(2, 3, 8, 8)
        output = ddp_model(x)
        assert output.shape == (2, 4)

    def test_gradients_exist(self, ddp_env, backend_name):
        model = SmallModel()
        compiled = torch.compile(model, backend=backend_name)
        ddp_model = DDP(compiled)

        x = torch.randn(2, 3, 8, 8)
        _train_step(ddp_model, x)

        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

    def test_multiple_training_steps(self, ddp_env, backend_name):
        model = SmallModel()
        compiled = torch.compile(model, backend=backend_name)
        ddp_model = DDP(compiled)

        x = torch.randn(2, 3, 8, 8)
        for step in range(3):
            model.zero_grad()
            output = _train_step(ddp_model, x)
            assert output.shape == (2, 4), f"Bad shape at step {step}"
            assert not output.is_meta, f"Got fake/meta tensor at step {step}"


@pytest.mark.parametrize("backend_name", ["iree_boo", "iree_boo_inductor"])
def test_ddp_correctness(ddp_env, backend_name):
    """Compare DDP + compile outputs/gradients vs compile-only.

    Verifies that DDP wrapping doesn't corrupt compilation results.
    """
    torch.manual_seed(42)
    model_ddp = SmallModel()
    torch.manual_seed(42)
    model_ref = SmallModel()

    ddp_model = DDP(model_ddp)
    compiled_ddp = torch.compile(ddp_model, backend=backend_name)
    compiled_ref = torch.compile(model_ref, backend=backend_name)

    x = torch.randn(2, 3, 8, 8)

    out_ddp = compiled_ddp(x)
    out_ddp.sum().backward()

    out_ref = compiled_ref(x)
    out_ref.sum().backward()

    # With world_size=1, DDP outputs should match non-DDP exactly
    torch.testing.assert_close(out_ddp, out_ref, atol=1e-5, rtol=1e-5)

    for (name, p_ddp), (_, p_ref) in zip(
        model_ddp.named_parameters(), model_ref.named_parameters()
    ):
        assert p_ddp.grad is not None, f"No DDP gradient for {name}"
        assert p_ref.grad is not None, f"No ref gradient for {name}"
        torch.testing.assert_close(
            p_ddp.grad,
            p_ref.grad,
            atol=1e-5,
            rtol=1e-5,
            msg=f"Gradient mismatch for {name}",
        )


def test_inductor_backward_sees_fusion_transform():
    """Verify that fusion_transform is invoked on the backward graph.

    The iree_boo_inductor backend uses separate fw_compiler and bw_compiler
    that both call fusion_transform. This test confirms the backward compiler
    actually receives and processes a graph through fusion_transform.
    """
    from iree.turbine.dynamo.backends import boo as boo_module

    call_count = 0
    original_fusion_transform = boo_module.fusion_transform

    def tracking_fusion_transform(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original_fusion_transform(*args, **kwargs)

    # Build a fresh backend so the closures capture the patched reference.
    with patch.object(boo_module, "fusion_transform", tracking_fusion_transform):
        test_backend = boo_module.backend(nested_backend="inductor")
        model = SmallModel()
        compiled = torch.compile(model, backend=test_backend)
        x = torch.randn(2, 3, 8, 8)
        output = compiled(x)
        output.sum().backward()

    # fw_compiler and bw_compiler should each call fusion_transform at least once
    assert call_count >= 2, (
        f"Expected fusion_transform to be called for both forward and backward, "
        f"but it was called {call_count} time(s)"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("backend_name", ["iree_boo", "iree_boo_inductor"])
class TestDDPCuda:
    """CUDA variants using the shared ddp_env fixture."""

    def test_ddp_then_compile(self, ddp_env, backend_name):
        device = torch.device("cuda")
        model = SmallModel().to(device)
        ddp_model = DDP(model)
        compiled = torch.compile(ddp_model, backend=backend_name)

        x = torch.randn(2, 3, 8, 8, device=device)
        for step in range(3):
            model.zero_grad()
            output = _train_step(compiled, x)
            assert output.shape == (2, 4), f"Bad shape at step {step}"
            assert not output.is_meta, f"Got fake/meta tensor at step {step}"

    def test_compile_then_ddp(self, ddp_env, backend_name):
        device = torch.device("cuda")
        model = SmallModel().to(device)
        compiled = torch.compile(model, backend=backend_name)
        ddp_model = DDP(compiled)

        x = torch.randn(2, 3, 8, 8, device=device)
        for step in range(3):
            model.zero_grad()
            output = _train_step(ddp_model, x)
            assert output.shape == (2, 4), f"Bad shape at step {step}"
            assert not output.is_meta, f"Got fake/meta tensor at step {step}"
