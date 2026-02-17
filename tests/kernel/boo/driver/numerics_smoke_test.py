# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import torch

from iree.turbine.kernel.boo.driver.numerics import (
    ErrorStatistics,
    NumericsVerdict,
    compute_error_statistics,
    evaluate_statistical_criteria,
    generate_structured_test_pattern,
    format_verdict_simple,
    format_verdict_verbose,
    format_results_table,
)


class TestComputeErrorStatistics:
    """Tests for compute_error_statistics function."""

    def test_basic_statistics(self):
        """Test that basic statistics are computed correctly."""
        errors = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        stats = compute_error_statistics(errors)

        assert stats.mean == pytest.approx(0.0)
        assert stats.stddev == pytest.approx(1.5811388, rel=1e-5)
        assert stats.max_abs_err == pytest.approx(2.0)
        assert stats.num_samples == 5

    def test_zero_errors(self):
        """Test statistics for all-zero errors."""
        errors = torch.zeros(100)
        stats = compute_error_statistics(errors)

        assert stats.mean == pytest.approx(0.0)
        assert stats.stddev == pytest.approx(0.0)
        assert stats.max_abs_err == pytest.approx(0.0)
        assert stats.num_samples == 100

    def test_single_value(self):
        """Test statistics for a single value."""
        errors = torch.tensor([-0.5])
        stats = compute_error_statistics(errors)

        assert stats.mean == pytest.approx(-0.5)
        # Bessel's correction: std of a single sample is NaN (ddof=1)
        import math

        assert math.isnan(stats.stddev)
        assert stats.max_abs_err == pytest.approx(0.5)
        assert stats.num_samples == 1


class TestEvaluateStatisticalCriteria:
    """Tests for evaluate_statistical_criteria function."""

    def test_all_pass(self):
        """Test when all criteria pass."""
        boo_stats = ErrorStatistics(
            mean=1e-8, stddev=1e-5, max_abs_err=1e-4, num_samples=1000
        )
        pytorch_stats = ErrorStatistics(
            mean=1e-8, stddev=1e-5, max_abs_err=1e-4, num_samples=1000
        )

        mean_ok, stddev_ok, reasons = evaluate_statistical_criteria(
            boo_stats,
            pytorch_stats,
            mean_check_atol=1e-6,
            mean_check_rtol=0.0,
            ref_abs_max=0.0,
        )

        assert mean_ok is True
        assert stddev_ok is True
        assert len(reasons) == 0

    def test_mean_too_large(self):
        """Test failure when mean is too large."""
        boo_stats = ErrorStatistics(
            mean=1e-4, stddev=1e-5, max_abs_err=1e-4, num_samples=1000
        )
        pytorch_stats = ErrorStatistics(
            mean=1e-8, stddev=1e-5, max_abs_err=1e-4, num_samples=1000
        )

        mean_ok, stddev_ok, reasons = evaluate_statistical_criteria(
            boo_stats,
            pytorch_stats,
            mean_check_atol=1e-6,
            mean_check_rtol=0.0,
            ref_abs_max=0.0,
        )

        assert mean_ok is False
        assert len(reasons) >= 1
        assert any("Mean error" in r for r in reasons)

    def test_stddev_ratio_too_large(self):
        """Test failure when stddev ratio exceeds tolerance."""
        boo_stats = ErrorStatistics(
            mean=1e-8, stddev=2e-4, max_abs_err=1e-3, num_samples=1000
        )
        pytorch_stats = ErrorStatistics(
            mean=1e-8, stddev=1e-4, max_abs_err=1e-4, num_samples=1000
        )

        mean_ok, stddev_ok, reasons = evaluate_statistical_criteria(
            boo_stats,
            pytorch_stats,
            mean_check_atol=1e-6,
            mean_check_rtol=0.0,
            ref_abs_max=0.0,
            stddev_check_rtol=1.2,
        )

        assert stddev_ok is False
        assert len(reasons) >= 1
        assert any("BOO stddev" in r for r in reasons)

    def test_pytorch_zero_stddev(self):
        """Test handling when PyTorch has zero stddev."""
        boo_stats = ErrorStatistics(
            mean=1e-8, stddev=0.0, max_abs_err=1e-8, num_samples=1000
        )
        pytorch_stats = ErrorStatistics(
            mean=0.0, stddev=0.0, max_abs_err=0.0, num_samples=1000
        )

        mean_ok, stddev_ok, reasons = evaluate_statistical_criteria(
            boo_stats,
            pytorch_stats,
            mean_check_atol=1e-6,
            mean_check_rtol=0.0,
            ref_abs_max=0.0,
        )

        # Both have zero stddev (below floor), should pass
        assert stddev_ok is True

    def test_custom_thresholds(self):
        """Test with custom threshold values."""
        boo_stats = ErrorStatistics(
            mean=5e-5, stddev=1e-4, max_abs_err=1e-3, num_samples=1000
        )
        pytorch_stats = ErrorStatistics(
            mean=1e-8, stddev=5e-5, max_abs_err=1e-4, num_samples=1000
        )

        # With strict threshold, this should fail
        mean_ok, stddev_ok, _ = evaluate_statistical_criteria(
            boo_stats,
            pytorch_stats,
            mean_check_atol=1e-6,
            mean_check_rtol=0.0,
            ref_abs_max=0.0,
        )
        assert mean_ok is False  # 5e-5 > 1e-6
        assert stddev_ok is False  # 2.0 ratio > 1.2

        # With relaxed thresholds, should pass
        mean_ok, stddev_ok, _ = evaluate_statistical_criteria(
            boo_stats,
            pytorch_stats,
            mean_check_atol=1e-4,
            mean_check_rtol=0.0,
            ref_abs_max=0.0,
            stddev_check_rtol=3.0,
        )
        assert mean_ok is True
        assert stddev_ok is True

    def test_allclose_mean_with_large_ref(self):
        """Test that mean threshold scales with ref_abs_max."""
        boo_stats = ErrorStatistics(
            mean=5e-2, stddev=1e-3, max_abs_err=1e-1, num_samples=1000
        )
        pytorch_stats = ErrorStatistics(
            mean=1e-8, stddev=1e-3, max_abs_err=1e-1, num_samples=1000
        )

        # With small ref, mean 5e-2 should fail (threshold = 1e-5 + 1e-4*0 = 1e-5)
        mean_ok, _, _ = evaluate_statistical_criteria(
            boo_stats,
            pytorch_stats,
            mean_check_atol=1e-5,
            mean_check_rtol=1e-4,
            ref_abs_max=0.0,
        )
        assert mean_ok is False

        # With large ref, mean 5e-2 should pass (threshold = 1e-5 + 1e-4*1000 = 0.1)
        mean_ok, _, _ = evaluate_statistical_criteria(
            boo_stats,
            pytorch_stats,
            mean_check_atol=1e-5,
            mean_check_rtol=1e-4,
            ref_abs_max=1000.0,
        )
        assert mean_ok is True

    def test_stddev_floor_prevents_false_failure(self):
        """Test that tiny stddevs below the floor don't fail on ratio."""
        # ratio is 1.25 (> default 1.2) but both are negligibly small
        boo_stats = ErrorStatistics(
            mean=0.0, stddev=1e-10, max_abs_err=1e-9, num_samples=1000
        )
        pytorch_stats = ErrorStatistics(
            mean=0.0, stddev=8e-11, max_abs_err=1e-9, num_samples=1000
        )

        _, stddev_ok, reasons = evaluate_statistical_criteria(
            boo_stats,
            pytorch_stats,
            mean_check_atol=1e-5,
            mean_check_rtol=1e-4,
            ref_abs_max=1.0,
            stddev_check_rtol=1.2,
        )
        # Both stddevs are below floor (1e-5 + 1e-4*1.0 = 1.1e-4), so pass
        assert stddev_ok is True
        assert len(reasons) == 0


class TestGenerateStructuredTestPattern:
    """Tests for generate_structured_test_pattern function."""

    def test_basic_pattern(self):
        """Test that pattern has zeros except for a block of ones."""
        tensor = generate_structured_test_pattern(
            shape=(100,),
            dtype=torch.float32,
            device=torch.device("cpu"),
        )

        assert tensor.shape == (100,)
        assert tensor.dtype == torch.float32

        # Check structure: should have zeros and ones
        unique_vals = tensor.unique()
        assert len(unique_vals) == 2
        assert 0.0 in unique_vals
        assert 1.0 in unique_vals

        # Block size should be numel // 4 = 25
        flat = tensor.flatten()
        ones_count = int((flat == 1.0).sum())
        assert ones_count == 25

        # Check that ones are in a contiguous block
        ones_indices = (flat == 1.0).nonzero().squeeze()
        assert ones_indices[-1] - ones_indices[0] + 1 == ones_count

    def test_2d_shape(self):
        """Test pattern generation for 2D tensor."""
        tensor = generate_structured_test_pattern(
            shape=(10, 20),
            dtype=torch.float32,
            device=torch.device("cpu"),
        )

        assert tensor.shape == (10, 20)
        flat = tensor.flatten()

        # Check that ones exist
        assert (flat == 1.0).sum() > 0
        assert (flat == 0.0).sum() > 0

    def test_small_tensor(self):
        """Test pattern generation for tensor smaller than block size limit."""
        tensor = generate_structured_test_pattern(
            shape=(8,),
            dtype=torch.float32,
            device=torch.device("cpu"),
        )

        assert tensor.shape == (8,)
        # block_size = max(1, min(8 // 4, 128)) = 2
        assert int((tensor == 1.0).sum()) == 2

    def test_seed_varies_offset(self):
        """Test that different seeds produce different block offsets."""
        t0 = generate_structured_test_pattern(
            shape=(200,), dtype=torch.float32, device=torch.device("cpu"), seed=0
        )
        t1 = generate_structured_test_pattern(
            shape=(200,), dtype=torch.float32, device=torch.device("cpu"), seed=1
        )

        # Same block size, different offset
        assert int((t0 == 1.0).sum()) == int((t1 == 1.0).sum())
        idx0 = (t0 == 1.0).nonzero().squeeze()[0]
        idx1 = (t1 == 1.0).nonzero().squeeze()[0]
        assert idx0 != idx1

    def test_empty_tensor(self):
        """Test pattern generation for empty tensor."""
        tensor = generate_structured_test_pattern(
            shape=(0,),
            dtype=torch.float32,
            device=torch.device("cpu"),
        )

        assert tensor.shape == (0,)
        assert tensor.numel() == 0

    def test_different_dtypes(self):
        """Test pattern generation with different dtypes."""
        for dtype in [torch.float32, torch.float64, torch.float16]:
            tensor = generate_structured_test_pattern(
                shape=(100,),
                dtype=dtype,
                device=torch.device("cpu"),
            )
            assert tensor.dtype == dtype


class TestOutputFormatters:
    """Tests for verdict formatting functions."""

    def test_format_verdict_simple_pass(self):
        """Test simple format for passing verdict."""
        verdict = NumericsVerdict(command="gemm --size 32", passed=True)
        output = format_verdict_simple(verdict)

        assert "gemm --size 32" in output
        assert "PASS" in output
        assert "FAIL" not in output

    def test_format_verdict_simple_fail(self):
        """Test simple format for failing verdict."""
        verdict = NumericsVerdict(
            command="gemm --size 32",
            passed=False,
            failure_reasons=["Mean too large"],
        )
        output = format_verdict_simple(verdict)

        assert "gemm --size 32" in output
        assert "FAIL" in output

    def test_format_verdict_verbose_with_stats(self):
        """Test verbose format with full statistics."""
        verdict = NumericsVerdict(
            command="gemm --size 32",
            passed=True,
            boo_gpu_err=ErrorStatistics(
                mean=1e-7, stddev=1e-5, max_abs_err=1e-4, num_samples=1000
            ),
            pytorch_gpu_err=ErrorStatistics(
                mean=1e-7, stddev=1e-5, max_abs_err=1e-4, num_samples=1000
            ),
            boo_pytorch_diff=ErrorStatistics(
                mean=1e-8, stddev=1e-6, max_abs_err=1e-5, num_samples=1000
            ),
            mean_near_zero=True,
            stddev_ratio_ok=True,
            structured_test_passed=True,
        )
        output = format_verdict_verbose(verdict)

        assert "gemm --size 32" in output
        assert "Error Statistics" in output
        assert "BOO GPU vs float64 Reference" in output
        assert "PyTorch GPU vs float64 Reference" in output
        assert "Statistical Tests" in output
        assert "Structured Test" in output
        assert "VERDICT: PASS" in output

    def test_format_verdict_verbose_with_error(self):
        """Test verbose format with error message."""
        verdict = NumericsVerdict(
            command="invalid",
            passed=False,
            error_message="Failed to parse command",
        )
        output = format_verdict_verbose(verdict)

        assert "invalid" in output
        assert "ERROR: Failed to parse command" in output
        assert "VERDICT: FAIL" in output

    def test_format_results_table(self):
        """Test table formatting for multiple verdicts."""
        verdicts = [
            NumericsVerdict(command="gemm1", passed=True),
            NumericsVerdict(command="gemm2", passed=False),
            NumericsVerdict(command="conv1", passed=True),
        ]
        output = format_results_table(verdicts)

        assert "Command" in output
        assert "Result" in output
        assert "gemm1" in output
        assert "gemm2" in output
        assert "conv1" in output


class TestNumericsVerdictDataclass:
    """Tests for NumericsVerdict dataclass."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        verdict = NumericsVerdict(command="test", passed=True)

        assert verdict.command == "test"
        assert verdict.passed is True
        assert verdict.boo_gpu_err is None
        assert verdict.pytorch_gpu_err is None
        assert verdict.boo_pytorch_diff is None
        assert verdict.mean_near_zero is True
        assert verdict.stddev_ratio_ok is True
        assert verdict.structured_test_passed is None
        assert verdict.boo_nan_mismatch is False
        assert verdict.pytorch_nan_mismatch is False
        assert verdict.boo_pytorch_nan_mismatch is False
        assert verdict.failure_reasons == []
        assert verdict.error_message is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
class TestVerifyNumericsEndToEnd:
    """End-to-end tests for verify_numerics (requires GPU)."""

    def test_simple_gemm(self):
        """Test verification of a simple GEMM command."""
        from iree.turbine.kernel.boo.driver.numerics import verify_numerics

        commands = ["gemmfp16 --a_w 32 --a_h 32 --b_w 32"]
        verdicts = verify_numerics(
            commands,
            device=0,
            min_samples=100,  # Use fewer samples for faster test
            run_structured_tests=False,  # Skip structured tests for speed
        )

        assert len(verdicts) == 1
        verdict = verdicts[0]
        assert verdict.boo_gpu_err is not None
        assert verdict.pytorch_gpu_err is not None
        assert verdict.boo_pytorch_diff is not None

    def test_invalid_command(self):
        """Test that invalid commands are handled gracefully."""
        from iree.turbine.kernel.boo.driver.numerics import verify_numerics

        commands = ["invalid_op --foo bar"]
        verdicts = verify_numerics(commands, device=0)

        assert len(verdicts) == 1
        verdict = verdicts[0]
        assert verdict.passed is False
        assert verdict.error_message is not None or len(verdict.failure_reasons) > 0

    def test_multiple_commands(self):
        """Test verification of multiple commands."""
        from iree.turbine.kernel.boo.driver.numerics import verify_numerics

        commands = [
            "gemmfp16 --a_w 16 --a_h 16 --b_w 16",
            "gemmfp16 --a_w 32 --a_h 32 --b_w 32",
        ]
        verdicts = verify_numerics(
            commands,
            device=0,
            min_samples=100,
            run_structured_tests=False,
        )

        assert len(verdicts) == 2
        for verdict in verdicts:
            assert verdict.boo_gpu_err is not None
