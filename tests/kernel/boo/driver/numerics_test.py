# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import torch

from iree.turbine.kernel.boo.driver import numerics
from iree.turbine.kernel.boo.driver.numerics import (
    ErrorStatistics,
    NumericsVerdict,
    compute_error_statistics,
    evaluate_statistical_criteria,
    generate_structured_test_pattern,
    format_verdict_simple,
    format_verdict_verbose,
    format_results_table,
    SCIPY_AVAILABLE,
)


class TestComputeErrorStatistics:
    """Tests for compute_error_statistics function."""

    def test_basic_statistics(self):
        """Test that basic statistics are computed correctly."""
        errors = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        stats = compute_error_statistics(errors)

        assert stats.mean == pytest.approx(3.0)
        assert stats.stddev == pytest.approx(1.4142135, rel=1e-5)
        assert stats.max_abs_err == pytest.approx(5.0)
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
        errors = torch.tensor([0.5])
        stats = compute_error_statistics(errors)

        assert stats.mean == pytest.approx(0.5)
        assert stats.stddev == pytest.approx(0.0)
        assert stats.max_abs_err == pytest.approx(0.5)
        assert stats.num_samples == 1

    def test_shapiro_pvalue_with_scipy(self):
        """Test that Shapiro p-value is computed when scipy is available."""
        # Generate normally distributed data
        torch.manual_seed(42)
        errors = torch.randn(100).abs()
        stats = compute_error_statistics(errors)

        if SCIPY_AVAILABLE:
            assert stats.shapiro_pvalue is not None
        else:
            assert stats.shapiro_pvalue is None

    def test_large_sample_shapiro_limit(self):
        """Test that Shapiro test is limited to 5000 samples."""
        # This should not crash even with large samples
        errors = torch.randn(10000).abs()
        stats = compute_error_statistics(errors)
        assert stats.num_samples == 10000
        # Shapiro should still work (on first 5000 samples)
        if SCIPY_AVAILABLE:
            assert stats.shapiro_pvalue is not None


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

        mean_ok, stddev_ok, norm_ok, reasons = evaluate_statistical_criteria(
            boo_stats, pytorch_stats
        )

        assert mean_ok is True
        assert stddev_ok is True
        assert norm_ok is True
        assert len(reasons) == 0

    def test_mean_too_large(self):
        """Test failure when mean is too large."""
        boo_stats = ErrorStatistics(
            mean=1e-4, stddev=1e-5, max_abs_err=1e-4, num_samples=1000
        )
        pytorch_stats = ErrorStatistics(
            mean=1e-8, stddev=1e-5, max_abs_err=1e-4, num_samples=1000
        )

        mean_ok, stddev_ok, norm_ok, reasons = evaluate_statistical_criteria(
            boo_stats, pytorch_stats, mean_threshold=1e-6
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

        mean_ok, stddev_ok, norm_ok, reasons = evaluate_statistical_criteria(
            boo_stats, pytorch_stats, stddev_tolerance=1.2
        )

        assert stddev_ok is False
        assert len(reasons) >= 1
        assert any("Stddev ratio" in r for r in reasons)

    def test_pytorch_zero_stddev(self):
        """Test handling when PyTorch has zero stddev."""
        boo_stats = ErrorStatistics(
            mean=1e-8, stddev=0.0, max_abs_err=1e-8, num_samples=1000
        )
        pytorch_stats = ErrorStatistics(
            mean=0.0, stddev=0.0, max_abs_err=0.0, num_samples=1000
        )

        mean_ok, stddev_ok, norm_ok, reasons = evaluate_statistical_criteria(
            boo_stats, pytorch_stats
        )

        # Both have zero stddev, should pass
        assert stddev_ok is True

    def test_normality_failure(self):
        """Test failure when normality test fails."""
        boo_stats = ErrorStatistics(
            mean=1e-8,
            stddev=1e-5,
            max_abs_err=1e-4,
            num_samples=1000,
            shapiro_pvalue=0.01,  # Below default alpha=0.05
        )
        pytorch_stats = ErrorStatistics(
            mean=1e-8, stddev=1e-5, max_abs_err=1e-4, num_samples=1000
        )

        mean_ok, stddev_ok, norm_ok, reasons = evaluate_statistical_criteria(
            boo_stats, pytorch_stats, alpha=0.05
        )

        assert norm_ok is False
        assert len(reasons) >= 1
        assert any("Normality" in r for r in reasons)

    def test_custom_thresholds(self):
        """Test with custom threshold values."""
        boo_stats = ErrorStatistics(
            mean=5e-5, stddev=1e-4, max_abs_err=1e-3, num_samples=1000
        )
        pytorch_stats = ErrorStatistics(
            mean=1e-8, stddev=5e-5, max_abs_err=1e-4, num_samples=1000
        )

        # With default thresholds, this should fail
        mean_ok, stddev_ok, _, _ = evaluate_statistical_criteria(
            boo_stats, pytorch_stats
        )
        assert mean_ok is False  # 5e-5 > 1e-6
        assert stddev_ok is False  # 2.0 ratio > 1.2

        # With relaxed thresholds, should pass
        mean_ok, stddev_ok, _, _ = evaluate_statistical_criteria(
            boo_stats, pytorch_stats, mean_threshold=1e-4, stddev_tolerance=3.0
        )
        assert mean_ok is True
        assert stddev_ok is True


class TestGenerateStructuredTestPattern:
    """Tests for generate_structured_test_pattern function."""

    def test_basic_pattern(self):
        """Test that pattern has zeros except for a block of ones."""
        tensor = generate_structured_test_pattern(
            shape=(100,),
            dtype=torch.float32,
            device=torch.device("cpu"),
            block_offset=10,
            block_size=20,
        )

        assert tensor.shape == (100,)
        assert tensor.dtype == torch.float32

        # Check structure: should have zeros and ones
        unique_vals = tensor.unique()
        assert len(unique_vals) == 2
        assert 0.0 in unique_vals
        assert 1.0 in unique_vals

        # Check that ones are in a contiguous block
        flat = tensor.flatten()
        ones_mask = flat == 1.0
        ones_indices = ones_mask.nonzero().squeeze()
        assert ones_indices[0] == 10
        assert len(ones_indices) == 20

    def test_2d_shape(self):
        """Test pattern generation for 2D tensor."""
        tensor = generate_structured_test_pattern(
            shape=(10, 20),
            dtype=torch.float32,
            device=torch.device("cpu"),
            block_offset=17,
            block_size=73,
        )

        assert tensor.shape == (10, 20)
        flat = tensor.flatten()

        # Check that ones exist
        assert (flat == 1.0).sum() > 0
        assert (flat == 0.0).sum() > 0

    def test_small_tensor(self):
        """Test pattern generation for tensor smaller than block size."""
        tensor = generate_structured_test_pattern(
            shape=(50,),
            dtype=torch.float32,
            device=torch.device("cpu"),
            block_offset=17,
            block_size=73,  # Larger than tensor
        )

        assert tensor.shape == (50,)
        # Should still have some ones (adjusted to fit)
        assert (tensor == 1.0).sum() > 0

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
            normality_ok=True,
            structured_test_passed=True,
        )
        output = format_verdict_verbose(verdict)

        assert "gemm --size 32" in output
        assert "Error Statistics" in output
        assert "BOO GPU vs f64 Reference" in output
        assert "PyTorch GPU vs f64 Reference" in output
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
        assert verdict.normality_ok is True
        assert verdict.structured_test_passed is None
        assert verdict.failure_reasons == []
        assert verdict.error_message is None

    def test_failure_reasons_list(self):
        """Test that failure_reasons is a proper list."""
        verdict1 = NumericsVerdict(command="test1", passed=False)
        verdict2 = NumericsVerdict(command="test2", passed=False)

        # Ensure lists are independent
        verdict1.failure_reasons.append("reason1")
        assert "reason1" not in verdict2.failure_reasons


class TestErrorStatisticsDataclass:
    """Tests for ErrorStatistics dataclass."""

    def test_required_fields(self):
        """Test that required fields must be provided."""
        stats = ErrorStatistics(
            mean=1.0, stddev=0.5, max_abs_err=2.0, num_samples=100
        )

        assert stats.mean == 1.0
        assert stats.stddev == 0.5
        assert stats.max_abs_err == 2.0
        assert stats.num_samples == 100
        assert stats.shapiro_pvalue is None

    def test_with_shapiro_pvalue(self):
        """Test with optional shapiro_pvalue."""
        stats = ErrorStatistics(
            mean=1.0,
            stddev=0.5,
            max_abs_err=2.0,
            num_samples=100,
            shapiro_pvalue=0.25,
        )

        assert stats.shapiro_pvalue == 0.25


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
