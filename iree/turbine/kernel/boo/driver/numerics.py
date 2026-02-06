# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Statistical validation of kernel numerics.

This module provides tools to validate the numerical accuracy of BOO-compiled
kernels by comparing them against PyTorch reference implementations using
statistical analysis.

Key comparisons:
- boo_gpu_err: BOO GPU output vs f64 CPU reference (ground truth)
- pytorch_gpu_err: PyTorch GPU output vs f64 CPU reference
- boo_pytorch_diff: BOO GPU vs PyTorch GPU (same-precision comparison)
"""

import shlex
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import torch

from iree.turbine.kernel.boo.exports.signature import OpSignature
from iree.turbine.kernel.boo.op_exports.registry import BooOpRegistry

# Optional scipy for D'Agostino-Pearson normality test
SCIPY_AVAILABLE = False
try:
    from scipy.stats import normaltest

    SCIPY_AVAILABLE = True
except ImportError:
    pass

# Constants
MIN_SAMPLES_DEFAULT = 1000
STDDEV_TOLERANCE_DEFAULT = 1.2
ALPHA_DEFAULT = 0.05
MEAN_EPS_MULTIPLE_DEFAULT = 10
STRUCTURED_TEST_ATOL = 1e-7
STRUCTURED_BLOCK_SIZE = 73  # Prime number for block size
STRUCTURED_BLOCK_OFFSET = 17  # Irregular offset


@dataclass
class ErrorStatistics:
    """Statistics computed from element-wise errors."""

    mean: float
    stddev: float
    max_abs_err: float
    num_samples: int
    normality_pvalue: Optional[float] = None


@dataclass
class NumericsVerdict:
    """Result of numerics verification for a single command."""

    command: str
    passed: bool
    boo_gpu_err: Optional[ErrorStatistics] = None
    pytorch_gpu_err: Optional[ErrorStatistics] = None
    boo_pytorch_diff: Optional[ErrorStatistics] = None
    mean_near_zero: bool = True
    stddev_ratio_ok: bool = True
    normality_ok: bool = True
    structured_test_passed: Optional[bool] = None
    failure_reasons: list[str] = field(default_factory=list)
    error_message: Optional[str] = None


def compute_f64_cpu_reference(
    sig: OpSignature,
    sample_args: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, ...]:
    """
    Compute high-precision reference output.

    Casts inputs to float64 on CPU, runs PyTorch reference, keeps as float64.
    """
    cpu = torch.device("cpu")
    # Cast to float64 on CPU for high precision reference
    f64_args = tuple(
        (
            arg.to(device=cpu, dtype=torch.float64)
            if arg.is_floating_point()
            else arg.to(device=cpu)
        )
        for arg in sample_args
    )

    reference_module = sig.get_nn_module(use_custom=False)

    with torch.no_grad():
        result = reference_module(*f64_args)

    # Wrap single tensor in tuple
    if isinstance(result, torch.Tensor):
        return (result,)
    return tuple(result)


def _compute_element_errors(
    comparee: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    """
    Compute element-wise signed errors between two tensors.

    Returns a 1D float32 tensor of signed errors (comparee - reference) on
    the comparee's device. Signed errors allow bias detection: if the mean
    is near zero, the compiler isn't introducing systematic positive/negative
    drift.

    Both tensors are cast to float32 to avoid f64 operations on GPU (which
    may not be supported on all hardware) while still providing sufficient
    precision for bf16/f16/f32 error measurement.
    """
    reference_on_device = reference.to(
        device=comparee.device, dtype=torch.float32, non_blocking=True
    )
    errors = (comparee.float() - reference_on_device).flatten()
    return errors


def collect_error_samples(
    sig: OpSignature,
    device: torch.device,
    min_samples: int = MIN_SAMPLES_DEFAULT,
) -> tuple[
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.dtype],
    Optional[str],
]:
    """
    Collect error samples for statistical analysis.

    If tensor.numel() < min_samples, runs multiple batches with different seeds.

    Returns:
        (boo_gpu_errors, pytorch_gpu_errors, boo_pytorch_errors, output_dtype, error_message)
        If an error occurs, the first three are None, output_dtype is None, and error_message is set.
    """
    cpu = torch.device("cpu")

    # Collect errors across batches
    boo_gpu_errors_list: list[torch.Tensor] = []
    pytorch_gpu_errors_list: list[torch.Tensor] = []
    boo_pytorch_errors_list: list[torch.Tensor] = []

    # Get reference and BOO compiled modules
    reference_module = sig.get_nn_module(use_custom=False)
    try:
        boo_module = sig.get_compiled_module(backend="iree_boo_experimental")
    except Exception as e:
        return None, None, None, None, f"BOO compilation failed: {e}"

    main_result_idx = sig.main_result_index
    num_batches: Optional[int] = None
    batch_idx = 0

    while num_batches is None or batch_idx < num_batches:
        seed = batch_idx * 12345 + 42  # Deterministic but varied seeds

        # Generate sample args for this batch
        sample_args = sig.get_sample_args(device=cpu, seed=seed)
        gpu_args = tuple(arg.to(device=device, copy=True) for arg in sample_args)

        # Compute f64 CPU reference
        f64_ref = compute_f64_cpu_reference(sig, sample_args)

        # On first pass, determine how many batches we need
        if num_batches is None:
            output_numel = f64_ref[main_result_idx].numel()
            num_batches = max(1, (min_samples + output_numel - 1) // output_numel)

        # Run PyTorch GPU
        with torch.no_grad():
            pytorch_gpu_result = reference_module(*gpu_args)
        if isinstance(pytorch_gpu_result, torch.Tensor):
            pytorch_gpu_result = (pytorch_gpu_result,)

        # Run BOO GPU
        try:
            with torch.no_grad():
                boo_gpu_result = boo_module(*gpu_args)
            if isinstance(boo_gpu_result, torch.Tensor):
                boo_gpu_result = (boo_gpu_result,)
        except Exception as e:
            return None, None, None, None, f"BOO runtime failed: {e}"

        # Get main result tensors
        f64_ref_main = f64_ref[main_result_idx]
        pytorch_gpu_main = pytorch_gpu_result[main_result_idx]
        boo_gpu_main = boo_gpu_result[main_result_idx]

        # Compute errors
        boo_gpu_errors_list.append(_compute_element_errors(boo_gpu_main, f64_ref_main))
        pytorch_gpu_errors_list.append(
            _compute_element_errors(pytorch_gpu_main, f64_ref_main)
        )
        boo_pytorch_errors_list.append(
            _compute_element_errors(boo_gpu_main, pytorch_gpu_main)
        )

        batch_idx += 1

    # Concatenate all errors
    boo_gpu_errors = torch.cat(boo_gpu_errors_list)
    pytorch_gpu_errors = torch.cat(pytorch_gpu_errors_list)
    boo_pytorch_errors = torch.cat(boo_pytorch_errors_list)

    output_dtype = boo_gpu_main.dtype
    return boo_gpu_errors, pytorch_gpu_errors, boo_pytorch_errors, output_dtype, None


def compute_error_statistics(errors: torch.Tensor) -> ErrorStatistics:
    """
    Compute statistics from a tensor of errors.

    Statistics: mean, stddev, max_abs_err, num_samples, normality_pvalue (if scipy available)
    """
    errors_np = errors.cpu().float().numpy()
    mean = float(errors_np.mean())
    stddev = float(errors_np.std())
    max_abs_err = float(np.abs(errors_np).max())
    num_samples = len(errors_np)

    normality_pvalue: Optional[float] = None
    if SCIPY_AVAILABLE and num_samples >= 20:
        # D'Agostino-Pearson requires at least 20 samples
        try:
            _, normality_pvalue = normaltest(errors_np)
        except Exception:
            # normaltest can fail in edge cases (e.g., all zeros)
            normality_pvalue = None

    return ErrorStatistics(
        mean=mean,
        stddev=stddev,
        max_abs_err=max_abs_err,
        num_samples=num_samples,
        normality_pvalue=normality_pvalue,
    )


def evaluate_statistical_criteria(
    boo_stats: ErrorStatistics,
    pytorch_stats: ErrorStatistics,
    mean_threshold: float,
    stddev_tolerance: float = STDDEV_TOLERANCE_DEFAULT,
    alpha: float = ALPHA_DEFAULT,
) -> tuple[bool, bool, bool, list[str]]:
    """
    Evaluate pass/fail criteria based on statistics.

    Returns:
        (mean_near_zero, stddev_ratio_ok, normality_ok, failure_reasons)
    """
    failure_reasons: list[str] = []

    # Check mean near zero
    mean_near_zero = abs(boo_stats.mean) < mean_threshold
    if not mean_near_zero:
        failure_reasons.append(
            f"Mean error |{boo_stats.mean:.2e}| >= threshold {mean_threshold:.2e}"
        )

    # Check stddev ratio
    # Avoid division by zero; if pytorch stddev is 0, boo stddev should also be 0.
    # This could happen if testing f64 convolutions, for example. This edge case
    # would be pretty much impossible otherwise.
    if pytorch_stats.stddev > 0:
        ratio = boo_stats.stddev / pytorch_stats.stddev
        stddev_ratio_ok = ratio <= stddev_tolerance
        if not stddev_ratio_ok:
            failure_reasons.append(
                f"Stddev ratio {ratio:.2f} > tolerance {stddev_tolerance:.2f}"
            )
    else:
        # If pytorch has zero stddev, boo should also have near-zero stddev
        stddev_ratio_ok = boo_stats.stddev < mean_threshold
        if not stddev_ratio_ok:
            failure_reasons.append(
                f"BOO stddev {boo_stats.stddev:.2e} > 0 while PyTorch stddev is 0"
            )

    # Check normality (if scipy available and pvalue exists)
    normality_ok = True
    if boo_stats.normality_pvalue is not None:
        normality_ok = boo_stats.normality_pvalue > alpha
        if not normality_ok:
            failure_reasons.append(
                f"Normality test failed: p-value {boo_stats.normality_pvalue:.3f} <= alpha {alpha}"
            )

    return mean_near_zero, stddev_ratio_ok, normality_ok, failure_reasons


def generate_structured_test_pattern(
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    block_offset: int = STRUCTURED_BLOCK_OFFSET,
    block_size: int = STRUCTURED_BLOCK_SIZE,
) -> torch.Tensor:
    """
    Generate a splat-0 tensor with a single block of 1's.

    Purpose: Catch indexing bugs as large, obvious differences.

    Args:
        shape: Tensor shape
        dtype: Tensor dtype
        device: Target device
        block_offset: Starting offset of the 1's block (irregular, e.g., 17)
        block_size: Size of the 1's block (prime, e.g., 73)
    """
    tensor = torch.zeros(shape, dtype=dtype, device=device)
    flat = tensor.flatten()
    numel = flat.numel()

    if numel > 0:
        # Ensure block fits within tensor
        actual_offset = block_offset % max(1, numel - block_size)
        actual_size = min(block_size, numel - actual_offset)
        flat[actual_offset : actual_offset + actual_size] = 1.0

    return tensor.view(shape)


def run_structured_test(
    sig: OpSignature,
    device: torch.device,
    atol: float = STRUCTURED_TEST_ATOL,
) -> tuple[bool, Optional[str]]:
    """
    Run structured test to detect indexing bugs.

    Returns:
        (passed, error_message)
    """
    cpu = torch.device("cpu")

    # Generate structured test inputs
    sample_args = sig.get_sample_args(device=cpu, seed=0)

    # Create structured patterns for each input
    structured_args = tuple(
        (
            generate_structured_test_pattern(
                arg.shape,
                arg.dtype,
                cpu,
                block_offset=STRUCTURED_BLOCK_OFFSET + i * 7,  # Vary offset per input
                block_size=STRUCTURED_BLOCK_SIZE,
            )
            if arg.is_floating_point()
            else arg
        )
        for i, arg in enumerate(sample_args)
    )

    gpu_args = tuple(arg.to(device=device, copy=True) for arg in structured_args)

    # Get reference and BOO modules
    reference_module = sig.get_nn_module(use_custom=False)
    try:
        boo_module = sig.get_compiled_module(backend="iree_boo_experimental")
    except Exception as e:
        return False, f"BOO compilation failed: {e}"

    # Run both
    with torch.no_grad():
        pytorch_result = reference_module(*gpu_args)
        try:
            boo_result = boo_module(*gpu_args)
        except Exception as e:
            return False, f"BOO runtime failed: {e}"

    if isinstance(pytorch_result, torch.Tensor):
        pytorch_result = (pytorch_result,)
    if isinstance(boo_result, torch.Tensor):
        boo_result = (boo_result,)

    # Compare main results on GPU
    main_idx = sig.main_result_index
    pytorch_main = pytorch_result[main_idx]
    boo_main = boo_result[main_idx]

    if not torch.allclose(boo_main, pytorch_main, atol=atol, rtol=0):
        max_diff = (boo_main - pytorch_main).abs().max().item()
        return (
            False,
            f"Structured test failed: max_diff={max_diff:.2e} > atol={atol:.2e}",
        )

    return True, None


def verify_numerics(
    commands: Sequence[str],
    *,
    device: int = 0,
    verbose: bool = False,
    min_samples: int = MIN_SAMPLES_DEFAULT,
    stddev_tolerance: float = STDDEV_TOLERANCE_DEFAULT,
    alpha: float = ALPHA_DEFAULT,
    mean_eps_multiple: float = MEAN_EPS_MULTIPLE_DEFAULT,
    run_structured_tests: bool = True,
) -> list[NumericsVerdict]:
    """
    Verify numerics for a list of commands.

    Returns pass/fail verdict with statistics for each command.

    Args:
        commands: List of command strings to verify
        device: GPU device index
        verbose: Print verbose output during verification
        min_samples: Minimum number of error samples to collect
        stddev_tolerance: Maximum allowed ratio of BOO stddev to PyTorch stddev
        alpha: Significance level for normality test
        mean_eps_multiple: The mean bias threshold is computed as
            mean_eps_multiple * torch.finfo(output_dtype).eps
        run_structured_tests: Whether to run structured pattern tests
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Cannot verify numerics.")

    cuda_device = torch.device(f"cuda:{device}")
    verdicts: list[NumericsVerdict] = []

    for cmd in commands:
        if verbose:
            print(f"Verifying: {cmd}")

        # Parse command
        sig = BooOpRegistry.parse_command(shlex.split(cmd), True)
        if sig is None:
            verdicts.append(
                NumericsVerdict(
                    command=cmd,
                    passed=False,
                    error_message=f"Failed to parse command: {cmd}",
                    failure_reasons=["Parse error"],
                )
            )
            continue

        # Collect error samples
        boo_err, pytorch_err, boo_pytorch_err, output_dtype, err_msg = (
            collect_error_samples(sig, cuda_device, min_samples)
        )

        if err_msg is not None:
            verdicts.append(
                NumericsVerdict(
                    command=cmd,
                    passed=False,
                    error_message=err_msg,
                    failure_reasons=["Execution error"],
                )
            )
            # Reset torch.compile cache
            torch.compiler.reset()
            continue

        # Compute statistics
        assert (
            boo_err is not None
            and pytorch_err is not None
            and boo_pytorch_err is not None
            and output_dtype is not None
        )
        boo_stats = compute_error_statistics(boo_err)
        pytorch_stats = compute_error_statistics(pytorch_err)
        boo_pytorch_stats = compute_error_statistics(boo_pytorch_err)

        # Compute mean threshold from dtype epsilon
        mean_threshold = mean_eps_multiple * torch.finfo(output_dtype).eps

        # Evaluate criteria
        mean_near_zero, stddev_ratio_ok, normality_ok, failure_reasons = (
            evaluate_statistical_criteria(
                boo_stats, pytorch_stats, mean_threshold, stddev_tolerance, alpha
            )
        )

        # Run structured test if requested
        structured_passed: Optional[bool] = None
        if run_structured_tests:
            structured_passed, struct_err = run_structured_test(sig, cuda_device)
            if not structured_passed and struct_err:
                failure_reasons.append(struct_err)

        # Determine overall pass/fail
        statistical_passed = mean_near_zero and stddev_ratio_ok and normality_ok
        overall_passed = statistical_passed and (
            structured_passed is None or structured_passed
        )

        verdict = NumericsVerdict(
            command=cmd,
            passed=overall_passed,
            boo_gpu_err=boo_stats,
            pytorch_gpu_err=pytorch_stats,
            boo_pytorch_diff=boo_pytorch_stats,
            mean_near_zero=mean_near_zero,
            stddev_ratio_ok=stddev_ratio_ok,
            normality_ok=normality_ok,
            structured_test_passed=structured_passed,
            failure_reasons=failure_reasons,
        )
        verdicts.append(verdict)

        # Reset torch.compile cache
        torch.compiler.reset()

    return verdicts


def format_verdict_simple(verdict: NumericsVerdict) -> str:
    """Format verdict as a single line."""
    result = "PASS" if verdict.passed else "FAIL"
    return f"{verdict.command:<60} | {result}"


def format_verdict_verbose(verdict: NumericsVerdict) -> str:
    """Format verdict with detailed statistics."""
    lines: list[str] = []
    lines.append(f"=== {verdict.command} ===")
    lines.append("")

    if verdict.error_message:
        lines.append(f"ERROR: {verdict.error_message}")
        lines.append("")
        lines.append(f"VERDICT: {'PASS' if verdict.passed else 'FAIL'}")
        return "\n".join(lines)

    lines.append("Error Statistics:")

    def _format_stats(name: str, stats: Optional[ErrorStatistics]) -> list[str]:
        if stats is None:
            return [f"  {name}: N/A"]
        result = [
            f"  {name}:",
            f"    mean:     {stats.mean:.2e}",
            f"    stddev:   {stats.stddev:.2e}",
            f"    max_abs:  {stats.max_abs_err:.2e}",
            f"    samples:  {stats.num_samples}",
        ]
        if stats.normality_pvalue is not None:
            result.append(f"    normality: p={stats.normality_pvalue:.3f}")
        return result

    lines.extend(_format_stats("BOO GPU vs f64 Reference", verdict.boo_gpu_err))
    lines.append("")
    lines.extend(_format_stats("PyTorch GPU vs f64 Reference", verdict.pytorch_gpu_err))
    lines.append("")
    lines.extend(_format_stats("BOO GPU vs PyTorch GPU", verdict.boo_pytorch_diff))
    lines.append("")

    lines.append("Statistical Tests:")
    lines.append(f"  Mean near zero:    {'PASS' if verdict.mean_near_zero else 'FAIL'}")
    lines.append(
        f"  Stddev ratio:      {'PASS' if verdict.stddev_ratio_ok else 'FAIL'}"
    )
    lines.append(f"  Normality (BOO):   {'PASS' if verdict.normality_ok else 'FAIL'}")
    lines.append("")

    if verdict.structured_test_passed is not None:
        lines.append(
            f"Structured Test:    {'PASS' if verdict.structured_test_passed else 'FAIL'}"
        )
        lines.append("")

    if verdict.failure_reasons:
        lines.append("Failure Reasons:")
        for reason in verdict.failure_reasons:
            lines.append(f"  - {reason}")
        lines.append("")

    lines.append(f"VERDICT: {'PASS' if verdict.passed else 'FAIL'}")

    return "\n".join(lines)


def format_results_table(verdicts: list[NumericsVerdict]) -> str:
    """Format all verdicts as a summary table."""
    lines: list[str] = []
    lines.append(f"{'Command':<60} | Result")
    lines.append("-" * 60 + "-|-------")
    for verdict in verdicts:
        lines.append(format_verdict_simple(verdict))
    return "\n".join(lines)


def print_verification_summary(
    verdicts: list[NumericsVerdict], verbose: bool = False
) -> None:
    """Print verification summary to stdout."""
    if verbose:
        for verdict in verdicts:
            print(format_verdict_verbose(verdict))
            print()
    else:
        print(format_results_table(verdicts))

    # Summary line
    passed = sum(1 for v in verdicts if v.passed)
    total = len(verdicts)
    print()
    print(f"Summary: {passed}/{total} commands passed")
