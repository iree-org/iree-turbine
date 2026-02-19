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
- boo_gpu_err: BOO GPU output vs high-precision CPU reference (ground truth)
- pytorch_gpu_err: PyTorch GPU output vs high-precision CPU reference
- boo_pytorch_diff: BOO GPU vs PyTorch GPU (same-precision comparison)
"""

import shlex
from dataclasses import dataclass, field
from typing import Optional, Sequence

import torch

from iree.turbine.kernel.boo.exports.signature import OpSignature
from iree.turbine.kernel.boo.op_exports.registry import BooOpRegistry

# Constants
MIN_SAMPLES_DEFAULT = 1000
STDDEV_CHECK_RTOL_DEFAULT = 1.2
MEAN_CHECK_ATOL_DEFAULT = 1e-5
MEAN_CHECK_RTOL_DEFAULT = 0.2
STDDEV_CHECK_ATOL_DEFAULT = 1e-5
STRUCTURED_TEST_ATOL = 1e-7
STRUCTURED_BLOCK_SIZE_LIMIT = 128


@dataclass
class ErrorStatistics:
    """Statistics computed from element-wise errors."""

    mean: float
    stddev: float
    max_abs_err: float
    num_samples: int


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
    structured_test_passed: Optional[bool] = None
    boo_nan_mismatch: bool = False
    pytorch_nan_mismatch: bool = False
    boo_pytorch_nan_mismatch: bool = False
    failure_reasons: list[str] = field(default_factory=list)
    error_message: Optional[str] = None
    reference_dtype: torch.dtype = torch.float64


@dataclass
class ErrorSampleResult:
    """Result of collecting error samples across batches."""

    boo_gpu_errors: Optional[torch.Tensor] = None
    pytorch_gpu_errors: Optional[torch.Tensor] = None
    boo_pytorch_errors: Optional[torch.Tensor] = None
    output_dtype: Optional[torch.dtype] = None
    error_message: Optional[str] = None
    boo_nan_mismatch: bool = False
    pytorch_nan_mismatch: bool = False
    boo_pytorch_nan_mismatch: bool = False


def compute_cpu_reference(
    sig: OpSignature,
    sample_args: tuple[torch.Tensor, ...],
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, ...]:
    """
    Compute high-precision reference output.

    Casts floating-point inputs to the specified dtype on CPU, runs PyTorch
    reference, keeps outputs in that dtype.
    """
    cpu = torch.device("cpu")
    ref_args = tuple(
        (
            arg.to(device=cpu, dtype=dtype)
            if arg.is_floating_point()
            else arg.to(device=cpu)
        )
        for arg in sample_args
    )

    reference_module = sig.get_nn_module(use_custom=False)

    with torch.no_grad():
        result = reference_module(*ref_args)

    # Wrap single tensor in tuple
    if isinstance(result, torch.Tensor):
        return (result,)
    return tuple(result)


def _compute_element_errors(
    comparee: torch.Tensor,
    reference: torch.Tensor,
) -> tuple[torch.Tensor, bool]:
    """
    Compute element-wise signed errors between two tensors.

    Returns a tuple of (errors, nan_mismatch):
    - errors: 1D tensor of signed errors (comparee - reference) on the
      reference's device, excluding positions where both are NaN. The dtype
      is auto-promoted by PyTorch (e.g. f16 comparee - f64 reference → f64).
    - nan_mismatch: True if one tensor has NaN where the other does not.

    Signed errors allow bias detection: if the mean is near zero, the
    compiler isn't introducing systematic positive/negative drift.
    """
    # Move comparee to reference's device; PyTorch auto-promotes the
    # subtraction to the reference's (higher-precision) dtype.
    comparee_on_ref = comparee.to(device=reference.device)

    comparee_nan = torch.isnan(comparee_on_ref)
    ref_nan = torch.isnan(reference)
    nan_mismatch = bool((comparee_nan != ref_nan).any().item())

    # Exclude positions where both are NaN
    valid = ~(comparee_nan & ref_nan)
    errors = (comparee_on_ref[valid] - reference[valid]).flatten()
    return errors, nan_mismatch


def collect_error_samples(
    sig: OpSignature,
    device: torch.device,
    min_samples: int = MIN_SAMPLES_DEFAULT,
    reference_dtype: torch.dtype = torch.float64,
) -> ErrorSampleResult:
    """
    Collect error samples for statistical analysis.

    If tensor.numel() < min_samples, runs multiple batches with different seeds.

    Returns an ErrorSampleResult.  On error the tensor fields are None,
    output_dtype is None, and error_message is set.
    """
    cpu = torch.device("cpu")

    # Collect errors across batches
    boo_gpu_errors_list: list[torch.Tensor] = []
    pytorch_gpu_errors_list: list[torch.Tensor] = []
    boo_pytorch_errors_list: list[torch.Tensor] = []
    boo_nan_mismatch = False
    pt_nan_mismatch = False
    bp_nan_mismatch = False

    # Get reference and BOO compiled modules
    reference_module = sig.get_nn_module(use_custom=False)
    try:
        boo_module = sig.get_compiled_module(backend="iree_boo_experimental")
    except Exception as e:
        return ErrorSampleResult(error_message=f"BOO compilation failed: {e}")

    main_result_idx = sig.main_result_index
    num_batches: Optional[int] = None
    batch_idx = 0

    while num_batches is None or batch_idx < num_batches:
        seed = batch_idx

        # Generate sample args for this batch
        sample_args = sig.get_sample_args(device=cpu, seed=seed)
        gpu_args = tuple(arg.to(device=device, copy=True) for arg in sample_args)

        # Compute high-precision CPU reference
        cpu_ref = compute_cpu_reference(sig, sample_args, dtype=reference_dtype)

        # On first pass, determine how many batches we need
        if num_batches is None:
            output_numel = cpu_ref[main_result_idx].numel()
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
            return ErrorSampleResult(error_message=f"BOO runtime failed: {e}")

        # Get main result tensors
        cpu_ref_main = cpu_ref[main_result_idx]
        pytorch_gpu_main = pytorch_gpu_result[main_result_idx]
        boo_gpu_main = boo_gpu_result[main_result_idx]

        # Capture output dtype on first batch
        if batch_idx == 0:
            output_dtype = boo_gpu_main.dtype

        # Compute errors, tracking NaN mismatches (but don't exit early)
        boo_errs, boo_nan = _compute_element_errors(boo_gpu_main, cpu_ref_main)
        boo_nan_mismatch = boo_nan_mismatch or boo_nan
        boo_gpu_errors_list.append(boo_errs)

        pt_errs, pt_nan = _compute_element_errors(pytorch_gpu_main, cpu_ref_main)
        pt_nan_mismatch = pt_nan_mismatch or pt_nan
        pytorch_gpu_errors_list.append(pt_errs)

        bp_errs, bp_nan = _compute_element_errors(boo_gpu_main, pytorch_gpu_main)
        bp_nan_mismatch = bp_nan_mismatch or bp_nan
        boo_pytorch_errors_list.append(bp_errs)

        batch_idx += 1

    # Concatenate all errors
    return ErrorSampleResult(
        boo_gpu_errors=torch.cat(boo_gpu_errors_list),
        pytorch_gpu_errors=torch.cat(pytorch_gpu_errors_list),
        boo_pytorch_errors=torch.cat(boo_pytorch_errors_list),
        output_dtype=output_dtype,
        boo_nan_mismatch=boo_nan_mismatch,
        pytorch_nan_mismatch=pt_nan_mismatch,
        boo_pytorch_nan_mismatch=bp_nan_mismatch,
    )


def compute_error_statistics(errors: torch.Tensor) -> ErrorStatistics:
    """
    Compute statistics from a tensor of errors.

    Statistics: mean, stddev, max_abs_err, num_samples.
    """
    return ErrorStatistics(
        mean=float(errors.mean().item()),
        stddev=float(errors.std().item()),
        max_abs_err=float(errors.abs().max().item()),
        num_samples=errors.numel(),
    )


def is_approximately_negligible(
    value: float, reference: float, atol: float, rtol: float
) -> bool:
    """Combined absolute + relative check for negligibility, similar in style to allclose."""
    return abs(value) <= atol + rtol * abs(reference)


def evaluate_statistical_criteria(
    boo_stats: ErrorStatistics,
    pytorch_stats: ErrorStatistics,
    mean_check_atol: float,
    mean_check_rtol: float,
    stddev_check_rtol: float = STDDEV_CHECK_RTOL_DEFAULT,
    stddev_check_atol: float = STDDEV_CHECK_ATOL_DEFAULT,
) -> tuple[bool, bool, list[str]]:
    """
    Evaluate pass/fail criteria for error statistics based on approximate negligibility.

    Checks:
        `mean_near_zero` = `boo_stats.mean` approx. negligible relative to `boo_stats.stddev`.
        `stddev_ok` = `boo_stats.stddev` approx. negligible relative to `pytorch_stats.stddev`.

    The negligibility tolerances should be interpreted as:
        `mean_check_atol`: Low-absolute bar. If |boo_stats.mean| < atol, mean check passes.
        `mean_check_rtol`: Relative to `boo_stats.stddev` (the BOO error spread). A mean that
            is small relative to the error spread indicates no detectable bias.
        `stddev_check_atol`: Low-absolute bar. If boo_stats.stddev < atol, stddev check passes.
        `stddev_check_rtol`: Relative to `pytorch_stats.stddev`. These should be comparable ~ 1.0.

    Returns:
        (mean_near_zero, stddev_ok, failure_reasons)
    """
    failure_reasons: list[str] = []

    mean_near_zero = is_approximately_negligible(
        boo_stats.mean, boo_stats.stddev, atol=mean_check_atol, rtol=mean_check_rtol
    )
    if not mean_near_zero:
        failure_reasons.append(
            f"Mean error |{boo_stats.mean:.6e}| > threshold "
            f"(atol={mean_check_atol:.6e} + "
            f"rtol={mean_check_rtol:.6e} * boo_stddev={boo_stats.stddev:.6e})"
        )

    stddev_ok = is_approximately_negligible(
        boo_stats.stddev,
        pytorch_stats.stddev,
        atol=stddev_check_atol,
        rtol=stddev_check_rtol,
    )
    if not stddev_ok:
        failure_reasons.append(
            f"BOO stddev {boo_stats.stddev:.6e} > threshold "
            f"(atol={stddev_check_atol:.6e} + "
            f"rtol={stddev_check_rtol:.6e} * pytorch_stddev={pytorch_stats.stddev:.6e})"
        )

    return mean_near_zero, stddev_ok, failure_reasons


def _apply_block_pattern(
    tensor: torch.Tensor,
    seed: int = 0,
    block_size_limit: int = STRUCTURED_BLOCK_SIZE_LIMIT,
) -> None:
    """
    Write a contiguous block of 1's into a tensor (in-place).

    The block size is adaptive: up to 1/4 of the tensor, capped by
    block_size_limit. The offset is chosen randomly within valid bounds.
    The caller is responsible for zero-initializing the tensor first.
    """
    flat = tensor.flatten()
    numel = flat.numel()

    if numel > 0:
        block_size = max(1, min(numel // 4, block_size_limit))
        gen = torch.Generator(device="cpu").manual_seed(seed)
        offset = int(torch.randint(0, max(1, numel - block_size), (1,), generator=gen))
        flat[offset : offset + block_size] = 1.0


def generate_structured_test_pattern(
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    seed: int = 0,
    block_size_limit: int = STRUCTURED_BLOCK_SIZE_LIMIT,
) -> torch.Tensor:
    """
    Generate a splat-0 tensor with a single block of 1's.

    Convenience wrapper around _apply_block_pattern for testing.
    """
    tensor = torch.zeros(shape, dtype=dtype, device=device)
    _apply_block_pattern(tensor, seed=seed, block_size_limit=block_size_limit)
    return tensor


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

    # Generate zero-splatted inputs, then apply block-of-ones pattern
    sample_args = sig.get_sample_args(device=cpu, splat_value=0)
    for i, arg in enumerate(sample_args):
        if arg.is_floating_point():
            _apply_block_pattern(arg, seed=i)

    gpu_args = tuple(arg.to(device=device, copy=True) for arg in sample_args)

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
            f"Structured test failed: max_diff={max_diff:.6e} > atol={atol:.6e}",
        )

    return True, None


def verify_numerics(
    commands: Sequence[str],
    *,
    device: int = 0,
    min_samples: int = MIN_SAMPLES_DEFAULT,
    stddev_check_rtol: float = STDDEV_CHECK_RTOL_DEFAULT,
    stddev_check_atol: float = STDDEV_CHECK_ATOL_DEFAULT,
    mean_check_atol: float = MEAN_CHECK_ATOL_DEFAULT,
    mean_check_rtol: float = MEAN_CHECK_RTOL_DEFAULT,
    run_structured_tests: bool = True,
    reference_dtype: torch.dtype = torch.float64,
) -> list[NumericsVerdict]:
    """
    Verify numerics for a list of commands.

    Returns pass/fail verdict with statistics for each command.

    Args:
        commands: List of command strings to verify
        device: GPU device index
        min_samples: Minimum number of error samples to collect
        stddev_check_rtol: Relative tolerance for the stddev check (scaled by pytorch stddev)
        stddev_check_atol: Absolute tolerance for the stddev check
        mean_check_atol: Absolute tolerance for mean bias check
        mean_check_rtol: Relative tolerance for mean bias check (scaled by boo error stddev)
        run_structured_tests: Whether to run structured pattern tests
        reference_dtype: dtype for high-precision CPU reference (default: float64)
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Cannot verify numerics.")

    cuda_device = torch.device(f"cuda:{device}")
    verdicts: list[NumericsVerdict] = []

    for cmd in commands:
        # Parse command
        sig = BooOpRegistry.parse_command(shlex.split(cmd), True)
        if sig is None:
            verdicts.append(
                NumericsVerdict(
                    command=cmd,
                    passed=False,
                    error_message=f"Failed to parse command: {cmd}",
                    failure_reasons=["Parse error"],
                    reference_dtype=reference_dtype,
                )
            )
            continue

        # Collect error samples
        result = collect_error_samples(sig, cuda_device, min_samples, reference_dtype)

        if result.error_message is not None:
            verdicts.append(
                NumericsVerdict(
                    command=cmd,
                    passed=False,
                    error_message=result.error_message,
                    failure_reasons=["Execution error"],
                    reference_dtype=reference_dtype,
                )
            )
            # Reset torch.compile cache
            torch.compiler.reset()
            continue

        # Compute statistics
        assert (
            result.boo_gpu_errors is not None
            and result.pytorch_gpu_errors is not None
            and result.boo_pytorch_errors is not None
            and result.output_dtype is not None
        )
        boo_stats = compute_error_statistics(result.boo_gpu_errors)
        pytorch_stats = compute_error_statistics(result.pytorch_gpu_errors)
        boo_pytorch_stats = compute_error_statistics(result.boo_pytorch_errors)

        # Evaluate criteria
        mean_near_zero, stddev_ratio_ok, failure_reasons = (
            evaluate_statistical_criteria(
                boo_stats,
                pytorch_stats,
                mean_check_atol,
                mean_check_rtol,
                stddev_check_rtol,
                stddev_check_atol,
            )
        )

        # Report NaN mismatches.  Only BOO-vs-reference NaN mismatch is a
        # hard failure; PyTorch GPU may legitimately produce different NaN
        # patterns from the high-precision reference.
        if result.boo_nan_mismatch:
            failure_reasons.append("NaN mismatch: BOO vs CPU reference")
        if result.pytorch_nan_mismatch:
            failure_reasons.append(
                "NaN mismatch [warning]: PyTorch GPU vs CPU reference"
            )
        if result.boo_pytorch_nan_mismatch:
            failure_reasons.append("NaN mismatch [warning]: BOO vs PyTorch GPU")

        # Run structured test if requested
        structured_passed: Optional[bool] = None
        if run_structured_tests:
            structured_passed, struct_err = run_structured_test(sig, cuda_device)
            if not structured_passed and struct_err:
                failure_reasons.append(struct_err)

        # Determine overall pass/fail
        statistical_passed = (
            mean_near_zero and stddev_ratio_ok and not result.boo_nan_mismatch
        )
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
            structured_test_passed=structured_passed,
            boo_nan_mismatch=result.boo_nan_mismatch,
            pytorch_nan_mismatch=result.pytorch_nan_mismatch,
            boo_pytorch_nan_mismatch=result.boo_pytorch_nan_mismatch,
            failure_reasons=failure_reasons,
            reference_dtype=reference_dtype,
        )
        verdicts.append(verdict)

        # Reset torch.compile cache
        torch.compiler.reset()

    return verdicts


def format_verdict_simple(verdict: NumericsVerdict) -> str:
    """Format verdict as a single line with brief failure reasons."""
    if verdict.passed:
        return "Numerics: PASS"
    reasons = []
    if not verdict.mean_near_zero:
        reasons.append("mean bias")
    if not verdict.stddev_ratio_ok:
        reasons.append("stddev")
    if verdict.boo_nan_mismatch:
        reasons.append("NaN mismatch")
    if verdict.structured_test_passed is False:
        reasons.append("structured test")
    if verdict.error_message:
        reasons.append("error")
    if not reasons:
        reasons.append("unknown")
    return f"Numerics: FAIL ({', '.join(reasons)})"


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
        return [
            f"  {name}:",
            f"    mean:     {stats.mean:.6e}",
            f"    stddev:   {stats.stddev:.6e}",
            f"    max_abs:  {stats.max_abs_err:.6e}",
            f"    samples:  {stats.num_samples}",
        ]

    ref_label = str(verdict.reference_dtype).replace("torch.", "")
    lines.extend(
        _format_stats(f"BOO GPU vs {ref_label} Reference", verdict.boo_gpu_err)
    )
    lines.append("")
    lines.extend(
        _format_stats(f"PyTorch GPU vs {ref_label} Reference", verdict.pytorch_gpu_err)
    )
    lines.append("")
    lines.extend(_format_stats("BOO GPU vs PyTorch GPU", verdict.boo_pytorch_diff))
    lines.append("")

    lines.append("Statistical Tests:")
    lines.append(f"  Mean near zero:    {'PASS' if verdict.mean_near_zero else 'FAIL'}")
    lines.append(
        f"  Stddev check:      {'PASS' if verdict.stddev_ratio_ok else 'FAIL'}"
    )
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
    """Format all verdicts as a summary list."""
    return "\n".join(format_verdict_simple(v) for v in verdicts)
