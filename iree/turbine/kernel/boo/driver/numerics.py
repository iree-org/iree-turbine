# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch
from typing import Sequence, TypeVar, TypeAlias, NamedTuple
from pathlib import Path
import argparse
import json
import csv


from iree.turbine.kernel.boo.op_exports.registry import BooOpRegistry
from iree.turbine.kernel.boo.driver.utils import load_commands

ALL_COMPARISONS = (
    ("boo_cpu", "boo_gpu"),
    ("boo_cpu", "pytorch_cpu"),
    ("boo_cpu", "pytorch_gpu"),
    ("boo_gpu", "pytorch_cpu"),
    ("boo_gpu", "pytorch_gpu"),
    ("pytorch_cpu", "pytorch_gpu"),
)

_Command: TypeAlias = str
_ComparisonString: TypeAlias = str


class ComparisonSummary(NamedTuple):
    passed: bool
    max_abs_diff: float
    max_rel_diff: float


ComparisonSummaryAsDict: TypeAlias = dict[str, bool | float]

ComparisonResult: TypeAlias = ComparisonSummaryAsDict | str

ResultSummary: TypeAlias = dict[
    _Command, tuple[dict[_ComparisonString, ComparisonResult], ...]
]


def _compare(
    y: torch.Tensor,
    y_ref: torch.Tensor,
) -> ComparisonSummary:
    """Compares two tensors and returns a dictionary indicating whether they are
    close, and the maximum absolute and relative difference for an element."""
    y_ref = y_ref.to(device=y.device)
    are_close = torch.allclose(y, y_ref, rtol=1e-4, atol=1e-5)
    d = y - y_ref
    rel_d = d / y_ref
    m_d = torch.max(torch.abs(d))
    m_rd = torch.max(torch.abs(rel_d))
    return ComparisonSummary(
        passed=are_close,
        max_abs_diff=m_d.item(),
        max_rel_diff=m_rd.item(),
    )


def _compare_lists(
    comparee: Sequence[torch.Tensor],
    reference: Sequence[torch.Tensor],
    num_results: int,
) -> tuple[ComparisonResult, ...]:
    """Compares two lists of tensors and returns a dictionary indicating, per
    tensor, whether they are close and the maximum absolute and relative
    difference for an element."""
    result_counts = [len(comparee), len(reference)]
    if 0 in result_counts:
        return ("Failure running a command (Compile/Runtime Error).",) * num_results
    if set(result_counts) != {num_results}:
        return (
            f"# results mismatch: expected all comparisons to have {num_results} results. Got {result_counts = }.",
        ) * num_results

    return tuple(_compare(a, e)._asdict() for a, e in zip(comparee, reference))


_T = TypeVar("_T")


def _wrap_in_tuple(x: _T | tuple[_T, ...] | list[_T]) -> tuple[_T, ...]:
    """Wraps the argument into a singleton tuple if it isn't already a tuple."""
    if isinstance(x, tuple):
        return x
    if isinstance(x, list):
        return tuple(x)
    return (x,)


def _run(
    commands: Sequence[_Command],
    *,
    seed: int | None = None,
    device: int = 0,
) -> ResultSummary:
    """Runs the specified commands and returns the dictionary containing, for
    each command, the tensor numerics comparison result or an error message."""
    if not torch.cuda.is_available():
        raise RuntimeError("No cuda drivers found: Cannot run tests.")

    results: ResultSummary = dict()
    cuda = torch.device(f"cuda:{device}")
    cpu = torch.device("cpu")

    for c in commands:
        print(c)
        sig = BooOpRegistry.parse_command(c, True)
        if sig is None:
            raise ValueError(f"Failed parsing a command: {c}.")

        reference_module = sig.get_nn_module(use_custom=False)
        boo_module = torch.compile(
            reference_module, dynamic=False, backend="iree_boo_experimental"
        )
        sample_args = sig.get_sample_args(device=cpu, seed=seed)
        arg_tensors = {
            "cpu": sample_args,
            "gpu": tuple(arg.to(device=cuda, copy=True) for arg in sample_args),
        }

        result_tensors: dict[str, tuple[torch.Tensor, ...]] = {}
        for name in ("cpu", "gpu"):
            result_tensors[f"pytorch_{name}"] = _wrap_in_tuple(
                reference_module(*arg_tensors[name])
            )
            try:
                result_tensors[f"boo_{name}"] = _wrap_in_tuple(
                    boo_module(*arg_tensors[name])
                )
            except Exception as e:
                print(f"Failed running command with boo: {c}\nDevice type: {name}\n{e}")
                result_tensors[f"boo_{name}"] = ()
        num_results = len(result_tensors["pytorch_cpu"])
        num_results_gpu = len(result_tensors["pytorch_gpu"])
        assert (
            num_results == num_results_gpu
        ), f"Expected same number of reference results on cpu vs. gpu. Got {num_results} vs. {num_results_gpu}."
        summary: list[dict[_ComparisonString, ComparisonResult]] = list(
            {} for _ in range(num_results)
        )
        for cmp_key, ref_key in ALL_COMPARISONS:
            comparison_results = _compare_lists(
                result_tensors[cmp_key], result_tensors[ref_key], num_results
            )
            assert len(comparison_results) == num_results
            for i in range(num_results):
                summary[i][f"{cmp_key} vs. {ref_key}"] = comparison_results[i]
        results[c] = tuple(summary)
        # Reset the cache to ensure we don't silently hit re-compile limits.
        torch.compiler.reset()

    return results


def _get_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "commands_file", type=str, help="specify a commands text file to run."
    )
    parser.add_argument(
        "--output-file",
        "-o",
        type=str,
        required=False,
        help="Specify a path to .json or .csv file to store results.",
    )
    parser.add_argument(
        "--seed", type=int, required=False, help="Random number generator seed."
    )
    parser.add_argument("--device", type=int, default=0, help="The device to run on.")
    return parser


def _log_results(results: ResultSummary, output_path: Path | None):
    """Parse output_path suffix and log results."""
    if output_path is not None and output_path.suffix == ".csv":
        return _log_csv(results, output_path)

    dumps = json.dumps(results, indent=4, separators=(",", " : "))

    if output_path is None:
        print(dumps)
        return

    if output_path.suffix == ".json":
        output_path.write_text(dumps)
        return

    print(dumps)
    raise NotImplementedError(f"Logs with file extension {output_path.suffix} nyi.")


def _log_csv(results: ResultSummary, output_path: Path):
    """Logs results to an output path in csv format. Rows are indexed by command and output index."""
    expand_key = lambda name, keys: list([f"{name} {k}" for k in keys])

    keys_0 = ["passed", "max_abs_diff", "max_rel_diff"]
    fieldnames = ["command", "output_index"]
    for k0, k1 in ALL_COMPARISONS:
        fieldnames.extend(expand_key(f"{k0} vs. {k1}", keys_0))

    with output_path.open(mode="w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for cmd, result_summaries in results.items():
            for output_index, result_summary in enumerate(result_summaries):
                line_dict = {"command": cmd, "output_index": output_index}
                for comparison_kind, comparison_result in result_summary.items():
                    if isinstance(comparison_result, dict):
                        for k in keys_0:
                            line_dict[f"{comparison_kind} {k}"] = comparison_result.get(
                                k, "ERR"
                            )
                        continue
                    assert isinstance(comparison_result, str)
                    for k in keys_0:
                        line_dict[f"{comparison_kind} {k}"] = comparison_result
                    line_dict[f"{comparison_kind} passed"] = False

                writer.writerow(line_dict)


def _run_numerics():
    """Runs numeric tests for operations listed in a commands file provided via
    CLI arguments parseable by any registered parser class."""
    parser = _get_arg_parser()
    args = parser.parse_args()
    commands = [c for c in load_commands(args.commands_file) if not c.startswith("#")]
    results = _run(
        commands,
        seed=args.seed,
        device=args.device,
    )
    _log_results(results, None if args.output_file is None else Path(args.output_file))


if __name__ == "__main__":
    _run_numerics()
