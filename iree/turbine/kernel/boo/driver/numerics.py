# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch
from typing import Sequence, TypeVar
from pathlib import Path
import argparse
import json
import csv


from iree.turbine.kernel.boo.exports.parser import OpCLIParser
from iree.turbine.kernel.boo.driver.launch import get_launchable
from iree.turbine.kernel.boo.driver.utils import load_commands


# TODO: it would be better not to have string-based dictionaries flying around...
def _compare(
    y: torch.Tensor, y_ref: torch.Tensor, suffix: str = ""
) -> dict[str, bool | float]:
    """Compares two tensors and returns a dictionary indicating whether they are
    close, and the maximum absolute and relative difference for an element."""
    y_ref = y_ref.to(device=y.device)
    are_close = torch.allclose(y, y_ref, rtol=1e-4, atol=1e-5)
    d = y - y_ref
    rel_d = d / y_ref
    m_d = torch.max(torch.abs(d))
    m_rd = torch.max(torch.abs(rel_d))
    return {
        f"pass {suffix}".strip(): are_close,
        f"max abs diff {suffix}".strip(): m_d.item(),
        f"max rel diff {suffix}".strip(): m_rd.item(),
    }


def _compare_lists(
    actual: Sequence[torch.Tensor], expected: Sequence[torch.Tensor]
) -> dict[str, bool | float]:
    """Compares two lists of tensors and returns a dictionary indicating, per
    tensor, whether they are close and the maximum absolute and relative
    difference for an element."""
    assert len(actual) == len(
        expected
    ), f"Length mismatch: expected {len(expected)}, got {len(actual)}."
    if len(actual) == 1:
        return _compare(actual[0], expected[0])

    result = {}
    for i, (a, e) in enumerate(zip(actual, expected)):
        result |= _compare(a, e, str(i))
    return result


_T = TypeVar("_T")


def _wrap_in_tuple(x: _T | tuple[_T, ...] | list[_T]) -> tuple[_T, ...]:
    """Wraps the argument into a singleton tuple if it isn't already a tuple."""
    if isinstance(x, tuple):
        return x
    if isinstance(x, list):
        return tuple(x)
    return (x,)


def _run(
    commands: Sequence[str],
    parser_cls: type[OpCLIParser],
    *,
    use_custom: bool,
    allow_jit_compile: bool,
    seed: int | None = None,
    device: int = 0,
) -> dict[str, str | dict[str, bool | float]]:
    """Runs the specified commands and returns the dictionary containing, for
    each command, the tensor numerics comparison result or an error message."""
    if not torch.cuda.is_available():
        raise RuntimeError("No cuda drivers found: Cannot run tests.")

    results = dict()
    cuda = torch.device(f"cuda:{device}")
    cpu = torch.device("cpu")

    for c in commands:
        # TODO: if would be nice not to have parsers flying around...
        sig = parser_cls.command_to_signature(c)
        print(c)
        is_fwd = sig.is_forward
        # get reference fwd nn module and sample args
        if is_fwd:
            m = sig.get_nn_module(use_custom=use_custom)
            args = sig.get_sample_args(device=cuda, seed=seed)
        else:
            fwd_sig = sig.make_signature_copy_for_forward()
            m = fwd_sig.get_nn_module(use_custom=use_custom)
            args = fwd_sig.get_sample_args(device=cuda, seed=seed)
        try:
            launch = get_launchable(sig, cache_only=(not allow_jit_compile))
        except Exception as e:
            print(e)
            results[c] = f"Failed import to MLIR."
            continue

        args_cpu = tuple(x.to(device=cpu, copy=True) for x in args)
        for arg_cpu in args_cpu:
            arg_cpu.requires_grad = True
        results_ref_cpu = _wrap_in_tuple(m(*args_cpu))

        for arg in args:
            arg.requires_grad = True
        results_ref_gpu = _wrap_in_tuple(m(*args))

        # If forward mode was requested, record results and exit here.
        if is_fwd:
            try:
                forward_results = _wrap_in_tuple(
                    launch(*tuple(x.detach() for x in args))
                )
            except Exception as e:
                print(e)
                results[c] = "Failed launch (compile/runtime error)."
                continue
            results[c] = {
                "gpu vs. pt_gpu": _compare_lists(forward_results, results_ref_gpu),
                "gpu vs. pt_cpu": _compare_lists(forward_results, results_ref_cpu),
                "pt_gpu vs. pt_cpu": _compare_lists(results_ref_gpu, results_ref_cpu),
            }
            continue

        # Compute backward values. Only propagate derivatives through the "main"
        # result of the operation. Otherwise we'd need to handle a full Jacobian
        # matrix here with significantly more complexity.
        main_result_ref_gpu = results_ref_gpu[sig.main_result_index]
        main_result_ref_gpu.retain_grad()
        # TODO: this loss function looks like it can easily explode given that
        # values are not normalized. Consider launching the backward kernel
        # directly instead, e.g., torch.ops.aten.convolution_backward (#1021).
        loss = torch.sum((main_result_ref_gpu * main_result_ref_gpu) / 2)
        loss.backward()

        main_result_ref_cpu = results_ref_cpu[sig.main_result_index]
        loss = torch.sum((main_result_ref_cpu * main_result_ref_cpu) / 2)
        loss.backward()

        # Arrange arguments for the backward call with the current mode. These
        # may include arguments of the forward pass as well as its results,
        # usually "saved" for reuse by autograd.
        backward_args = sig.arrange_backward_launch_args(args, results_ref_gpu)
        try:
            # This assumes the backward launch always returns one argument.
            grad = launch(
                *tuple(x.detach() for x in [main_result_ref_gpu, *backward_args])
            )
            assert not isinstance(grad, tuple)
        except Exception as e:
            print(e)
            results[c] = "Failed launch (compile/runtime error)."
            continue
        derivative_arg_index = sig.get_arg_index_for_backward()
        results[c] = {
            "gpu vs. pt_gpu": _compare(grad, args[derivative_arg_index].grad),
            "gpu vs. pt_cpu": _compare(grad, args_cpu[derivative_arg_index].grad),
            "pt_gpu vs. pt_cpu": _compare(
                args[derivative_arg_index].grad, args_cpu[derivative_arg_index].grad
            ),
        }

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
        "--allow-jit-compile",
        "-j",
        action="store_true",
        default=False,
        help="Whether to allow jit compile during runs.",
    )
    parser.add_argument(
        "--seed", type=int, required=False, help="Random number generator seed."
    )
    parser.add_argument("--device", type=int, default=0, help="The device to run on.")
    return parser


def run_numerics(parser_cls: type[OpCLIParser], *, use_custom: bool):
    """Runs numeric tests for operations listed in a commands file provided via
    CLI arguments parseable by the given parser class."""
    parser = _get_arg_parser()
    args = parser.parse_args()
    commands = load_commands(args.commands_file, parser_cls)
    results = _run(
        commands,
        parser_cls,
        allow_jit_compile=args.allow_jit_compile,
        use_custom=use_custom,
    )
    dumps = json.dumps(results, indent=4, separators=(",", " : "))
    if not args.output_file:
        print(dumps)
        return

    output_path = Path(args.output_file)
    print(f"Saving results to {output_path}")
    if output_path.suffix == ".json":
        output_path.write_text(dumps)
        return
    if output_path.suffix != ".csv":
        print(dumps)
        raise NotImplementedError(f"Logs with file extension {output_path.suffix} nyi.")
    expand_key = lambda name, keys: list([f"{name} {k}" for k in keys])

    keys_0 = ["pass", "max abs diff", "max rel diff"]
    keys_1 = ["gpu vs. pt_gpu", "gpu vs. pt_cpu", "pt_gpu vs. pt_cpu"]
    fieldnames = ["name"]
    for k in keys_1:
        fieldnames.extend(expand_key(k, keys_0))

    def default_dict(keys, default):
        return {k: default for k in keys}

    with open(args.output_file, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for name, d in results.items():
            line_dict = {"name": name}
            if not isinstance(d, dict):
                d = default_dict(keys_1, "NA")
            for k1, v1 in d.items():
                if not isinstance(v1, dict):
                    v1 = default_dict(keys_0, "NA")
                    v1["pass"] = False
                for k0, v0 in v1.items():
                    line_dict[f"{k1} {k0}"] = v0
            writer.writerow(line_dict)
