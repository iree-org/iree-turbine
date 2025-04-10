import csv
import json

from pathlib import Path
from typing import Dict, Union, Sequence

import torch

from iree.turbine.kernel.boo.conv_exports import (
    get_launchable,
    ConvSignature,
    Mode,
    command_to_signature,
)
from iree.turbine.kernel.boo.conv_exports.generate import _load_commands


def compare(y: torch.Tensor, y_ref: torch.Tensor) -> Dict[str, Union[bool, float]]:
    y_ref = y_ref.to(device=y.device)
    are_close = torch.allclose(y, y_ref, rtol=1e-4, atol=1e-5)
    d = y - y_ref
    rel_d = d / y_ref
    m_d = torch.max(torch.abs(d))
    m_rd = torch.max(torch.abs(rel_d))
    return {"pass": are_close, "max abs diff": m_d.item(), "max rel diff": m_rd.item()}


def _run(commands: Sequence[str], allow_jit_compile: bool):
    if not torch.cuda.is_available():
        raise RuntimeError("No cuda drivers found: Cannot run tests.")
    results = dict()
    cuda = torch.device("cuda:0")
    cpu = torch.device("cpu")
    for c in commands:
        sig = command_to_signature(c)
        print(c)
        is_fwd = sig.mode == Mode.parse("fwd")
        # get reference fwd nn module and sample args
        if is_fwd:
            m = sig.get_nn_module(use_custom=False)
            x, w = sig.get_sample_conv_args(device=cuda, seed=4)
        else:
            kwargs = sig._signature._asdict()
            kwargs.pop("num_spatial_dims")
            kwargs["mode"] = "fwd"
            fwd_sig = ConvSignature(**kwargs)
            x, w = fwd_sig.get_sample_conv_args(device=cuda, seed=4)
            m = fwd_sig.get_nn_module(use_custom=False)
        # get a launchable
        try:
            launch = get_launchable(sig, cache_only=(not allow_jit_compile))
        except Exception as e:
            print(e)
            results[c] = f"Failed import to MLIR."
            continue

        x_cpu = x.to(device=cpu, copy=True)
        w_cpu = w.to(device=cpu, copy=True)
        x_cpu.requires_grad = True
        w_cpu.requires_grad = True
        y_ref_cpu = m(x_cpu, w_cpu)

        x.requires_grad = True
        w.requires_grad = True
        y_ref = m(x, w)

        if is_fwd:
            try:
                y = launch(x.detach(), w.detach())
            except Exception as e:
                print(e)
                results[c] = "Failed launch (compile/runtime error)."
                continue
            results[c] = {
                "gpu vs. pt_gpu": compare(y, y_ref),
                "gpu vs. pt_cpu": compare(y, y_ref_cpu),
                "pt_gpu vs. pt_cpu": compare(y_ref, y_ref_cpu),
            }
            continue

        # compute backward value
        y_ref.retain_grad()
        loss = torch.sum((y_ref * y_ref) / 2)
        loss.backward()
        dLdy = y_ref.grad

        loss_cpu = torch.sum((y_ref_cpu * y_ref_cpu) / 2)
        loss_cpu.backward()

        if sig.mode == Mode.parse("bwd"):
            try:
                dLdx = launch(dLdy.detach(), w.detach())
            except Exception as e:
                print(e)
                results[c] = "Failed launch (compile/runtime error)."
                continue
            results[c] = {
                "gpu vs. pt_gpu": compare(dLdx, x.grad),
                "gpu vs. pt_cpu": compare(dLdx, x_cpu.grad),
                "pt_gpu vs. pt_cpu": compare(x.grad, x_cpu.grad),
            }
            continue

        # is wrw conv:
        try:
            dLdw = launch(dLdy.detach(), x.detach())
        except Exception as e:
            print(e)
            results[c] = "Failed launch (compile/runtime error)."
            continue
        results[c] = {
            "gpu vs. pt_gpu": compare(dLdw, w.grad),
            "gpu vs. pt_cpu": compare(dLdw, w_cpu.grad),
            "pt_gpu vs. pt_cpu": compare(w.grad, w_cpu.grad),
        }

    return results


def _get_arg_parser():
    import argparse

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
    return parser


def main(args):
    commands = _load_commands(args.commands_file)
    results = _run(commands, args.allow_jit_compile)
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


if __name__ == "__main__":
    parser = _get_arg_parser()
    args = parser.parse_args()
    main(args)
