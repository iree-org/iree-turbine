from pathlib import Path
import shlex
import pytest
import torch
import tempfile
import csv

from iree.turbine.kernel.boo.driver import driver


def _read_csv_as_dicts(
    path: Path, *, backends: list[str] = [driver.DEFAULT_BACKEND]
) -> list[dict[str, str]]:
    """Reads the csv file at `path`, verifies headers, and returns the data as a list of dicts."""
    assert path.is_file()
    assert path.suffix == ".csv"
    lines = path.read_text().splitlines()
    assert len(lines) > 1
    expected_headers = ["arguments"] + list(
        f"{b} {s}" for b in backends for s in driver.ALL_STATS
    )
    assert lines[0] == ",".join(expected_headers)
    reader = csv.DictReader(lines[1:], fieldnames=expected_headers, strict=True)
    return list(reader)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="driver requires GPU to run")
def test_roundtrip_csv_single_command():
    """Test the csv format is what we expect.

    Verifies a few individual items that should always be true:

    1. A forward conv should be a single dispatch.
    2. The `arguments` should be the conv command string.
    """
    with tempfile.TemporaryDirectory() as td:
        backend = driver.DEFAULT_BACKEND
        csv_file = Path(td) / "conv_stats.csv"
        iters = 4
        meta_args = [f"--csv={csv_file}"]
        command_args = ["convbfp16", "-F=1", f"--iter={iters}"]
        args = meta_args + command_args
        # Check we don't encounter an error.
        assert driver.main(args) == 0
        data = _read_csv_as_dicts(csv_file, backends=[backend])
        # Check we have exactly one command being run in the driver.
        assert len(data) == 1
        row = data[0]
        # Check the arguments column contains the individual command.
        assert row["arguments"] == shlex.join(command_args)
        # Check the number of dispatches is equal to the number of iterations.
        assert row[f"{backend} num_dispatches"] == str(iters)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="driver requires GPU to run")
def test_roundtrip_csv_commands_file():
    """Test the csv format is what we expect when loading multiple commands from a `commands-file`."""
    with tempfile.TemporaryDirectory() as td:
        commands_file = Path(td) / "commands.txt"
        commands = [
            ["convbfp16", "-F", "1", "--iter", "4"],
            [
                "convbfp16",
                "-F",
                "1",
                "--in_layout",
                "NHWC",
                "--fil_layout",
                "NHWC",
                "--out_layout",
                "NHWC",
                "--iter",
                "2",
            ],
        ]
        commands_file.write_text("\n".join([shlex.join(c) for c in commands]))
        backend = driver.DEFAULT_BACKEND
        csv_file = Path(td) / "conv_stats.csv"
        meta_args = [f"--csv={csv_file}", f"--commands-file={commands_file}"]
        args = meta_args
        # Check we don't encounter an error.
        assert driver.main(args) == 0
        data = _read_csv_as_dicts(csv_file, backends=[backend])
        for d, c in zip(data, commands, strict=True):
            # Check the arguments column contains the individual command.
            assert d["arguments"] == shlex.join(c)
            # Check all convs have a single dispatch per launch.
            assert d[f"{backend} num_dispatches"] == c[-1]
