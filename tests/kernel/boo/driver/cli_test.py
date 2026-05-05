from pathlib import Path

import pytest
import torch

from iree.turbine.kernel.boo.driver import driver


@pytest.mark.skipif(not torch.cuda.is_available(), reason="driver requires GPU to run")
@pytest.mark.parametrize(
    "args, expected_exit_code",
    [
        (["conv"], 0),
        (["invalid-command"], 1),
    ],
)
def test_main(args: list[str], expected_exit_code: int):
    assert driver.main(args) == expected_exit_code


@pytest.mark.skipif(not torch.cuda.is_available(), reason="driver requires GPU to run")
def test_commands_file(tmp_path: Path):
    commands_file = tmp_path / "commands.txt"
    commands_file.write_text(
        "\n".join(
            [
                "conv",
                "# ignored comment",
                "",
                "convbfp16 -F 1",
            ]
        )
    )

    assert (
        driver.main(
            [
                "--commands-file",
                str(commands_file),
                "--iter",
                "1",
                "--no-verbose",
            ]
        )
        == 0
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="driver requires GPU to run")
def test_cache_dir(tmp_path: Path):
    """--cache-dir should populate the cache directory with compiled artifacts."""
    cache_dir = tmp_path / "test-cache"
    assert (
        driver.main(
            [
                "--cache-dir",
                str(cache_dir),
                "--iter",
                "1",
                "conv",
            ]
        )
        == 0
    )
    assert any(cache_dir.iterdir()), f"cache dir {cache_dir} is empty"
