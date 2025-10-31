from pathlib import Path
import pytest
import torch

from iree.turbine.kernel.boo.driver import driver

SAMPLE_COMMANDS_PATH = Path(driver.__file__).parent / "sample_commands.txt"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="driver requires GPU to run")
@pytest.mark.parametrize(
    "args, expected_exit_code",
    [
        (["conv"], 0),
        (["invalid-command"], 1),
        (["--commands-file", str(SAMPLE_COMMANDS_PATH)], 0),
    ],
)
def test_main(args: list[str], expected_exit_code: int):
    assert driver.main(args) == expected_exit_code
