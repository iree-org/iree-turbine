from pathlib import Path
import pytest
from iree.turbine.kernel.boo.runtime import set_cache_dir, use_cache_dir


@pytest.fixture(autouse=True)
def _isolate_tests(tmp_path):
    # Ensure each test has its own cache directory, in case the cache is enabled.
    set_cache_dir(tmp_path / ".cache" / "turbine_kernels" / "boo")


@pytest.fixture
def boo_cache_dir(tmp_path: Path):
    # Enable the cache for the duration of the test.
    with use_cache_dir(tmp_path / ".boo_cache") as cache_dir:
        yield cache_dir
