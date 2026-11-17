import importlib.util
from pathlib import Path

import pytest
from synthetic_epts import write_epts

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


@pytest.fixture
def run_script():
    """run(name, *args): scripts/<name>.py's main() in-process, faster than a subprocess."""

    def run(name, *args):
        spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.main([str(arg) for arg in args])

    return run


@pytest.fixture
def epts(tmp_path):
    """(meta_path, raw_path) of a fresh synthetic EPTS pair."""
    return write_epts(tmp_path)
