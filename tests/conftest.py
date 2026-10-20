import tempfile
from pathlib import Path

import pytest

# project-local scratch: some Windows profiles deny access to the global Temp
# folder that pytest's tmp_path would use
_SCRATCH = Path(__file__).resolve().parent.parent / ".pytest-tmp"


@pytest.fixture()
def scratch_dir():
    _SCRATCH.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=_SCRATCH) as d:
        yield Path(d)
