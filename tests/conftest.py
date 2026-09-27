import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from newsclf.synthetic import write_synthetic_data  # noqa: E402


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("data")
    write_synthetic_data(d)
    return d
