import tempfile
import os
import pytest

from breds.commons import blocks


@pytest.fixture
def tmp_file():
    """Create a temporary file with 100 lines, each containing a single byte"""
    with tempfile.NamedTemporaryFile(mode="wt", delete=False) as f:
        for _ in range(100):
            f.write("\n")
        f.flush()
        tmp_path = f.name  # store path before file is closed
    yield tmp_path
    os.remove(tmp_path)  # clean up after the test


def test_blocks(tmp_file):
    """Test that the blocks function returns the correct number of lines"""
    with open(tmp_file, "rb") as f_in:
        assert sum(bl.count(b"\n") for bl in blocks(f_in)) == 100
