from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def test_data_mrc_dir():
    """Fixture to provide the MRC test data directory."""
    return Path(Path(__file__).parent.joinpath("testdata_mrc"))


@pytest.fixture(scope="session")
def test_data_npy_dir():
    """Fixture to provide the NPY test data directory."""
    return Path(Path(__file__).parent.joinpath("testdata_npy"))


@pytest.fixture(scope="session")
def test_corrupt_file():
    """Fixture to provide the path to a corrupt file for testing."""
    return Path(__file__).parent / "corrupt.mrc"



