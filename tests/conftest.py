from __future__ import annotations

import pytest


@pytest.fixture
def basic_smiles() -> list[str]:
    """A basic set of SMILES to test fingerprints."""
    return ["C", "CCCC", "CCCC[OH]", "c1ccccc1CCCCC[OH]"]
