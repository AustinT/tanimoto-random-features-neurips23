from __future__ import annotations

import pytest
from rdkit import Chem
from rdkit.Chem import Crippen

from trf23.datasets.guacamol import get_guacamol_smiles


@pytest.fixture
def basic_smiles() -> list[str]:
    """A basic set of SMILES to test fingerprints."""
    return ["C", "CCCC", "CCCC[OH]", "c1ccccc1CCCCC[OH]"]


@pytest.fixture
def smiles_and_logp() -> tuple[list[str], list[float]]:
    """Labelled dataset of smiles and logP values."""
    smiles = get_guacamol_smiles(100)
    logp_values = [Crippen.MolLogP(Chem.MolFromSmiles(s)) for s in smiles]
    return smiles, logp_values
