from __future__ import annotations

from pathlib import Path
from typing import Optional


def get_guacamol_smiles(n: Optional[int] = None) -> list[str]:
    smiles_file = Path(__file__).parent / "guacamol_10k.smiles"
    assert smiles_file.exists()
    with open(smiles_file) as f:
        smiles_list = [line.strip() for line in f.readlines()]
    if n is not None:
        smiles_list = smiles_list[:n]
    return smiles_list
