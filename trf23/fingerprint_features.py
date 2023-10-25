"""Code for getting features from molecules."""
from __future__ import annotations

from typing import Dict

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

FP_Dict = Dict[int, int]  # Dict not dict to support earlier python versions


def _smiles_to_mols(smiles_list: list[str]) -> list[Chem.Mol]:
    """
    Helper function to convert list of SMILES to list of mols,
    raising an error if any invalid SMILES are found.
    """

    # Define a separate function since rdkit functions cannot be pickled by joblib
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    assert None not in mols, "Invalid SMILES found"
    return mols


def mol_to_fp_dict(
    mols: list[Chem.Mol],
    radius: int,
) -> list[FP_Dict]:
    """Get Morgan fingerprint bit dict from a list of mols."""
    out: list[FP_Dict] = list()
    for mol in mols:
        fp = rdMolDescriptors.GetMorganFingerprint(mol, radius=radius, useCounts=True)
        out.append(fp.GetNonzeroElements())
    return out


def fp_dicts_to_arr(fp_dicts: list[FP_Dict], nbits: int, binarize: bool = False) -> np.ndarray:
    """Convert a list of fingerprint dicts to a numpy array."""

    # Fold fingerprints into array
    out = np.zeros((len(fp_dicts), nbits))
    for i, fp in enumerate(fp_dicts):
        for k, v in fp.items():
            out[i, k % nbits] += v

    # Potentially binarize
    if binarize:
        out = np.minimum(out, 1.0)
        assert set(np.unique(out)) <= {0.0, 1.0}

    return out


def smiles_to_fp_dicts(smiles_list: list[str], radius: int) -> list[FP_Dict]:
    """Convert a list of SMILES to a list of fingerprint dicts."""
    mol_list = _smiles_to_mols(
        smiles_list,
    )
    return mol_to_fp_dict(mols=mol_list, radius=radius)
