"""Test for fingerprint features."""
from __future__ import annotations

from trf23 import fingerprint_features as ff


def test_fingerprint_hard_coded(
    basic_mols,
):
    """Test that the fingerprints match hard-coded ones."""

    expected_fp_bit_dicts = [  # for radius=1, use_counts=True
        {2246733040: 1},
        {1173125914: 2, 2245384272: 2, 2246728737: 2, 3542456614: 2},
        {
            864662311: 1,
            1173125914: 1,
            1510461303: 1,
            1535166686: 1,
            2245384272: 3,
            2246728737: 1,
            3542456614: 1,
            4023654873: 1,
        },
        {
            3624155: 1,
            98513984: 3,
            864662311: 1,
            951226070: 2,
            1510461303: 3,
            1535166686: 1,
            2245384272: 5,
            3217380708: 1,
            3218693969: 5,
            4023654873: 1,
            4121755354: 1,
        },
    ]

    fp_dicts = ff.mol_to_fp_dict(
        basic_mols,
        radius=1,
    )
    assert fp_dicts == expected_fp_bit_dicts
