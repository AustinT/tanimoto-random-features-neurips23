from __future__ import annotations

import math

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdMolDescriptors

from trf23 import fingerprint_features as ff
from trf23 import tanimoto_functions as tf

RADIUS = 2
NBITS = 1024


def get_count_fingerprint_obj(smiles: str):
    return rdMolDescriptors.GetMorganFingerprint(
        Chem.MolFromSmiles(smiles),
        radius=RADIUS,
        useCounts=True,
    )


def get_binary_fingerprint_obj(smiles: str):
    return rdMolDescriptors.GetMorganFingerprint(
        Chem.MolFromSmiles(smiles),
        radius=RADIUS,
        useCounts=False,
    )


class Test_TMM:
    def test_manual(self):
        """Test a hand-calculated example."""

        x1 = np.asarray([[1.0, 2.0, 3.0, 4.0], [0.0, 1.0, 0.0, 2.0]])
        x2 = np.asarray([[1.0, 1.0, 1.0, 1.0], [0.5, 0.5, 0.5, 0.5], [4.0, 3.0, 2.0, 1.0]])
        ans = tf.batch_tmm_sim_np(
            x1,
            x2,
        )
        expected_ans = np.array([[4 / 10, 2 / 10, 6 / 14], [2 / 5, 1 / 4, 2 / 11]])
        assert np.allclose(ans, expected_ans)

    def test_fingerprints_manual(self):
        """Hand-crafted example with fingerprints."""
        fps = ff.fp_dicts_to_arr(ff.smiles_to_fp_dicts(["CCCC", "CCCCCCCC[OH]"], radius=RADIUS), nbits=NBITS)
        ans = float(tf.batch_tmm_sim_np(fps[0:1], fps[1:2]))
        assert math.isclose(ans, 0.1724137931034483)

    def test_fingerprints_rdkit(self, basic_smiles: list[str]):
        """
        Check match between rdkit similarites, min-max function, and min-max kernel.
        """

        # Create fingerprints
        fp_arr = ff.fp_dicts_to_arr(ff.smiles_to_fp_dicts(basic_smiles, radius=RADIUS), nbits=NBITS)
        fp_objs = [get_count_fingerprint_obj(s) for s in basic_smiles]

        # Get similarities from both minmax function and minmax kernel
        ans = tf.batch_tmm_sim_np(fp_arr, fp_arr)

        # Check that both answers match
        for i in range(ans.shape[0]):
            for j in range(ans.shape[1]):
                assert math.isclose(
                    ans[i, j].item(),
                    DataStructs.TanimotoSimilarity(fp_objs[i], fp_objs[j]),
                    abs_tol=1e-6,
                )


class Test_TDP:
    def test_manual(self):
        """Test hand-calculated examples."""

        # 1D example
        x1 = np.asarray([[1.0, 0.0, 1.0]])
        x2 = np.asarray([[1.0, 0.0, 0.0]])
        ans = tf.batch_tdp_sim_np(x1, x2)
        assert ans.shape == (1, 1)
        assert np.allclose(ans, 0.5)

        # Multidimensional example (binary)
        x1 = np.asarray([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        x2 = np.asarray([[1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0]])
        ans = tf.batch_tdp_sim_np(x1, x2)
        expected_ans = np.array([[1.0, 2 / 3, 1 / 3], [0.0, 1 / 3, 1 / 2]])
        assert ans.shape == (2, 3)
        assert np.allclose(ans, expected_ans)

        # Multidimensional example (non-binary)
        x1 = np.asarray([[2.0, 0.0], [2.0, 4.0]])
        x2 = np.asarray([[1.0, 0.0], [2.0, 1.0]])
        ans = tf.batch_tdp_sim_np(x1, x2)
        expected_ans = np.array([[2 / 3, 4 / 5], [2 / 19, 8 / 17]])
        assert np.allclose(ans, expected_ans)

    def test_fingerprints_manual(self):
        """Test an example using fingerprints."""
        fps = ff.fp_dicts_to_arr(ff.smiles_to_fp_dicts(["CCC", "CC[OH]"], radius=RADIUS), nbits=NBITS, binarize=True)
        ans = float(tf.batch_tdp_sim_np(fps[0:1], fps[1:2]))
        expected_ans = 0.4285714
        assert math.isclose(ans, expected_ans, abs_tol=1e-4)

    def test_fingerprints_rdkit(self, basic_smiles: list[str]):
        """
        Check that output matches rdkit's Tanimoto function function.

        Also use this opportunity to test the kernel's output.
        """

        for smiles1 in basic_smiles:
            for smiles2 in basic_smiles:
                # Calculate expected answer with rdkit
                fp1 = get_binary_fingerprint_obj(smiles1)
                fp2 = get_binary_fingerprint_obj(smiles2)
                expected_ans = DataStructs.TanimotoSimilarity(fp1, fp2)

                # Calculate answer with this method
                fps = ff.fp_dicts_to_arr(
                    ff.smiles_to_fp_dicts([smiles1, smiles2], radius=RADIUS), nbits=NBITS, binarize=True
                )
                ans = float(tf.batch_tdp_sim_np(fps[:1], fps[1:]))
                assert math.isclose(ans, expected_ans, rel_tol=1e-4, abs_tol=1e-6)
