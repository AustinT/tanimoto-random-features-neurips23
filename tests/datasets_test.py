from trf23.datasets import guacamol


def test_guacamol():
    smiles = guacamol.get_guacamol_smiles()
    assert len(smiles) == 10_000
