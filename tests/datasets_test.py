from trf23.datasets import dockstring, guacamol


def test_guacamol():
    smiles = guacamol.get_guacamol_smiles()
    assert len(smiles) == 10_000


def test_dockstring():
    # Load one target
    smiles_train, smiles_test, y_train, y_test = dockstring.get_train_test_smiles("F2")
    assert len(smiles_train) == len(y_train) == 221269
    assert len(smiles_test) == len(y_test) == 38881

    # Load a second target
    smiles_train, smiles_test, y_train, y_test = dockstring.get_train_test_smiles("KIT")
    assert len(smiles_train) == len(y_train) == 221270
    assert len(smiles_test) == len(y_test) == 38880
