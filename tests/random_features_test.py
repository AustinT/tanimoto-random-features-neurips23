import numpy as np
import pytest

from trf23 import fingerprint_features as ff
from trf23 import tanimoto_functions as tf
from trf23.random_features import tdp as tdp_rf
from trf23.random_features import tmm as tmm_rf

fake_fp1 = {0: 1, 2: 3}
fake_fp2 = {0: 1, 2: 5, 10: 2, 15: 7}
fake_fp3 = {3: 2, 4: 1}


fp_with_sim_examples = [
    ([fake_fp1, fake_fp1], 1.0),  # matching fingerprints should be the same
    ([fake_fp2, fake_fp2], 1.0),
    ([fake_fp1, fake_fp2], 4 / 15),
    ([fake_fp1, fake_fp3], 0.0),
]

NBITS = 16
TMM_FEATURIZER_KWARGS = dict(
    max_fp_dim=NBITS,
    num_features=10_000,
    use_tqdm=False,
    rng=np.random.default_rng(),
    n_jobs=1,
)


@pytest.mark.parametrize("input,match_prob", fp_with_sim_examples)
@pytest.mark.parametrize("distribution", ["Rademacher", "Gaussian"])
def test_TMM_Dict_Featurizer(input, match_prob, distribution):
    featurizer = tmm_rf.TMM_Dict_Featurizer(distribution=distribution, **TMM_FEATURIZER_KWARGS)
    features = featurizer(input)

    # Test: shape is right
    assert features.shape == (len(input), featurizer.num_features)

    # Test: inner prod is right
    normalized_inner_prod = np.dot(features[0], features[1])
    assert abs(normalized_inner_prod - match_prob) < 0.05


@pytest.mark.parametrize("input,match_prob", fp_with_sim_examples)
def test_TMM_Arr_Featurizer(input, match_prob):
    featurizer = tmm_rf.TMM_ArrFeaturizer(**TMM_FEATURIZER_KWARGS)
    features = featurizer(ff.fp_dicts_to_arr(input, nbits=NBITS))

    # Test: shape is right
    assert features.shape == (len(input), featurizer.num_features)

    # Test: inner prod is right
    normalized_inner_prod = np.dot(features[0], features[1])
    assert abs(normalized_inner_prod - match_prob) < 0.05


def test_default_TDP_featurizer():
    # Create normalized fingerprints
    fp_arr = np.sqrt(ff.fp_dicts_to_arr([fake_fp1, fake_fp2, fake_fp3], nbits=NBITS))
    fp_arr /= max(np.linalg.norm(fp_arr, axis=1))

    # True TDP
    tdp_true = tf.batch_tdp_sim_np(fp_arr, fp_arr)

    # Compute random features
    featurizer = tdp_rf.Default_TDP_Featurizer(
        input_dim=NBITS,
        num_rf=100_000,
        max_R=5,
        rng=np.random.default_rng(),
        min_input_square_norm=np.linalg.norm(fp_arr, axis=1).min() ** 2,
        bias_correction="normalize",
    )
    features = featurizer(fp_arr)

    # Compute error
    tdp_approx = features @ features.T
    assert np.allclose(tdp_approx, tdp_true, atol=0.02)
