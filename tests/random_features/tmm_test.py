import numpy as np
import pytest

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


@pytest.mark.parametrize("input,match_prob", fp_with_sim_examples)
@pytest.mark.parametrize("distribution", ["Rademacher", "Gaussian"])
def test_feat_inner_prod_close(input, match_prob, distribution):
    featurizer = tmm_rf.TMM_Dict_Featurizer(
        max_fp_dim=16,
        num_features=10_000,
        distribution=distribution,
        use_tqdm=False,
        rng=np.random.default_rng(),
        n_jobs=1,
    )
    features = featurizer(input)

    # Test: shape is right
    assert features.shape == (len(input), featurizer.num_features)

    # Test: inner prod is right
    normalized_inner_prod = np.dot(features[0], features[1])
    assert abs(normalized_inner_prod - match_prob) < 0.05
