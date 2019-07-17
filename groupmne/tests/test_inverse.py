import numpy as np
from groupmne.inverse import compute_group_inverse
from groupmne import utils
import pytest
from itertools import product


@pytest.mark.parametrize("hemi, time_independent",
                         product(["lh", "rh", "both"], [False, True]))
def test_inverse(hemi, time_independent):
    seed = 42
    rnd = np.random.RandomState(seed)
    n_features = 10
    n_samples = 5
    n_subjects = 2
    n_times = 2
    gains = rnd.randn(n_subjects, n_samples, n_features)
    M = rnd.randn(n_subjects, n_samples, n_times)

    group_info = utils._make_fake_group_info(n_sources=n_features,
                                             n_subjects=n_subjects,
                                             hemi=hemi)
    stcs, log = compute_group_inverse(gains, M, group_info,
                                      method="grouplasso",
                                      depth=0.9, alpha=0.1, return_stc=True,
                                      time_independent=time_independent,
                                      n_jobs=4)

    coefs, log = compute_group_inverse(gains, M, group_info,
                                       method="grouplasso",
                                       depth=0.9, alpha=0.1, return_stc=False,
                                       time_independent=time_independent,
                                       n_jobs=4)
