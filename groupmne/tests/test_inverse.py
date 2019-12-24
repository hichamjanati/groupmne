import numpy as np
from groupmne.inverse import compute_group_inverse
from groupmne import utils
import pytest


def check_coefs_stc_match(coefs, stcs):

    coefs_ = []
    for stc in stcs:
        coefs_.append(stc.data)
    coefs_ = np.array(coefs_)
    coefs_ = np.swapaxes(coefs_, 0, 2)
    np.testing.assert_equal(coefs_, coefs)


@pytest.mark.parametrize("hemi", ["lh", "rh", "both"])
def test_inverse(hemi):
    seed = 42
    rnd = np.random.RandomState(seed)
    n_features = 10
    n_samples = 5
    n_subjects = 2
    n_times = 3
    gains = rnd.randn(n_subjects, n_samples, n_features)
    meeg = rnd.randn(n_subjects, n_samples, n_times)

    group_info = utils._make_fake_group_info(n_sources=n_features,
                                             n_subjects=n_subjects,
                                             hemi=hemi)
    ground_metric = rnd.rand(n_features, n_features)
    epsilon = 1.
    gamma = 0.01
    beta = 0.2
    alpha_ot = 0.01
    alpha = 0.2

    ot_dict = dict(alpha=alpha_ot, beta=beta, epsilon=epsilon,
                   gamma=gamma, M=ground_metric)
    lasso_dict = dict(alpha=alpha)
    estim_params = dict(lasso=lasso_dict,
                        grouplasso=lasso_dict,
                        dirty=dict(alpha=alpha, beta=beta),
                        mtw=ot_dict, remtw=ot_dict)

    for model in ["lasso", "grouplasso", "dirty", "mtw", "remtw"]:
        # print("Doing model %s ... " % model)
        coefs, log = compute_group_inverse(gains, meeg, group_info,
                                           method=model,
                                           depth=0.9,
                                           return_stc=False,
                                           time_independent=True,
                                           **estim_params[model])
        stcs_, log = compute_group_inverse(gains, meeg, group_info,
                                           method=model,
                                           depth=0.9,
                                           return_stc=True,
                                           time_independent=True,
                                           **estim_params[model])

        check_coefs_stc_match(coefs, stcs_)


@pytest.mark.parametrize("hemi", ["lh", "rh", "both"])
def test_time_dependent_group_lasso(hemi):
    seed = 42
    rnd = np.random.RandomState(seed)
    n_features = 10
    n_samples = 5
    n_subjects = 2
    n_times = 3
    gains = rnd.randn(n_subjects, n_samples, n_features)
    meeg = rnd.randn(n_subjects, n_samples, n_times)

    group_info = utils._make_fake_group_info(n_sources=n_features,
                                             n_subjects=n_subjects,
                                             hemi=hemi)
    alpha = 0.2

    coefs, log = compute_group_inverse(gains, meeg, group_info,
                                       method="grouplasso",
                                       depth=0.9,
                                       return_stc=False,
                                       time_independent=False,
                                       alpha=alpha)

    stcs_, log = compute_group_inverse(gains, meeg, group_info,
                                       method="grouplasso",
                                       depth=0.9,
                                       return_stc=True,
                                       time_independent=False,
                                       alpha=alpha)
    check_coefs_stc_match(coefs, stcs_)
