import os
import os.path as op

import numpy as np

import mne
from mne.datasets import testing

from groupmne import prepare_fwds, compute_group_inverse

import pytest


data_path = testing.data_path()
subjects_dir = op.join(data_path, 'subjects')
cov_fname = op.join(data_path, 'MEG', 'sample',
                    'sample_audvis_trunc-cov.fif')
ave_fname = op.join(data_path, 'MEG', 'sample',
                    'sample_audvis_trunc-ave.fif')

os.environ['SUBJECTS_DIR'] = subjects_dir


@pytest.mark.filterwarnings("ignore:Objective did not converge")
@pytest.mark.parametrize("method", ["lasso", "relasso", "grouplasso", "dirty",
                                    "mtw", "remtw"])
def test_inverse(src_fwds, method):
    src_ref, fwds = src_fwds
    cov = mne.read_cov(cov_fname, verbose=False)
    noise_covs = [cov, cov]
    ev = mne.read_evokeds(ave_fname, verbose=False)[0]
    ev = ev.crop(tmin=0.1, tmax=0.12)
    evokeds = [ev, ev]
    fwds = prepare_fwds(fwds, src_ref)
    epsilon = 1.
    gamma = 0.01
    beta = 0.1
    alpha_ot = 0.01
    alpha = 0.2

    n_sources = fwds[0]["sol_group"]["data"].shape[1]
    n_times = ev.times.size
    ot_dict = dict(alpha=alpha_ot, beta=beta, epsilon=epsilon,
                   gamma=gamma, tol=100.)
    lasso_dict = dict(alpha=alpha, tol=100.)
    solver_params = dict(lasso=lasso_dict,
                         relasso=lasso_dict,
                         grouplasso=lasso_dict,
                         dirty=dict(alpha=alpha, beta=beta, tol=100.),
                         mtw=ot_dict, remtw=ot_dict)

    stcs = compute_group_inverse(fwds, evokeds, noise_covs, method=method,
                                 spatiotemporal=False,
                                 **solver_params[method])
    assert len(stcs) == len(fwds)
    for stc in stcs:
        assert (stc.data.shape == (n_sources, n_times))


def test_spatiotemporal_grouplasso(src_fwds):
    src_ref, fwds = src_fwds
    cov = mne.read_cov(cov_fname, verbose=False)
    noise_covs = [cov, cov]
    ev = mne.read_evokeds(ave_fname, verbose=False)[0]
    ev = ev.crop(tmin=0.1, tmax=0.12)
    evokeds = [ev, ev]
    fwds = prepare_fwds(fwds, src_ref)
    alpha = 0.2

    lasso_dict = dict(alpha=alpha)

    stcs = compute_group_inverse(fwds, evokeds, noise_covs,
                                 method='grouplasso',
                                 spatiotemporal=True,
                                 **lasso_dict)

    # check group l21 norm works as expected
    for stc in stcs:
        data = stc.data
        positive_any = (data != 0).any(1)
        positive_all = (data != 0).all(1)
        assert (positive_any == positive_all).all()


@pytest.mark.parametrize("method", ["lasso", "relasso", "grouplasso", "dirty"])
def test_hyperparam_max(src_fwds, method):
    src_ref, fwds = src_fwds
    cov = mne.read_cov(cov_fname, verbose=False)
    noise_covs = [cov, cov]
    ev = mne.read_evokeds(ave_fname, verbose=False)[0]
    ev = ev.crop(tmin=0.1, tmax=0.12)
    evokeds = [ev, ev]
    fwds = prepare_fwds(fwds, src_ref)

    alpha = 1.1
    d1 = dict(alpha=alpha)
    d2 = dict(alpha=alpha, beta=alpha)
    params = dict(lasso=d1, grouplasso=d1, relasso=d1, dirty=d2)
    stcs = compute_group_inverse(fwds, evokeds, noise_covs,
                                 method=method,
                                 spatiotemporal=False,
                                 **params[method])
    for stc in stcs:
        assert abs(stc.data).max() == 0.


def test_implemented_models(src_fwds):
    src_ref, fwds = src_fwds
    cov = mne.read_cov(cov_fname, verbose=False)
    noise_covs = [cov, cov]
    ev = mne.read_evokeds(ave_fname, verbose=False)[0]
    ev = ev.crop(tmin=0.1, tmax=0.12)
    evokeds = [ev, ev]
    fwds = prepare_fwds(fwds, src_ref)

    alpha = 1.
    lasso_dict = dict(alpha=alpha)

    IMPLEMENTED_METHODS = ["lasso", "grouplasso", "dirty", "mtw", "remtw",
                           "relasso"]

    for method in IMPLEMENTED_METHODS:
        if method != "grouplasso":
            with pytest.raises(ValueError,
                               match="not feasible as a time dependent"
                                     " method"):
                compute_group_inverse(fwds, evokeds, noise_covs, method=method,
                                      spatiotemporal=True, **lasso_dict)

    method = "foo"
    for time_independent in [True, False]:
        with pytest.raises(ValueError, match="not a valid method"):
            compute_group_inverse(fwds, evokeds, noise_covs, method=method,
                                  spatiotemporal=False, **lasso_dict)


@pytest.mark.parametrize("method", ["mtw", "remtw"])
def test_ot_groundmetric(src_fwds, method):
    src_ref, fwds = src_fwds
    cov = mne.read_cov(cov_fname, verbose=False)
    noise_covs = [cov, cov]
    ev = mne.read_evokeds(ave_fname, verbose=False)[0]
    ev = ev.crop(tmin=0.1, tmax=0.12)
    evokeds = [ev, ev]
    fwds = prepare_fwds(fwds, src_ref)

    # Test ground metric with a wrong shape
    M_wrong_shape = np.ones((2, 3))
    ot_dict = dict(M=M_wrong_shape, tol=100.)
    with pytest.raises(ValueError, match="M must be an array"):
        compute_group_inverse(fwds, evokeds, noise_covs, method=method,
                              spatiotemporal=False, **ot_dict)

    # Test negative ground metric
    n_features = fwds[0]["sol_group"]["data"].shape[1]
    M_negative = - np.ones((n_features, n_features))
    ot_dict = dict(M=M_negative, tol=100.)
    with pytest.raises(ValueError, match="M must be non-negative"):
        compute_group_inverse(fwds, evokeds, noise_covs, method=method,
                              spatiotemporal=False, **ot_dict)


@pytest.mark.parametrize("method", ["lasso", "relasso", "grouplasso", "dirty",
                                    "mtw", "remtw"])
def test_evokeds(src_fwds, method):
    src_ref, fwds = src_fwds
    cov = mne.read_cov(cov_fname, verbose=False)
    noise_covs = [cov, cov]
    ev = mne.read_evokeds(ave_fname, verbose=False)[0]
    ev0 = ev.copy().crop(tmin=0.1, tmax=0.13)
    ev1 = ev.copy().crop(tmin=0.1, tmax=0.15)
    ev3 = ev.copy().crop(tmin=0.12, tmax=0.15)
    fwds = prepare_fwds(fwds, src_ref)

    evokeds = [ev0, ev0, ev0]
    with pytest.raises(ValueError, match="The number of evokeds is not equal"):
        compute_group_inverse(fwds, evokeds, noise_covs, method=method)
    evokeds = [ev0, ev1]
    with pytest.raises(ValueError,
                       match="times array with a different length"):
        compute_group_inverse(fwds, evokeds, noise_covs, method=method)

    evokeds = [ev0, ev3]
    with pytest.raises(ValueError, match="different time coordinates"):
        compute_group_inverse(fwds, evokeds, noise_covs, method=method)
