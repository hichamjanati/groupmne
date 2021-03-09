import os
import os.path as op

import numpy as np

# from numpy.testing import assert_array_equal

import mne
# from mne.inverse_sparse import mixed_norm
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
@pytest.mark.parametrize("method", ["lasso", "relasso", "multitasklasso",
                                    "dirty", "mtw", "remtw"])
def test_inverse(fsaverage_ref_data, method):
    srcs1, fwds = fsaverage_ref_data
    src_ref = srcs1[1]
    cov = mne.read_cov(cov_fname, verbose=False)
    noise_covs = [cov, cov]
    ev = mne.read_evokeds(ave_fname, verbose=False)[0]
    ev = ev.crop(tmin=0.1, tmax=0.12)
    evokeds = [ev, ev]
    fwds = prepare_fwds(fwds, src_ref)
    beta = 0.1
    alpha_ot = 0.01
    alpha = 0.2

    n_sources = fwds[0]["sol_group"]["data"].shape[1]
    n_times = ev.times.size
    ot_dict = dict(alpha=alpha_ot, beta=beta, tol=100.)
    lasso_dict = dict(alpha=alpha, tol=100.)
    solver_params = dict(lasso=lasso_dict,
                         relasso=lasso_dict,
                         multitasklasso=lasso_dict,
                         dirty=dict(alpha=alpha, beta=beta, tol=100.),
                         mtw=ot_dict, remtw=ot_dict)

    stcs = compute_group_inverse(fwds, evokeds, noise_covs, method=method,
                                 spatiotemporal=False,
                                 **solver_params[method])
    assert len(stcs) == len(fwds)
    for stc in stcs:
        assert (stc.data.shape == (n_sources, n_times))


def test_spatiotemporal_multitasklasso(fsaverage_ref_data, sample_ref_data):
    srcs1, fwds1 = fsaverage_ref_data
    srcs2, fwds2 = sample_ref_data

    for src_ref, fwds in zip([srcs1[1], srcs2[0]], [fwds1, fwds2]):
        cov = mne.read_cov(cov_fname, verbose=False)
        noise_covs = [cov, cov]
        ev = mne.read_evokeds(ave_fname, verbose=False)[0]
        ev = ev.crop(tmin=0.1, tmax=0.12)
        evokeds = [ev, ev]
        fwds = prepare_fwds(fwds, src_ref)
        alpha = 0.2

        lasso_dict = dict(alpha=alpha)

        stcs = compute_group_inverse(fwds, evokeds, noise_covs,
                                     method='multitasklasso',
                                     spatiotemporal=True,
                                     **lasso_dict)

        # check group l21 norm works as expected
        for stc in stcs:
            data = stc.data
            positive_any = (data != 0).any(1)
            positive_all = (data != 0).all(1)
            assert (positive_any == positive_all).all()


@pytest.mark.parametrize("method", ["lasso", "relasso", "multitasklasso",
                                    "dirty"])
def test_hyperparam_max(fsaverage_ref_data, method):
    srcs1, fwds = fsaverage_ref_data
    src_ref = srcs1[1]
    cov = mne.read_cov(cov_fname, verbose=False)
    noise_covs = [cov, cov]
    ev = mne.read_evokeds(ave_fname, verbose=False)[0]
    ev = ev.crop(tmin=0.1, tmax=0.12)
    evokeds = [ev, ev]
    fwds = prepare_fwds(fwds, src_ref)

    alpha = 1.1
    d1 = dict(alpha=alpha)
    d2 = dict(alpha=alpha, beta=alpha)
    params = dict(lasso=d1, multitasklasso=d1, relasso=d1, dirty=d2)
    stcs = compute_group_inverse(fwds, evokeds, noise_covs,
                                 method=method,
                                 spatiotemporal=False,
                                 **params[method])
    for stc in stcs:
        assert abs(stc.data).max() == 0.


def test_implemented_models(fsaverage_ref_data):
    srcs1, fwds = fsaverage_ref_data
    src_ref = srcs1[1]
    cov = mne.read_cov(cov_fname, verbose=False)
    noise_covs = [cov, cov]
    ev = mne.read_evokeds(ave_fname, verbose=False)[0]
    ev = ev.crop(tmin=0.1, tmax=0.12)
    evokeds = [ev, ev]
    fwds = prepare_fwds(fwds, src_ref)

    alpha = 1.
    lasso_dict = dict(alpha=alpha)

    IMPLEMENTED_METHODS = ["lasso", "multitasklasso", "dirty", "mtw", "remtw",
                           "relasso"]

    for method in IMPLEMENTED_METHODS:
        if method != "multitasklasso":
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
def test_ot_groundmetric(fsaverage_ref_data, sample_ref_data, method):
    srcs1, fwds1 = fsaverage_ref_data
    srcs2, fwds2 = sample_ref_data

    for src_ref, fwds in zip([srcs1[1], srcs2[0]], [fwds1, fwds2]):
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


@pytest.mark.parametrize("method", ["lasso", "relasso", "multitasklasso",
                                    "dirty", "mtw", "remtw"])
def test_evokeds(fsaverage_ref_data, method):
    srcs1, fwds = fsaverage_ref_data
    src_ref = srcs1[1]
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


# def test_vs_mixed_norm(src_fwds):
#     src_ref, fwds = src_fwds
#     cov = mne.read_cov(cov_fname, verbose=False)
#     noise_covs = [cov]
#     fwds_prepared = prepare_fwds(fwds[:1], src_ref)
#     ev = mne.read_evokeds(ave_fname, verbose=False)[0]
#     ev.crop(tmin=0.1, tmax=0.12)
#
#     stc_gl = compute_group_inverse(fwds_prepared, [ev], noise_covs,
#                                    method="multitasklasso",
#                                    spatiotemporal=True,
#                                    alpha=0.5)[0]
#     stc_mx = mixed_norm(ev, fwds_prepared[0], cov, alpha=50, loose=0.,
#                         debias=False)
#     stc_mx.data *= 1e9
#     stc_mx.subject = "fsaverage"
#
#     support_gl = np.where(stc_gl.data.any(1))[0]
#     support_mx = np.where(stc_mx.data.any(1))[0]
#     ampl_gl = stc_gl.data[support_gl]
#     ampl_mx = stc_mx.data[support_mx]
#     # vertices_gl = stc_gl.vertices[0][support_gl]
#     # vertices_mx = stc_mx.vertices[0][support_mx]
#
#     # assert_array_equal(vertices_gl, vertices_mx)
#     assert abs(ampl_gl - ampl_mx).max() < 1e-4
