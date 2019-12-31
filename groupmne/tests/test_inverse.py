import os
import os.path as op

import numpy as np

import mne
from mne.datasets import testing

from groupmne.inverse import InverseOperator

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

    epsilon = 1.
    gamma = 0.01
    beta = 0.9
    alpha_ot = 0.01
    alpha = 0.9

    ot_dict = dict(alpha=alpha_ot, beta=beta, epsilon=epsilon,
                   gamma=gamma, tol=100)
    lasso_dict = dict(alpha=alpha, tol=100)
    solver_params = dict(lasso=lasso_dict,
                         relasso=lasso_dict,
                         grouplasso=lasso_dict,
                         dirty=dict(alpha=alpha, beta=beta, tol=100),
                         mtw=ot_dict, remtw=ot_dict)

    inv_op = InverseOperator(fwds, noise_covs, src_ref, ch_type="grad",
                             depth=0.9)
    inv_op.compute_group_model()
    inv_op.solve(evokeds, method=method,
                 time_independent=True,
                 verbose=False, **solver_params[method])
    stc_data = inv_op._stc_data
    inv_op.solve(evokeds, method=method,
                 time_independent=True,
                 verbose=False, **solver_params[method])
    np.testing.assert_array_equal(inv_op._stc_data, stc_data)


def test_time_gl(src_fwds):
    src_ref, fwds = src_fwds
    cov = mne.read_cov(cov_fname, verbose=False)
    noise_covs = [cov, cov]
    ev = mne.read_evokeds(ave_fname, verbose=False)[0]
    ev = ev.crop(tmin=0.1, tmax=0.12)
    evokeds = [ev, ev]

    alpha = 0.2

    lasso_dict = dict(alpha=alpha)

    inv_op = InverseOperator(fwds, noise_covs, src_ref, ch_type="grad",
                             depth=0.9)
    inv_op.compute_group_model()
    inv_op.solve(evokeds, method="grouplasso",
                 time_independent=False,
                 verbose=False, **lasso_dict)


@pytest.mark.parametrize("method", ["lasso", "relasso", "grouplasso", "dirty"])
def test_hyperparam_max(src_fwds, method):
    src_ref, fwds = src_fwds
    cov = mne.read_cov(cov_fname, verbose=False)
    noise_covs = [cov, cov]
    ev = mne.read_evokeds(ave_fname, verbose=False)[0]
    ev = ev.crop(tmin=0.1, tmax=0.12)
    evokeds = [ev, ev]

    alpha = 1.1
    lasso_dict = dict(alpha=alpha)

    inv_op = InverseOperator(fwds, noise_covs, src_ref, ch_type="grad",
                             depth=0.9)
    inv_op.compute_group_model()
    inv_op.solve(evokeds, method="grouplasso",
                 time_independent=False,
                 verbose=False, **lasso_dict)
    assert abs(inv_op._stc_data).max() == 0.


def test_implemented_models(src_fwds):
    src_ref, fwds = src_fwds
    cov = mne.read_cov(cov_fname, verbose=False)
    noise_covs = [cov, cov]
    ev = mne.read_evokeds(ave_fname, verbose=False)[0]
    ev = ev.crop(tmin=0.1, tmax=0.12)
    evokeds = [ev, ev]

    alpha = 1.
    lasso_dict = dict(alpha=alpha)

    inv_op = InverseOperator(fwds, noise_covs, src_ref, ch_type="grad",
                             depth=0.9)
    inv_op.compute_group_model()

    IMPLEMENTED_METHODS = ["lasso", "grouplasso", "dirty", "mtw", "remtw",
                           "relasso"]

    for method in IMPLEMENTED_METHODS:
        if method != "grouplasso":
            with pytest.raises(ValueError,
                               match="not feasible as a time dependent"
                                     " method"):
                inv_op.solve(evokeds, method=method,
                             time_independent=False,
                             verbose=False, **lasso_dict)

    method = "foo"
    for time_independent in [True, False]:
        with pytest.raises(ValueError, match="not a valid method"):
            inv_op.solve(evokeds, method=method, time_independent=False,
                         verbose=False, **lasso_dict)


@pytest.mark.parametrize("method", ["mtw", "remtw"])
def test_ot_groundmetric(src_fwds, method):
    src_ref, fwds = src_fwds
    cov = mne.read_cov(cov_fname, verbose=False)
    noise_covs = [cov, cov]
    ev = mne.read_evokeds(ave_fname, verbose=False)[0]
    ev = ev.crop(tmin=0.1, tmax=0.12)
    evokeds = [ev, ev]

    inv_op = InverseOperator(fwds, noise_covs, src_ref, ch_type="grad",
                             depth=0.9)
    inv_op.compute_group_model()

    # Test ground metric with a wrong shape
    M_wrong_shape = np.ones((2, 3))
    ot_dict = dict(M=M_wrong_shape, tol=100)
    with pytest.raises(ValueError, match="M must be an array"):
        inv_op.solve(evokeds, method=method, time_independent=True,
                     verbose=False, **ot_dict)

    # Test negative ground metric
    inv_op.compute_group_model()
    n_features = inv_op._gains.shape[-1]
    M_negative = - np.ones((n_features, n_features))
    ot_dict = dict(M=M_negative, tol=100)
    with pytest.raises(ValueError, match="M must be non-negative"):
        inv_op.solve(evokeds, method=method, time_independent=True,
                     verbose=False, **ot_dict)


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

    inv_op = InverseOperator(fwds, noise_covs, src_ref, ch_type="grad",
                             depth=0.9)
    inv_op.compute_group_model()

    evokeds = [ev0, ev0, ev0]
    with pytest.raises(ValueError, match="The number of evokeds is not equal"):
        inv_op.solve(evokeds, method=method, verbose=False)

    evokeds = [ev0, ev1]
    with pytest.raises(ValueError, match="different numbers of time points"):
        inv_op.solve(evokeds, method=method, verbose=False)

    evokeds = [ev0, ev3]
    with pytest.raises(ValueError, match="have different time coordinates."):
        inv_op.solve(evokeds, method=method, verbose=False)


def test_resolve(src_fwds):
    src_ref, fwds = src_fwds
    cov = mne.read_cov(cov_fname, verbose=False)
    noise_covs = [cov, cov]
    ev = mne.read_evokeds(ave_fname, verbose=False)[0]
    ev1 = ev.copy().crop(tmin=0.1, tmax=0.12)
    ev2 = ev.copy().crop(tmin=0.13, tmax=0.15)

    evokeds = [ev1, ev1]
    method = "lasso"
    inv_op = InverseOperator(fwds, noise_covs, src_ref, ch_type="grad",
                             depth=0.9)
    inv_op.compute_group_model()

    inv_op.solve(evokeds, method=method, time_independent=True,
                 verbose=False)
    _meeg_data_processed = inv_op._meeg_data_processed.copy()

    evokeds2 = [ev2, ev2]
    inv_op.solve(evokeds2, method=method, time_independent=True,
                 verbose=False)
    # test that the data whitening changed when changing evokeds
    np.testing.assert_raises(AssertionError, np.testing.assert_array_equal,
                             _meeg_data_processed, inv_op._meeg_data_processed)

    # test that a second call to solve is reliable
    inv_op_2 = InverseOperator(fwds, noise_covs, src_ref, ch_type="grad",
                               depth=0.9)
    inv_op_2.compute_group_model()
    inv_op_2.solve(evokeds2, method=method, time_independent=True,
                   verbose=False)
    np.testing.assert_array_equal(inv_op_2._stc_data, inv_op._stc_data)
