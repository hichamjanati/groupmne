from groupmne import compute_gains, compute_inv_data
import mne
from mne.datasets import testing
import numpy as np
import os.path as op
import os
import pytest


data_path = testing.data_path()
subjects_dir = op.join(data_path, 'subjects')

bem_fname = op.join(data_path, 'subjects', 'sample', 'bem',
                    'sample-1280-1280-1280-bem-sol.fif')
fname_fs = op.join(subjects_dir, 'fsaverage', 'bem', 'fsaverage-ico-5-src.fif')
trans_fname = op.join(data_path, 'MEG', 'sample',
                      'sample_audvis_trunc-trans.fif')

raw_fname = op.join(data_path, 'MEG', 'sample',
                    'sample_audvis_trunc_raw.fif')
ave_fname = op.join(data_path, 'MEG', 'sample',
                    'sample_audvis_trunc-ave.fif')
cov_fname = op.join(data_path, 'MEG', 'sample',
                    'sample_audvis_trunc-cov.fif')
rng = np.random.RandomState(0)
resolution = 3
spacing = "ico%d" % resolution
os.environ['SUBJECTS_DIR'] = subjects_dir


@testing.requires_testing_data
@pytest.mark.parametrize("hemi", ["lh", "rh", "both"])
def test_gains(src_fwds, hemi):
    src_ref, fwds = src_fwds
    gains, group_info = compute_gains(fwds, src_ref, ch_type="grad",
                                      hemi=hemi)
    n_ch = len(group_info["sel"])
    if hemi == "both":
        n_s = sum(group_info["n_sources"])
    else:
        i = int(hemi == "rh")
        n_s = group_info["n_sources"][i]
    assert gains.shape == (2, n_ch, n_s)


@testing.requires_testing_data
def test_inverse_data(src_fwds):
    src_ref, fwds = src_fwds
    ev = mne.read_evokeds(ave_fname)[0]
    cov = mne.read_cov(cov_fname)
    evoked_s = [ev, ev]
    covs = [cov, cov]
    gains, M, group_info = \
        compute_inv_data(fwds, src_ref, evoked_s, covs,
                         ch_type="grad", tmin=0.02, tmax=0.04)
    n_ch = len(group_info["sel"])
    n_s = sum(group_info["n_sources"])
    n_t = ev.crop(0.02, 0.04).times.shape[0]
    assert gains.shape == (2, n_ch, n_s)
    assert M.shape == (2, n_ch, n_t)


@testing.requires_testing_data
@pytest.mark.parametrize("hemi", ["lh", "rh", "both"])
def test_different_gains(src_fwds, hemi):
    src_ref, fwds = src_fwds
    # are different
    gains, group_info = compute_gains(fwds, src_ref, ch_type="grad",
                                      hemi=hemi)
    n_ch = len(group_info["sel"])
    if hemi == "both":
        n_s = sum(group_info["n_sources"])
    else:
        i = int(hemi == "rh")
        n_s = group_info["n_sources"][i]
    assert gains.shape == (2, n_ch, n_s)
    with pytest.raises(AssertionError):
        gains /= abs(gains).max()
        np.testing.assert_allclose(gains[0], gains[1])
