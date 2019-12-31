from groupmne import compute_gains, utils
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
@pytest.mark.parametrize("hemi", ["lh", "rh", "both"])
def test_different_gains(src_fwds, hemi):
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
    with pytest.raises(AssertionError):
        gains /= abs(gains).max()
        np.testing.assert_allclose(gains[0], gains[1])


@testing.requires_testing_data
@pytest.mark.parametrize("hemi", ["lh", "rh", "both"])
def test_filtering_fsaverage(src_fwds, hemi):
    src_ref, (fwd0, fwd1) = src_fwds
    fwds = [fwd1, fwd1]
    gains, group_info = compute_gains(fwds, src_ref, ch_type="grad",
                                      hemi=hemi)

    n_lh = fwds[0]["src"][0]["nuse"]
    if hemi == "lh":
        col0 = 0
        col1 = n_lh
    elif hemi == "rh":
        col0 = n_lh
        col1 = None
    elif hemi == "both":
        col0 = 0
        col1 = None
    for fwd, gain in zip(fwds, gains):
        fwd = mne.convert_forward_solution(fwd, surf_ori=True,
                                           force_fixed=True,
                                           use_cps=True)
        ch_names = utils._get_channels(fwd)
        sel = utils._filter_channels(fwd["info"], ch_names, "grad")
        fwd_gain = fwd["sol"]["data"][sel]
        fwd_gain = fwd_gain[:, col0:col1]
        np.testing.assert_allclose(gain, fwd_gain)
