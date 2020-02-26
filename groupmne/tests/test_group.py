import os.path as op
import os

import numpy as np

import mne
from mne.datasets import testing

from groupmne import prepare_fwds

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
def test_different_gains(src_fwds):
    src_ref, fwds = src_fwds
    fwds = prepare_fwds(fwds, src_ref)
    gains = np.stack([fwd["sol_group"]["data"] for fwd in fwds])
    group_info = fwds[0]["sol_group"]["group_info"]
    n_ch = fwds[0]["nchan"]
    n_s = sum(group_info["n_sources"])
    assert gains.shape == (2, n_ch, n_s)
    with pytest.raises(AssertionError):
        gains /= abs(gains).max()
        np.testing.assert_allclose(gains[0], gains[1])


@testing.requires_testing_data
def test_filtering_fsaverage(src_fwds):
    src_ref, (fwd0, fwd1) = src_fwds
    fwds = [fwd1, fwd1]
    fwds_prep = prepare_fwds(fwds, src_ref)
    gains = np.stack([fwd["sol_group"]["data"] for fwd in fwds_prep])

    for fwd, gain in zip(fwds, gains):
        fwd = mne.convert_forward_solution(fwd, surf_ori=True,
                                           force_fixed=True,
                                           use_cps=True)
        fwd_gain = fwd["sol"]["data"]
        np.testing.assert_allclose(gain, fwd_gain)
