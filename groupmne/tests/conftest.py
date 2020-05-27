"""Fixtures for group testing."""
import pytest
import os
import os.path as op

import mne
from mne.datasets import testing

from groupmne import compute_fwd


@pytest.fixture(scope='session', autouse=True)
def fsaverage_ref_data(request):
    """Compute source space and forward operators with fsaverage as ref."""
    data_path = testing.data_path()
    subjects_dir = op.join(data_path, 'subjects')

    bem_fname = op.join(data_path, 'subjects', 'sample', 'bem',
                        'sample-1280-1280-1280-bem-sol.fif')

    trans_fname = op.join(data_path, 'MEG', 'sample',
                          'sample_audvis_trunc-trans.fif')

    raw_fname = op.join(data_path, 'MEG', 'sample',
                        'sample_audvis_trunc_raw.fif')
    resolution = 3
    spacing = "ico%d" % resolution
    os.environ['SUBJECTS_DIR'] = subjects_dir

    src_fs = mne.setup_source_space(subject="fsaverage",
                                    spacing=spacing,
                                    subjects_dir=subjects_dir,
                                    add_dist=False)
    src_sample = mne.morph_source_spaces(src_fs, subject_to="sample",
                                         subjects_dir=subjects_dir)
    fwd_sample = compute_fwd("sample", src_sample, raw_fname, trans_fname,
                             bem_fname, mindist=5)
    fwd_fs = compute_fwd("fsaverage", src_fs, raw_fname, trans_fname,
                         bem_fname, mindist=10)
    fwd2 = compute_fwd("fsaverage", src_fs, raw_fname, trans_fname, bem_fname,
                       mindist=0, meg=False)
    fwd3 = compute_fwd("fsaverage", src_fs, raw_fname, trans_fname, bem_fname,
                       mindist=0, eeg=False)

    assert len(fwd2["sol"]["data"]) < len(fwd_sample["sol"]["data"])
    assert len(fwd3["sol"]["data"]) < len(fwd_sample["sol"]["data"])

    src_fwds = [src_sample, src_fs], [fwd_sample, fwd_fs]
    return src_fwds


@pytest.fixture(scope='session', autouse=True)
def sample_ref_data(request):
    """Compute source space and forward operators with sample as ref."""
    data_path = testing.data_path()
    subjects_dir = op.join(data_path, 'subjects')

    bem_fname = op.join(data_path, 'subjects', 'sample', 'bem',
                        'sample-1280-1280-1280-bem-sol.fif')

    trans_fname = op.join(data_path, 'MEG', 'sample',
                          'sample_audvis_trunc-trans.fif')

    raw_fname = op.join(data_path, 'MEG', 'sample',
                        'sample_audvis_trunc_raw.fif')
    resolution = 3
    spacing = "ico%d" % resolution
    os.environ['SUBJECTS_DIR'] = subjects_dir

    src_sample = mne.setup_source_space(subject="sample",
                                        spacing=spacing,
                                        subjects_dir=subjects_dir,
                                        add_dist=False)
    src_fs = mne.morph_source_spaces(src_sample, subject_to="fsaverage",
                                     subjects_dir=subjects_dir)
    fwd_sample = compute_fwd("sample", src_sample, raw_fname, trans_fname,
                             bem_fname, mindist=5)
    fwd_fs = compute_fwd("fsaverage", src_fs, raw_fname, trans_fname,
                         bem_fname, mindist=10)

    src_fwds = [src_sample, src_fs], [fwd_sample, fwd_fs]
    return src_fwds
