"""Fixtures for group testing."""
import pytest
import os
import os.path as op

import mne
from mne.datasets import testing

from groupmne import compute_fwd


@pytest.fixture(scope='session', autouse=True)
def src_fwds(request):
    """Compute reference source space and forward operators."""
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

    src_ref = mne.setup_source_space(subject="fsaverage",
                                     spacing=spacing,
                                     subjects_dir=subjects_dir,
                                     add_dist=False)
    fwd0 = compute_fwd("sample", src_ref, raw_fname, trans_fname, bem_fname,
                       mindist=5)
    fwd1 = compute_fwd("fsaverage", src_ref, raw_fname, trans_fname, bem_fname,
                       mindist=10)
    fwd2 = compute_fwd("fsaverage", src_ref, raw_fname, trans_fname, bem_fname,
                       mindist=0, meg=False)
    fwd3 = compute_fwd("fsaverage", src_ref, raw_fname, trans_fname, bem_fname,
                       mindist=0, eeg=False)

    assert len(fwd2["sol"]["data"]) < len(fwd0["sol"]["data"])
    assert len(fwd3["sol"]["data"]) < len(fwd0["sol"]["data"])

    src_fwds = src_ref, [fwd0, fwd1]
    return src_fwds
