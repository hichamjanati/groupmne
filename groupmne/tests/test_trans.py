import os.path as op
import os

from mne.datasets import testing

from groupmne.utils import _compute_coreg_dist


data_path = testing.data_path(download=True)
subjects_dir = op.join(data_path, 'subjects')
subject = "sample"
trans_fname = op.join(data_path, 'MEG', 'sample',
                      'sample_audvis_trunc-trans.fif')
raw_fname = op.join(data_path, 'MEG', 'sample',
                    'sample_audvis_trunc_raw.fif')


os.environ['SUBJECTS_DIR'] = subjects_dir


@testing.requires_testing_data
def test_trans():
    d = _compute_coreg_dist(subject, trans_fname, raw_fname, subjects_dir)
    assert d < 5e-2
