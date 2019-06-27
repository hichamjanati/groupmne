from groupmne.utils import compute_coreg_dist
from mne.datasets import testing
import os.path as op
import os


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
    d = compute_coreg_dist(subject, trans_fname, raw_fname, subjects_dir)
    assert d < 5e-2
