import os.path as op
import glob


def get_trigger_info(camcan_meg_path):
    """Read trigger info."""
    pass


def get_subjects(camcan_meg_raw_path):
    """Get MEG subjects."""
    out = [cc.split('/')[-1] for cc in sorted(glob.glob(
        op.join(camcan_meg_raw_path, 'CC??????')))]
    return out
