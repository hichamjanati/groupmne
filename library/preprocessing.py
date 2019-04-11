import os.path as op
import mne

_curr_dir = op.dirname(op.realpath(__file__))


def run_maxfilter(raw, coord_frame='head'):
    """Run maxfilter."""

    cal = op.join(_curr_dir, 'sss_params', 'sss_cal.dat')
    ctc = op.join(_curr_dir, 'sss_params', 'ct_sparse.fif')

    raw = mne.preprocessing.maxwell_filter(
        raw, calibration=cal,
        cross_talk=ctc,
        st_duration=10.,
        st_correlation=.98,
        coord_frame=coord_frame)
    return raw


def parse_bad_channels(sss_log):
    """Parse bad channels from sss_log."""
    with open(sss_log) as fid:
        bad_lines = {l for l in fid.readlines() if 'Static bad' in l}
    bad_channels = list()
    for line in bad_lines:
        chans = line.split(':')[1].strip(' \n').split(' ')
        for cc in chans:
            ch_name = 'MEG%01d' % int(cc)
            if ch_name not in bad_channels:
                bad_channels.append(ch_name)
    return bad_channels
