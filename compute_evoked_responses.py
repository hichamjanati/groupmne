import os.path as op
from collections import Counter

import os
import numpy as np
import pandas as pd
import mne

from joblib import Parallel, delayed
from autoreject import get_rejection_threshold

from config import get_subjects_list
import library as lib

camcan_path = '/storage/store/data/camcan'
camcan_meg_path = op.join(
    camcan_path, 'camcan47/cc700/meg/pipeline/release004/')
camcan_meg_raw_path = op.join(camcan_meg_path, 'data/aamod_meg_get_fif_00001')

mne_camcan_freesurfer_path = (
    '/storage/store/data/camcan-mne/freesurfer')

derivative_path = '/home/parietal/hjanati/data/camcan/meg/'


max_filter_info_path = op.join(
    camcan_meg_path,
    "data_nomovecomp/"
    "aamod_meg_maxfilt_00001")

kinds = ['passive', 'task']

task_info = {
    'passive': {
        'event_id': [{
            'Aud300Hz': 6, 'Aud600Hz': 7, 'Aud1200Hz': 8, 'Vis': 9}],
        'epochs_params': [{
            'tmin': -0.2, 'tmax': 0.7, 'baseline': (-.2, None),
            'decim': 8}],
        'lock': ['stim']
    },
    'task': {
        'event_id': [
            {'AudVis300Hz': 1, 'AudVis600Hz': 2, 'AudVis1200Hz': 3},
            {'resp': 8192}],
        'epochs_params': [
            {'tmin': -0.2, 'tmax': 0.7, 'baseline': (-.2, None),
             'decim': 8},
            {'tmin': -0.5, 'tmax': 1,
             'baseline': (.8, 1.0), 'decim': 8}],
        'lock': ['stim', 'resp'],
    }
}


def _parse_bads(subject, kind):
    sss_log = op.join(
        max_filter_info_path, subject,
        kind, "mf2pt2_{kind}_raw.log".format(kind=kind))

    try:
        bads = lib.preprocessing.parse_bad_channels(sss_log)
    except Exception as err:
        print(err)
        bads = []
    # first 100 channels ommit the 0.
    bads = [''.join(['MEG', '0', bb.split('MEG')[-1]])
            if len(bb) < 7 else bb for bb in bads]
    return bads


def _get_global_reject_ssp(raw):
    eog_epochs = mne.preprocessing.create_eog_epochs(raw)
    if len(eog_epochs) >= 5:
        reject_eog = get_rejection_threshold(eog_epochs, decim=8)
        del reject_eog['eog']
    else:
        reject_eog = None

    ecg_epochs = mne.preprocessing.create_ecg_epochs(raw)
    if len(ecg_epochs) >= 5:
        reject_ecg = get_rejection_threshold(ecg_epochs[:200], decim=8)
    else:
        reject_eog = None

    if reject_eog is None:
        reject_eog = reject_ecg
    if reject_ecg is None:
        reject_ecg = reject_eog
    return reject_eog, reject_ecg


def _run_maxfilter(raw, subject, kind):

    bads = _parse_bads(subject, kind)

    raw.info['bads'] = bads

    raw = lib.preprocessing.run_maxfilter(raw, coord_frame='head')
    return raw


def _compute_add_ssp_exg(raw):
    reject_eog, reject_ecg = _get_global_reject_ssp(raw)

    proj_eog = mne.preprocessing.compute_proj_eog(
        raw, average=True, reject=reject_eog, n_mag=1, n_grad=1, n_eeg=1)

    proj_ecg = mne.preprocessing.compute_proj_ecg(
        raw, average=True, reject=reject_ecg, n_mag=1, n_grad=1, n_eeg=1)

    raw.add_proj(proj_eog[0])
    raw.add_proj(proj_ecg[0])


def _get_global_reject_epochs(raw, events, event_id, epochs_params):
    epochs = mne.Epochs(
        raw, events, event_id=event_id, proj=False,
        **epochs_params)
    epochs.load_data()
    epochs.pick_types(meg=True)
    epochs.apply_proj()
    reject = get_rejection_threshold(epochs, decim=8)
    return reject


def _compute_evoked(subject, kind):

    fname = op.join(
        camcan_meg_raw_path,
        subject, kind, '%s_raw.fif' % kind)

    raw = mne.io.read_raw_fif(fname)
    mne.channels.fix_mag_coil_types(raw.info)
    raw = _run_maxfilter(raw, subject, kind)
    raw.filter(0.1, 45)
    _compute_add_ssp_exg(raw)

    out = {}
    for ii, event_id in enumerate(task_info[kind]['event_id']):
        epochs_params = task_info[kind]['epochs_params'][ii]
        lock = task_info[kind]['lock'][ii]
        events = mne.find_events(
            raw, uint_cast=True, min_duration=2. / raw.info['sfreq'])

        if kind == 'task' and lock == 'resp':
            event_map = np.array(
                [(k, v) for k, v in Counter(events[:, 2]).items()])
            button_press = event_map[:, 0][np.argmax(event_map[:, 1])]
            if event_map[:, 1][np.argmax(event_map[:, 1])] >= 50:
                events[events[:, 2] == button_press, 2] = 8192
            else:
                raise RuntimeError('Could not guess button press')

        reject = _get_global_reject_epochs(
            raw,
            events=events,
            event_id=event_id,
            epochs_params=epochs_params)

        epochs = mne.Epochs(
            raw, events=events, event_id=event_id, reject=reject,
            preload=True,
            **epochs_params)

        evokeds = list()
        for kk in event_id:
            evoked = epochs[kk].average()
            evoked.comment = kk
            evokeds.append(evoked)

        # tmax is 0.05 to account for the shift error of 50ms in camcan
        noise_covs = mne.compute_covariance(epochs, tmin=None, tmax=0.05,
                                            verbose=False, n_jobs=1,
                                            projs=None)

        out_path = op.join(derivative_path, subject)
        if not op.exists(out_path):
            os.makedirs(out_path)
        epo_fname = op.join(out_path,
                            '%s_%s_sensors-epo.fif' % (kind, lock))
        cov_fname = op.join(out_path,
                            '%s_%s_sensors-cov.fif' % (kind, lock))
        ave_fname = op.join(out_path,
                            '%s_%s_sensors-ave.fif' % (kind, lock))

        mne.write_evokeds(ave_fname, evokeds)
        mne.write_cov(cov_fname, noise_covs)
        # no need to save epochs
        epochs.save(epo_fname)

        out.update({lock: (kind, epochs.average().nave)})

    return out


def _run_all(subject, kind):
    mne.utils.set_log_level('warning')
    print(subject)
    error = 'None'
    result = dict()
    try:
        result = _compute_evoked(subject, kind)
    except Exception as err:
        error = repr(err)
        print(error)

    out = dict(subject=subject, kind=kind, error=error)
    out.update(result)
    return out


subjects = get_subjects_list("camcan", simu=True)
out_passive = Parallel(n_jobs=40)(
    delayed(_run_all)(subject=subject, kind='passive')
    for subject in subjects)

out_df_passive = pd.DataFrame(out_passive)
out_df_passive.to_csv(
    op.join(
        derivative_path,
        'log_compute_evoked_passive.csv'))
