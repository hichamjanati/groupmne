"""
Multi-subject MEG processing
============================

The aim of this tutorial is to show how to process MEG with the purpose of
computing source estimates jointly for a group of subjects (see Group Lasso
example).
"""

# Author: Hicham Janati (hicham.janati@inria.fr)
#
# License: BSD (3-clause)

import os
import os.path as op
from mne import compute_covariance, write_cov, find_events, Epochs, pick_types
from mne.io import read_raw_fif
from mne.datasets import hf_sef
from matplotlib import pyplot as plt


##########################################################
# Download and process MEG data
# -----------------------------
#
# We download the raw data to estimate the noise covariance of each subject


data_path = hf_sef.data_path("raw")
meg_path = data_path + "/MEG/"

data_path = op.expanduser(data_path)
subjects_dir = data_path + "/subjects/"
os.environ['SUBJECTS_DIR'] = subjects_dir

raw_name_s = [meg_path + s for s in ["subject_a/sef_right_raw.fif",
              "subject_b/hf_sef_15min_raw.fif"]]


def get_epoch_cov(raw_name):
    raw = read_raw_fif(raw_name)
    events = find_events(raw)[:500]
    event_id = dict(hf=1)  # event trigger and conditions
    tmin = -0.05  # start of each epoch (50ms before the trigger)
    tmax = 0.3  # end of each epoch (300ms after the trigger)
    baseline = (None, 0)  # means from the first instant to t = 0
    epochs = Epochs(raw, events, event_id, tmin, tmax, proj=True,
                    baseline=baseline)
    del events, raw
    cov = compute_covariance(epochs, tmin=None, tmax=0.)
    return epochs, cov


evoked_s = []

# compute noise covariance (takes a few minutes)
noise_cov_s = []
for subj, raw_name in zip(["a", "b"], raw_name_s):
    print("computing noise covariance for subject %s ..." % subj)
    ep, cov = get_epoch_cov(raw_name)
    noise_cov_s.append(cov)
    cov_fname = meg_path + "subject_%s/sef-cov.fif" % subj
    write_cov(cov_fname, cov)
    evoked_s.append(ep.average())

f, axes = plt.subplots(1, 2, sharey=True)
for ax, ev, ll in zip(axes.ravel(), evoked_s, ["a", "b"]):
    picks = pick_types(ev.info, meg="grad")
    ev.plot(picks=picks, axes=ax, show=False)
    ax.set_title("Subject %s" % ll, fontsize=15)
plt.show()
