"""
Multi-subject joint source localization with multi-task models.
===============================================================

The aim of this tutorial is to show how to leverage functional similarity
across subjects to improve source localization. For that purpose we use the
the high frequency SEF MEG dataset of (Nurminen et al., 2017) which provides
MEG and MRI data for two subjects.
"""

# Author: Hicham Janati (hicham.janati@inria.fr)
#
# License: BSD (3-clause)

import mne
import os
import os.path as op
from mne.parallel import parallel_func
from mne.datasets import hf_sef
from matplotlib import pyplot as plt

from groupmne import compute_group_inverse, prepare_fwds, compute_fwd

##########################################################
# Download and process MEG data
# -----------------------------
#
# For this example, we use the HF somatosensory dataset [2].
# We need the raw data to estimate the noise covariance
# since only average MEG data (and MRI) are provided in "evoked".
# The data will be downloaded in the same location


_ = hf_sef.data_path("raw")
data_path = hf_sef.data_path("evoked")
meg_path = data_path + "/MEG/"

data_path = op.expanduser(data_path)
subjects_dir = data_path + "/subjects/"
os.environ['SUBJECTS_DIR'] = subjects_dir

raw_name_s = [meg_path + s for s in ["subject_a/sef_right_raw.fif",
              "subject_b/hf_sef_15min_raw.fif"]]


def process_meg(raw_name):
    """Extract epochs from a raw fif file.

    Parameters
    ----------
    raw_name: str.
        path to the raw fif file.

    Returns
    -------
    epochs: Epochs instance

    """
    raw = mne.io.read_raw_fif(raw_name)
    events = mne.find_events(raw)

    event_id = dict(hf=1)  # event trigger and conditions
    tmin = -0.05  # start of each epoch (50ms before the trigger)
    tmax = 0.3  # end of each epoch (300ms after the trigger)
    baseline = (None, 0)  # means from the first instant to t = 0
    epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True,
                        baseline=baseline)
    return epochs


epochs_s = [process_meg(raw_name) for raw_name in raw_name_s]
evokeds = [ep.average() for ep in epochs_s]

# compute noise covariance (takes a few minutes)
noise_covs = []
for subj, ep in zip(["a", "b"], epochs_s):
    cov_fname = meg_path + f"subject_{subj}/sef-cov.fif"
    cov = mne.compute_covariance(ep[:100], tmin=None, tmax=0.)
    noise_covs.append(cov)


f, axes = plt.subplots(1, 2, sharey=True)
for ax, ev, nc, ll in zip(axes.ravel(), evokeds, noise_covs, ["a", "b"]):
    picks = mne.pick_types(ev.info, meg="grad")
    ev.plot(picks=picks, axes=ax, noise_cov=nc, show=False)
    ax.set_title("Subject %s" % ll, fontsize=15)
plt.show()

del epochs_s

#########################################################
# Source and forward modeling
# ---------------------------
# To guarantee an alignment across subjects, we start by
# computing the source space of `fsaverage`

resolution = 4
spacing = "ico%d" % resolution
src_ref = mne.setup_source_space(subject="fsaverage",
                                 spacing=spacing,
                                 subjects_dir=subjects_dir,
                                 add_dist=False)

######################################################
# Compute forward models with a reference source space
# ----------------------------------------------------
# the function `compute_fwd` morphs the source space src_ref to the
# surface of each subject by mapping the sulci and gyri patterns
# and computes their forward operators. Next we prepare the forward operators
# to be aligned across subjects

subjects = ["subject_a", "subject_b"]
trans_fname_s = [meg_path + "%s/sef-trans.fif" % s for s in subjects]
bem_fname_s = [subjects_dir + "%s/bem/%s-5120-bem-sol.fif" % (s, s)
               for s in subjects]
n_jobs = 1
parallel, run_func, _ = parallel_func(compute_fwd, n_jobs=n_jobs)


fwds_ = parallel(run_func(s, src_ref, info, trans, bem,  mindist=3)
                 for s, info, trans, bem in zip(subjects, raw_name_s,
                                                trans_fname_s, bem_fname_s))

fwds = prepare_fwds(fwds_, src_ref, copy=False)

##################################################
# Solve the inverse problems with Multi-task Lasso
# ------------------------------------------------

# The Multi-task Lasso assumes the source locations are the same across
# subjects for all instants i.e if a source is zero for one subject, it will
# be zero for all subjects. "alpha" is a hyperparameter that controls this
# structured sparsity prior. it must be set as a positive number between 0
# and 1. With alpha = 1, all the sources are 0.

# We restric the time points around 20ms in order to reconstruct the sources of
# the N20 response.
evokeds = [ev.crop(0.015, 0.025) for ev in evokeds]

stcs = compute_group_inverse(fwds, evokeds, noise_covs,
                             method='multitasklasso',
                             spatiotemporal=True,
                             alpha=0.8)

############################################
# Let's visualize the N20 response. The stimulus was applied on the right
# hand, thus we only show the left hemisphere. The activation is exactly in
# the primary somatosensory cortex. We highlight the borders of the post
# central gyrus.


t = 0.02
plot_kwargs = dict(
    hemi='lh', subjects_dir=subjects_dir, views="lateral",
    initial_time=t, time_unit='s', size=(800, 800),
    smoothing_steps=5, cortex=("gray", -1, 6, True))

t_idx = stcs[0].time_as_index(t)

for stc, subject in zip(stcs, subjects):
    g_post_central = mne.read_labels_from_annot(subject, "aparc.a2009s",
                                                subjects_dir=subjects_dir,
                                                regexp="G_postcentral-lh")[0]
    n_sources = [stc.vertices[0].size, stc.vertices[1].size]
    m = abs(stc.data[:n_sources[0], t_idx]).max()
    plot_kwargs["clim"] = dict(kind='value', pos_lims=[0., 0.2 * m, m])
    brain = stc.plot(**plot_kwargs)
    brain.add_text(0.1, 0.9, subject + "_groupmne", "title")
    brain.add_label(g_post_central, borders=True, color="green")

#####################################
# Group MNE leads to better accuracy
# ----------------------------------
# To evaluate the effect of the joint inverse solution, we compute the
# individual solutions independently for each subject


for subject, fwd, evoked, cov in zip(subjects, fwds_, evokeds, noise_covs):
    fwd_ = prepare_fwds([fwd], src_ref)
    stc = compute_group_inverse(fwd_, [ev], [cov],
                                method='multitasklasso',
                                spatiotemporal=True,
                                alpha=0.8)[0]
    stc.subject = subject
    g_post_central = mne.read_labels_from_annot(subject, "aparc.a2009s",
                                                subjects_dir=subjects_dir,
                                                regexp="G_postcentral-lh")[0]
    n_sources = [stc.vertices[0].size, stc.vertices[1].size]
    m = abs(stc.data[:n_sources[0], t_idx]).max()
    plot_kwargs["clim"] = dict(kind='value', pos_lims=[0., 0.2 * m, m])
    brain = stc.plot(**plot_kwargs)
    brain.add_text(0.1, 0.9, subject + "_mxne", "title")
    brain.add_label(g_post_central, borders=True, color="green")


###########################################
# References
# ----------
# [1] Michael Lim, Justin M. Ales, Benoit R. Cottereau, Trevor Hastie,
# Anthony M. Norcia. Sparse EEG/MEG source estimation via a group lasso,
# PLOS ONE, 2017
#
# [2] Jussi Nurminen, Hilla Paananen, & Jyrki Mäkelä. (2017). High frequency
# somatosensory MEG: evoked responses, FreeSurfer reconstruction [Data set].
# Zenodo. http://doi.org/10.5281/zenodo.889235
