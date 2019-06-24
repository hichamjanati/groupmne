"""
Multi-subject joint source localization with multi-task models
==============================================================

The aim of this tutorial is to show how to leverage functional similarity
across subjects to improve source localization. For that purpose we use the
the high frequency SEF MEG dataset of (Nurminen et al., 2017) which provides
MEG and MRI data for two subjects.
"""

# Author: Hicham Janati (hicham.janati@inria.fr)
#
# License: BSD (3-clause)

import os
import os.path as op
import mne
from mne.datasets import hf_sef
from matplotlib import pyplot as plt

from groupmne import group_model
from groupmne.inverse import compute_group_inverse

##########################################################
# Download and process MEG data
# -----------------------------
#
# Averaged MEG data (and MRI) are provided in "evoked".
# We assume that the noise covariance matrices have been computed
# using the raw data as shown in example plot_process_meg.


data_path = hf_sef.data_path("evoked")
meg_path = data_path + "/MEG/"

data_path = op.expanduser(data_path)
subjects_dir = data_path + "/subjects/"
os.environ['SUBJECTS_DIR'] = subjects_dir

ev_name_s = [meg_path + s for s in ["subject_a/sef_right-ave.fif",
             "subject_b/hf_sef_15min-ave.fif"]]
cov_name_s = [meg_path + s for s in ["subject_a/sef-cov.fif",
              "subject_b/sef-cov.fif"]]

evoked_s = []
f, axes = plt.subplots(1, 2, sharey=True)
for ax, ev_name, ll in zip(axes.ravel(), ev_name_s, ["a", "b"]):
    ev = mne.read_evokeds(ev_name)[0]
    evoked_s.append(ev)
    picks = mne.pick_types(ev.info, meg="grad")
    ev.plot(picks=picks, axes=ax, show=False)
    ax.set_title("Subject %s" % ll, fontsize=15)
plt.show()

#########################################################
# Source and forward modeling
# ---------------------------
# To guarantee an alignment across subjects, we start by
# computing (or reading if available) the source space of the average
# subject of freesurfer `fsaverage`
# If fsaverage is not available, it will be fetched to the data_path

# resolution = 3
# spacing = "ico%d" % resolution
# src_ref = group_model.get_src_reference(spacing=spacing,
#                                         subjects_dir=subjects_dir)
#
# ###################################################################
#
# # the function group_model.compute_fwd morphs the source space src_ref to
# # the surface of each subject by mapping the sulci and gyri patterns
# # and computes their forward operators
#
# subjects = ["subject_a", "subject_b"]
# trans_fname_s = [meg_path + "%s/sef-trans.fif" % s for s in subjects]
# bem_fname_s = [subjects_dir + "%s/bem/%s-5120-bem-sol.fif" % (s, s)
#                for s in subjects]
# # n_jobs = 2
# # parallel, run_func, _ = parallel_func(group_model.compute_fwd,
# #                                       n_jobs=n_jobs)
# #
# # fwds = parallel(run_func(s, src_ref, info, trans, bem,  mindist=3)
# #                 for s, info, trans, bem in zip(subjects, raw_name_s,
# #                                                trans_fname_s, bem_fname_s))
#
# fwds = []
# noise_cov_s = []
# for s, info, trans, bem, cov_name in zip(subjects, ev_name_s,
#                                          trans_fname_s, bem_fname_s,
#                                          cov_name_s):
#     fwd = group_model.compute_fwd(s, src_ref, info, trans, bem,  mindist=3)
#     fwds.append(fwd)
#     cov = mne.read_cov(cov_name)
#     noise_cov_s.append(cov)
#     del fwd, cov
#
# ############################################
# # We can now compute the data of the inverse problem.
# # `group_info` is a dictionary that contains the selected channels and the
# # alignment maps between src_ref and the subjects which are required if you
# # want to plot source estimates on the brain surface of each subject.
#
# gains, M, group_info = \
#     group_model.compute_inv_data(fwds, src_ref, evoked_s, noise_cov_s,
#                                  ch_type="grad", tmin=0.02, tmax=0.04)
# print("(# subjects, # channels, # sources) = ", gains.shape)
# print("(# subjects, # channels, # time points) = ", M.shape)
#
# ############################################
# # Solve the inverse problems
# # --------------------------
# #
# stcs, log = compute_group_inverse(gains, M, group_info,
#                                   method="grouplasso",
#                                   depth=0.9, alpha=0.1, return_stc=True,
#                                   n_jobs=4)
# t = 0.025
# t_idx = stcs[0].time_as_index(t)
# for stc, subject in zip(stcs, subjects):
#     m = abs(stc.data[:group_info["n_sources"][0], t_idx]).max()
#     surfer_kwargs = dict(
#         clim=dict(kind='value', pos_lims=[0., 0.1 * m, m]),
#         hemi='lh', subjects_dir=subjects_dir, views='lateral',
#         initial_time=t, time_unit='s', size=(800, 800),
#         smoothing_steps=5)
#     brain = stc.plot(**surfer_kwargs)
#     brain.add_text(0.1, 0.9, subject, "title")
