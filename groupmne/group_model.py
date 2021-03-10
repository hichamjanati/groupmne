"""
Multi-subject source modeling.

This module implements the computation of the forward operators with aligned
source locations across subjects. This is done through morphing a reference
head model (fsaverage by default) to the surface of each subject.
"""

from copy import deepcopy

import mne

import numpy as np

from . import utils


def compute_fwd(subject, src_ref, info, trans_fname, bem_fname,
                meg=True, eeg=True, mindist=2, subjects_dir=None,
                n_jobs=1, verbose=None):
    """Morph the source space of fsaverage to a subject.

    Parameters
    ----------
    subject : str
        Name of the reference subject.
    src_ref : instance of SourceSpaces
        Source space of the reference subject. See `get_src_reference.`
    info : str | instance of mne.Info
        Instance of an MNE info file or path to a raw fif file.
    trans_fname : str
        Path to the trans file of the subject.
    bem_fname : str
        Path to the bem solution of the subject.
    meg : bool
        Include MEG channels or not.
    eeg : bool
        Include EEG channels or not.
    mindist : float
        Safety distance from the outer skull. Sources below `mindist` will be
        discarded in the forward operator.
    subjects_dir : str
        Path to the freesurfer `subjects` directory.
    n_jobs : int
        The number jobs to run in parallel.
    verbose : None | bool
        Use verbose mode. If None use MNE default.

    """
    print("Processing subject %s" % subject)

    src = mne.morph_source_spaces(src_ref, subject_to=subject,
                                  verbose=verbose,
                                  subjects_dir=subjects_dir)
    bem = mne.read_bem_solution(bem_fname, verbose=verbose)
    fwd = mne.make_forward_solution(info, trans=trans_fname, src=src,
                                    bem=bem, meg=meg, eeg=eeg,
                                    mindist=mindist, verbose=verbose,
                                    n_jobs=n_jobs)
    return fwd


def prepare_fwds(fwds, src_ref, copy=True, subjects_dir=None):
    """Compute the group alignement of the forward operators.

    Parameters
    ----------
    fwds : list of `mne.Forward`
        The forward operators computed on the morphed source
        space `src_ref`.
    src_ref : instance of SourceSpace instance
        Reference source model.
    copy : bool
        If copy is False the fwds are modified inplace.

    Returns
    -------
    fwds : list of `mne.Forward`
        Prepared forward operators.

    """
    n_sources = [src["nuse"] for src in src_ref]
    vertno_ref = [src["vertno"].tolist() for src in src_ref]
    vertices = [], []
    positions = [], []
    gains = []
    group_info = dict(subjects=[])

    if copy:
        fwds = [deepcopy(fwd) for fwd in fwds]

    # compute gain matrices
    for fwd in fwds:
        fwd = mne.convert_forward_solution(fwd, surf_ori=True,
                                           force_fixed=True,
                                           use_cps=True,
                                           verbose=False)
        src = fwd["src"]
        subject = src[0]["subject_his_id"]
        group_info["subjects"].append(subject)

        # find mapping between src_ref and new src space of fwd
        # src may have less sources than src_ref if they are
        # outside the inner skull
        mapping = utils.get_morph_src_mapping(src_ref, src,
                                              subjects_dir=subjects_dir)

        # create a gain with the full src_ref resolution
        gain = np.ones((fwd["nchan"], sum(n_sources)))
        for i in range(2):
            # pos contains the reference alignment sources of src_ref
            # if no source is elliminated, it is given by np.arange(n_sources)
            vertno_ref_kept_ = list(mapping[0][i].keys())
            pos = [vertno_ref[i].index(v) for v in vertno_ref_kept_]
            pos = np.array(pos)
            positions[i].append(pos)
            vertno = - np.ones(n_sources[i]).astype(int)
            # re-order columns of the gain matrices
            # vertno_tmp contains the new vertices in the right order
            vertno_tmp = np.array(list(mapping[0][i].values()))
            # find the appropriate permutation to reorder columns
            permutation = np.argsort(np.argsort(vertno_tmp))

            # these indices allow to switch between left and right hemispheres
            col_0 = i * n_sources[0]
            col_1 = i * fwd["src"][0]["nuse"]
            full_gain = fwd["sol"]["data"]

            # filter the gain on kept sources
            gain[:, col_0 + pos] = full_gain[:, col_1 + permutation]

            # add the kept vertices to the list
            vertno[pos] = vertno_tmp
            vertices[i].append(vertno)
        gains.append(gain)

    # Now we can compute the intersection of all common vertices
    common_order_lh = np.array(list(
        set(positions[0][0]).intersection(*positions[0])))
    common_order_rh = np.array(list(
        set(positions[1][0]).intersection(*positions[1])))

    common_order = [common_order_lh, common_order_rh]
    vertno_ref = [np.array(vertno_ref)[0][common_order_lh],
                  np.array(vertno_ref)[1][common_order_rh]]

    # Compute the final filtered gains
    gain_filter = np.r_[common_order_lh, common_order_rh + n_sources[0]]

    for i in range(len(fwds)):
        gains[i] = gains[i][:, gain_filter]
        for j, common in enumerate(common_order):
            vertices[j][i] = vertices[j][i][common.astype(int)]

    group_info["vertno_lh"] = vertices[0]
    group_info["vertno_rh"] = vertices[1]
    group_info["vertno_ref"] = vertno_ref
    group_info["common_order"] = common_order
    group_info["n_sources"] = [len(common_order_lh), len(common_order_rh)]

    for fwd, gain in zip(fwds, gains):
        fwd["sol_group"] = dict(data=gain, group_info=group_info,
                                src_ref=src_ref, n_subjects=len(fwds))
    return fwds
