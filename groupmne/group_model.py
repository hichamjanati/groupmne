"""
Multi-subject source modeling.

This module implements the computation of the forward operators with aligned
source locations across subjects. This is done through morphing a reference
head model (fsaverage by default) to the surface of each subject.
"""

import mne
import os
import os.path as op

import numpy as np

from . import utils


def compute_fwd(subject, src_ref, info, trans_fname, bem_fname,
                meg=True, eeg=True, mindist=2, subjects_dir=None,
                n_jobs=1, verbose=False):
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
    mindist : float
        Safety distance from the outer skull. Sources below `mindist` will be
        discarded in the forward operator.
    subjects_dir : str
        Path to the freesurfer `subjects` directory.

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


def prepare_fwds(fwds, src_ref):
    """Compute the group alignement of the forward operators.

    Parameters
    ----------
    fwds : list of `mne.Forward`
        The forward operators computed on the morphed source
        space `src_ref`.
    src_ref : instance of SourceSpace instance
        Reference source model.

    Returns
    -------
    fwds : list of `mne.Forward`
        Prepared forward operators.

    """
    n_sources = [src_ref[i]["nuse"] for i in [0, 1]]
    vertices = [], []
    positions = [], []
    gains = []
    group_info = dict(subjects=[])

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
        mapping = utils.get_morph_src_mapping(src_ref, src)

        # create a gain with the full src_ref resolution
        gain = np.ones((fwd["nchan"], sum(n_sources)))
        for i in range(2):
            # pos contains the reference sources of src_ref
            pos = list(mapping[0][i].keys())
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
    common_pos_lh = np.array(list(
        set(positions[0][0]).intersection(*positions[0])))
    common_pos_rh = np.array(list(
        set(positions[1][0]).intersection(*positions[1])))

    common_pos = np.r_[common_pos_lh, common_pos_rh + n_sources[0]]
    vertno_ref = [common_pos_lh, common_pos_rh]

    # Compute the final filtered gains
    for i in range(len(fwds)):
        gains[i] = gains[i][:, common_pos]
        for j, common in enumerate(vertno_ref):
            vertices[j][i] = vertices[j][i][common.astype(int)]

    group_info["vertno_lh"] = vertices[0]
    group_info["vertno_rh"] = vertices[1]
    group_info["vertno_ref"] = vertno_ref
    group_info["n_sources"] = [len(common_pos_lh), len(common_pos_rh)]

    for fwd, gain in zip(fwds, gains):
        fwd["sol_group"] = dict(data=gain, group_info=group_info,
                                src_ref=src_ref, n_subjects=len(fwds))
    return fwds
