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


def get_src_reference(subject="fsaverage", spacing="ico4",
                      subjects_dir=None, fetch_fsaverage=False):
    """Compute source space of the reference subject.

    Parameters
    ----------
    subject : str
        Name of the reference subject.
    spacing : str
        The spacing to use. Can be ``'ico#'`` for a recursively
        subdivided icosahedron, ``'oct#'`` for a recursively subdivided
        octahedron, ``'all'`` for all points, or an integer to use
        appoximate distance-based spacing (in mm).
    fetch_fsaverage : bool
        If `True` and `subject` is fsaverage the data of fsaverage is fetched
        if not found.

    Returns
    -------
    src : SourceSpaces
        The source space for each hemisphere.

    """
    subjects_dir = \
        mne.utils.get_subjects_dir(subjects_dir=subjects_dir, raise_error=True)
    fname_src = op.join(subjects_dir, subject, 'bem', '%s-%s-src.fif'
                        % (subject, spacing))
    if os.path.isfile(fname_src):
        src_ref = mne.read_source_spaces(fname_src)
    elif os.path.exists(os.path.join(subjects_dir, subject)):
        src_ref = mne.setup_source_space(subject=subject,
                                         spacing=spacing,
                                         subjects_dir=subjects_dir,
                                         add_dist=False)
    elif subject == "fsaverage":
        if fetch_fsaverage:
            mne.datasets.fetch_fsaverage(subjects_dir)
        else:
            raise FileNotFoundError("The fsaverage data could not be found." +
                                    "To download the fsaverage data, set " +
                                    "`fetch_fsaverage` to True")
        src_ref = mne.setup_source_space(subject=subject,
                                         spacing=spacing,
                                         subjects_dir=subjects_dir,
                                         add_dist=False)
    else:
        raise FileNotFoundError("The data of %s could not be found" % subject)
    return src_ref


def compute_fwd(subject, src_ref, info, trans_fname, bem_fname,
                meg=True, eeg=True, mindist=2, subjects_dir=None,
                n_jobs=1):
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
                                  subjects_dir=subjects_dir)
    bem = mne.read_bem_solution(bem_fname)
    fwd = mne.make_forward_solution(info, trans=trans_fname, src=src,
                                    bem=bem, meg=meg, eeg=eeg,
                                    mindist=mindist,
                                    n_jobs=n_jobs)
    return fwd


def _group_filtering(fwds, src_ref, noise_covs=None):
    """Get common vertices across subjects."""
    n_sources = [src_ref[i]["nuse"] for i in [0, 1]]
    vertices = [], []
    positions = [], []
    gains = []
    ch_names = []
    group_info = dict(subjects=[])
    if noise_covs is None:
        noise_covs = len(fwds) * [None]
    # compute gain matrices
    for fwd, cov in zip(fwds, noise_covs):
        fwd = mne.convert_forward_solution(fwd, surf_ori=True,
                                           force_fixed=True,
                                           use_cps=True)
        src = fwd["src"]
        subject = src[0]["subject_his_id"]
        group_info["subjects"].append(subject)
        ch_names.append(utils._get_channels(fwd, cov))

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
    gains = np.stack(gains, axis=0)

    ch_names = set(ch_names[0]).intersection(*ch_names)
    group_info["ch_names"] = list(ch_names)
    group_info["vertno_lh"] = vertices[0]
    group_info["vertno_rh"] = vertices[1]
    group_info["vertno_ref"] = vertno_ref
    group_info["ch_filter"] = False
    group_info["n_sources"] = [len(common_pos_lh), len(common_pos_rh)]

    return gains, group_info


def compute_gains(fwds, src_ref, ch_type="grad", hemi="lh"):
    """Compute aligned gain matrices of the group of subjects.

    Computation is done with respect to a reference source space.

    Parameters
    ----------
    fwds : list
        The forward operators computed on the morphed source
        space `src_ref`.
    src_ref : instance of SourceSpace instance
        Reference source model.
    ch_type : str
        Type of channels used for source reconstruction. Can be one
        of ("mag", "grad", "eeg"). Using more than one type of channels is not
        yet supported.
    hemi : str
        Hemisphere, "lh", "rh" or "both".

    Returns
    -------
    gains: ndarray, shape (n_subjects, n_channels, n_sources)
        The gain matrices.
    group_info: dict
        Group information (channels, alignments maps across subjects)

    """
    gains, group_info = _group_filtering(fwds, src_ref, noise_covs=None)
    n_lh = group_info["n_sources"][0]
    if hemi == "lh":
        col0 = 0
        col1 = n_lh
    elif hemi == "rh":
        col0 = n_lh
        col1 = None
    elif hemi == "both":
        col0 = 0
        col1 = None
    else:
        raise ValueError("hemi must be in ('lh', 'rh', 'both')")
    info = fwds[0]["info"]
    ch_names = group_info["ch_names"]
    sel = utils._filter_channels(info, ch_names, ch_type)
    group_info["ch_filter"] = True
    group_info["sel"] = sel
    gains = gains[:, sel, :]
    group_info["hemi"] = hemi
    return gains[:, :, col0:col1], group_info
