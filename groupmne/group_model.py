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
        # find removed vertices
        mapping = utils.get_morph_src_mapping(src_ref, src)
        gain = []
        for i in range(2):
            pos = list(mapping[0][i].keys())
            positions[i].append(pos)
            vertno = - np.ones(n_sources[i]).astype(int)
            gain_h = np.ones((fwd["nchan"], n_sources[i]))
            # re-order columns of the gain matrices
            vertno_tmp = np.array(list(mapping[0][i].values()))
            permutation = np.argsort(np.argsort(vertno_tmp))
            gain_h[:, pos] = fwd["sol"]["data"][:, permutation]
            vertno[pos] = vertno_tmp
            gain.append(gain_h)
            vertices[i].append(vertno)

        gain = np.hstack(gain)
        gains.append(gain)

    common_pos_lh = np.array(list(
        set(positions[0][0]).intersection(*positions[0])))
    common_pos_rh = np.array(list(
        set(positions[1][0]).intersection(*positions[1])))
    common_pos = np.r_[common_pos_lh, common_pos_rh + len(positions[0])]
    vertno_ref = [common_pos_lh, common_pos_rh]
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


def compute_inv_data(fwds, src_ref, evokeds, noise_cov_s, ch_type="grad",
                     tmin=None, tmax=None):
    """Compute aligned gain matrices and M-EEG data of the group of subjects.

    Parameters
    ----------
    fwds : list
        The forward operators computed on the morphed source
        space `src_ref`.
    src_ref : instance of SourceSpace instance
        Reference source model.
    evokeds : list
        The Evoked instances, one element for each subject.
    noise_cov_s : list of instances of Covariance
        The noise covariances, one element for each subject.
    ch_type : str (default "grad")
        Type of channels used for source reconstruction. Can be one
        of ("mag", "grad", "eeg"). Using more than one type of channels is not
        yet supported.
    tmin: float (default None)
        initial time point.
    tmax: float (default None)
        final time point.

    Returns
    -------
    gains: ndarray, shape (n_subjects, n_channels, n_sources)
        The gain matrices.
    M: ndarray, shape (n_subjects, n_channels, n_times)
        M-EEG data.
    group_info: dict
        Group information (channels, alignments maps across subjects)

    """
    if len(fwds) != len(noise_cov_s):
        raise ValueError("""The length of `fwds` and `noise_cov_s`
                         do not match.""")
    gains, group_info = _group_filtering(fwds, src_ref, noise_covs=noise_cov_s)
    info = fwds[0]["info"]
    ch_names = group_info["ch_names"]
    sel = utils._filter_channels(info, ch_names, ch_type)
    group_info["ch_filter"] = True
    group_info["sel"] = sel
    gains = gains[:, sel, :]
    M = []
    for i, (noise_cov, evoked, gain) \
            in enumerate(zip(noise_cov_s, evokeds, gains)):
        ev = evoked.copy()
        ev.crop(tmin, tmax)
        whitener, _ = mne.cov.compute_whitener(noise_cov, ev.info,
                                               sel, pca=False)
        gains[i] = (ev.nave) ** 0.5 * whitener.dot(gain)
        M.append((ev.nave) ** 0.5 * whitener.dot(ev.data[sel]))
    group_info["tmin"] = tmin
    if len(ev.times) > 1:
        tstep = ev.times[1] - ev.times[0]
    else:
        tstep = 0.
    group_info["tstep"] = tstep
    M = np.stack(M, axis=0)
    group_info["hemi"] = "both"
    return gains, M, group_info
