"""util functions."""
import warnings
import os

import numpy as np

from numba import jit

from sklearn.metrics import euclidean_distances

import mne
from mne.source_space import (_ensure_src, _get_morph_src_reordering,
                              _ensure_src_subject, SourceSpaces)
from mne.utils import check_version

from mne.transforms import _get_trans, apply_trans


def get_morph_src_mapping(src_from, src_to, subject_from=None,
                          subject_to=None, subjects_dir=None):
    """Get a mapping between an original source space and its morphed version.

    It is assumed that the number of vertices and their positions match between
    the source spaces, only the ordering is different. This is commonly the
    case when using :func:`morph_source_spaces`.

    Parameters
    ----------
    src_from : instance of SourceSpaces
        The original source space that was morphed to the target subject.
    src_to : instance of SourceSpaces | list of two arrays
        Either the source space to which ``src_from`` was morphed, or the
        vertex numbers of this source space.
    subject_from : str | None
        The name of the Freesurfer subject to which ``src_from`` belongs. By
        default, the value stored in the SourceSpaces object is used.
    subject_to : str | None
        The name of the Freesurfer subject to which ``src_to`` belongs. By
        default, the value stored in the SourceSpaces object is used.
    subjects_dir : str | None
        Path to SUBJECTS_DIR if it is not set in the environment.

    Returns
    -------
    from_to : dict | pair of dicts
        For each hemisphere, a
        dictionary mapping vertex numbers from src_from -> src_to.
    to_from : dict | pair of dicts
        For each hemisphere, a
        dictionary mapping vertex numbers from src_to -> src_from.

    See also
    --------
    _get_morph_src_reordering

    """
    if subject_from is None:
        subject_from = src_from[0]['subject_his_id']
    if subject_to is None:
        subject_to = src_to[0]['subject_his_id']
    subjects_dir = mne.utils.get_subjects_dir(subjects_dir, raise_error=True)

    if check_version('mne', '0.19'):
        src_from = _ensure_src(src_from, kind='surface')
    else:  # older version of mne
        src_from = _ensure_src(src_from, kind='surf')
    subject_from = _ensure_src_subject(src_from, subject_from)

    if isinstance(src_to, SourceSpaces):
        to_vert_lh = src_to[0]['vertno']
        to_vert_rh = src_to[1]['vertno']
    else:
        if subject_to is None:
            ValueError('When supplying vertex numbers as `src_to`, the '
                       '`subject_to` parameter must be set.')
        to_vert_lh, to_vert_rh = src_to

    order, from_vert = _get_morph_src_reordering(
        [to_vert_lh, to_vert_rh], src_from, subject_from, subject_to,
        subjects_dir=subjects_dir
    )
    from_vert_lh, from_vert_rh = from_vert

    # Re-order the vertices of src_to to match the ordering of src_from
    to_n_lh = len(to_vert_lh)
    to_vert_lh = to_vert_lh[order[:to_n_lh]]
    to_vert_rh = to_vert_rh[order[to_n_lh:] - to_n_lh]

    # Create the mappings
    from_to = [dict(zip(from_vert_lh, to_vert_lh)),
               dict(zip(from_vert_rh, to_vert_rh))]
    to_from = [dict(zip(to_vert_lh, from_vert_lh)),
               dict(zip(to_vert_rh, from_vert_rh))]

    return from_to, to_from


def _make_stc(data, vertices, tstep=0.1, tmin=0., subject=None):
    """Create stc from data."""
    n_sources_l, n_sources_r = [len(v) for v in vertices]
    assert data.shape[0] == n_sources_l + n_sources_r
    data_l = data[:n_sources_l]
    data_r = data[n_sources_l:]
    data_lr = [data_l, data_r]
    ns = [n_sources_l, n_sources_r]
    data = []
    for i, (v, n) in enumerate(zip(vertices, ns)):
        if n:
            order = np.argsort(v)
            data.append(data_lr[i][order])
            vertices[i] = np.asarray(vertices[i][order])
    if n_sources_l * n_sources_r:
        data = np.concatenate(data)
    else:
        data = np.array(data[0])
    stc = mne.SourceEstimate(data, vertices, tstep=tstep, tmin=tmin,
                             subject=subject)
    return stc


def _get_channels(forward, noise_cov=None):
    """Get channels from a forward object and exclude bad ones."""
    fwd_sol_ch_names = forward['sol']['row_names']
    info = forward["info"]
    all_ch_names = set(fwd_sol_ch_names)
    all_bads = set(info['bads'])
    if noise_cov is not None:
        all_ch_names &= set(noise_cov['names'])
        all_bads |= set(noise_cov['bads'])
    ch_names = [c['ch_name'] for c in info['chs']
                if c['ch_name'] not in all_bads and
                c['ch_name'] in all_ch_names]
    return ch_names


def _filter_channels(info, ch_names, ch_type):
    """Filter bad channels and keep only ch_type."""
    if ch_type in ["mag", "grad"]:
        meg = ch_type
        eeg = False
    elif ch_type == "eeg":
        meg = False
        eeg = True
    else:
        raise ValueError("""ch_type must be in ("mag", "grad", "eeg").
                         Got %s""" % ch_type)
    sel_type = mne.pick_types(info, eeg=eeg, meg=meg)
    all_channels = info["ch_names"]
    sel = [all_channels.index(name) for name in ch_names]
    sel = list(set(sel).intersection(sel_type))
    return sel


def _make_fake_group_info(n_sources=2562, n_subjects=2, hemi="lh"):
    """Create fake group info for testing."""
    if hemi == "both":
        n_sources /= 2
    group_info = dict(subjects=[str(i) for i in range(n_subjects)])
    group_info["ch_names"] = [None]
    group_info["vertno_lh"] = n_subjects * [np.arange(n_sources)]
    group_info["vertno_rh"] = n_subjects * [np.arange(n_sources)]
    group_info["vertno_ref"] = np.arange(n_sources)
    group_info["ch_filter"] = True
    group_info["n_sources"] = [n_sources, n_sources]
    group_info["hemi"] = hemi
    group_info["tmin"] = 0.
    group_info["tstep"] = 0.1
    return group_info


def _compute_coreg_dist(subject, trans_fname, info_fname, subjects_dir):
    """Assess quality of coregistration."""
    trans = mne.read_trans(trans_fname)

    high_res_surf = subjects_dir + "/%s/surf/lh.seghead" % subject
    low_res_surf = subjects_dir + "/%s/bem/%s-outer_skull.surf" % (subject,
                                                                   subject)
    low_res_surf_2 = subjects_dir + "/%s/bem/outer_skull.surf" % subject
    if os.path.exists(high_res_surf):
        pts, _ = mne.read_surface(high_res_surf, verbose=False)
        pts /= 1e3  # convert to mm
    elif os.path.exists(low_res_surf):
        warnings.warn("""Using low resolution head surface,
                         the average distance will be potentially
                         overestimated""")
        pts, _ = mne.read_surface(low_res_surf, verbose=False)
        pts /= 1e3  # convert to mm
    elif os.path.exists(low_res_surf_2):
        warnings.warn("""Using low resolution head surface,
                         the average distance will be potentially
                         overestimated""")
        pts, _ = mne.read_surface(low_res_surf_2, verbose=False)
        pts /= 1e3  # convert to mm
    else:
        raise FileNotFoundError("No MRI surface was found!")
    trans = _get_trans(trans, fro="mri", to="head")[0]
    pts = apply_trans(trans, pts)
    info = mne.io.read_info(info_fname, verbose=False)
    info_dig = np.stack([list(x["r"]) for x in info["dig"]],
                        axis=0)
    M = euclidean_distances(info_dig, pts)
    idx = np.argmin(M, axis=1)
    dist = M[np.arange(len(info_dig)), idx].mean()

    return dist


def mesh_all_distances(points, tris, verts=None):
    """Compute all pairwise distances on the mesh based on edges lengths
    using Floyd-Warshall algorithm
    """
    A = mne.surface.mesh_dist(tris, points)
    if verts is not None:
        A = A[verts][:, verts]
    A = A.toarray()
    A[A == 0.] = 1e6
    A.flat[::len(A) + 1] = 0.
    A = floyd_warshall(A)
    return A


@jit(nogil=True, cache=True, nopython=True)
def floyd_warshall(dist):
    npoints = dist.shape[0]
    for k in range(npoints):
        for i in range(npoints):
            for j in range(npoints):
                # If i and j are different nodes and if
                # the paths between i and k and between
                # k and j exist, do
                # d_ikj = min(dist[i, k] + dist[k, j], dist[i, j])
                d_ikj = dist[i, k] + dist[k, j]
                if ((d_ikj != 0.) and (i != j)):
                    # See if you can't get a shorter path
                    # between i and j by interspacing
                    # k somewhere along the current
                    # path
                    if ((d_ikj < dist[i, j]) or (dist[i, j] == 0)):
                        dist[i, j] = d_ikj
    return dist


def compute_ground_metric(src, group_info):
    """Compute geodesic distance matrix on the triangulated mesh of src."""
    vertnos_filtered = group_info["vertno_ref"]
    hemis = ["lh", "rh"]
    Ds_f, Ds = [], []
    for i, h in enumerate(hemis):
        tris = src[i]["use_tris"]
        vertno = src[i]["vertno"]
        points = src[i]["rr"][vertno]

        vert_used = vertnos_filtered[i]

        D = mesh_all_distances(points, tris)
        D_filtered = D[vert_used][:, vert_used]

        Ds_f.append(D_filtered)
        Ds.append(D)

    n1, n2 = len(Ds_f[0]), len(Ds_f[1])
    D_filtered = (Ds_f[0]).max() * np.ones((n1 + n2, n1 + n2))
    D_filtered[:n1, :n1] = Ds_f[0]
    D_filtered[n1:, n1:] = Ds_f[1]

    n1, n2 = len(Ds[0]), len(Ds[1])
    D = (Ds[0]).max() * np.ones((n1 + n2, n1 + n2))
    D[:n1, :n1] = Ds[0]
    D[n1:, n1:] = Ds[1]
    return D_filtered
