# Compute ground metrix
import os
import numpy as np
from numba import jit
import mne

import config as cfg
from shutil import rmtree


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
    print('Running Floyd-Warshall algorithm')
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


if __name__ == "__main__":
    from joblib import delayed, Parallel

    dataset_name = "camcan"
    save_dir = "~/data/%s/" % dataset_name
    save_dir = os.path.expanduser(save_dir)
    metrics_dir = save_dir + "metrics/"
    try:
        rmtree(metrics_dir)
    except FileNotFoundError:
        pass
    os.makedirs(metrics_dir)

    subjects_dir = cfg.get_subjects_dir(dataset_name)
    subjects = cfg.get_subjects_list(dataset_name, simu=True)
    subjects = ["fsaverage"]
    os.environ['SUBJECTS_DIR'] = subjects_dir

    save = True
    plot = False

    def compute_metric(s, hemis=["lh", "rh"]):
        fwd_fname = save_dir + "bem/%s-ico4-fwd.fif" % s
        if s != "fsaverage":
            src = mne.read_forward_solution(fwd_fname)["src"]
        else:
            src_fname = save_dir + "bem/%s-ico4-morphed-src.fif" % s
            src = mne.read_source_spaces(src_fname)

        Ds = []
        for i, h in enumerate(hemis):
            tris = src[i]["use_tris"]
            vertno = src[i]["vertno"]
            points = src[i]["rr"][vertno]
            D_fname = metrics_dir + "metric_%s_%s.npy" % (s, h)
            vert_fname = save_dir + "vertno/%s-ico4-filtered-%s-vrt.npy" %\
                (s, h)
            vert_used = np.load(vert_fname)

            if os.path.exists(D_fname):
                print('Loading D')
                D_filtered = np.load(D_fname)
            else:
                D = mesh_all_distances(points, tris)
                D_filtered = D[vert_used][:, vert_used]
                if save:
                    np.save(D_fname, D_filtered)
            Ds.append(D_filtered)
        D_fname = metrics_dir + "metric_%s_lrh.npy" % s
        if os.path.exists(D_fname):
            D_filtered = np.load(D_fname)
        elif len(hemis) == 2:
            n1, n2 = len(Ds[0]), len(Ds[1])
            D_filtered = (Ds[0]).max() * np.ones((n1 + n2, n1 + n2))
            D_filtered[:n1, :n1] = Ds[0]
            D_filtered[n1:, n1:] = Ds[1]
            if save:
                np.save(D_fname, D_filtered)

        if plot:
            # Visualize the metric
            from mayavi import mlab
            scalars = D[100]
            mlab.triangular_mesh(*points.T, tris,
                                 scalars=np.log(scalars + 0.1))
            mlab.show()

    pll = Parallel(n_jobs=len(subjects))
    it = (delayed(compute_metric)(s, ["lh"]) for s in subjects)
    out = pll(it)
