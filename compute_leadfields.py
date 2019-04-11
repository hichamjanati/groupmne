import os
import config as cfg
import numpy as np
import mne
from mne.parallel import parallel_func
from os import path as op
from brain_utils import get_multiple_indices
from shutil import rmtree
from morph_map import get_morph_src_mapping


dataset_name = "camcan"
save_dir = "~/data/%s/" % dataset_name
save_dir = op.expanduser(save_dir)
subfolders = ["bem", "leadfields", "vertno"]
for subfolder in subfolders:
    path = save_dir + subfolder
    try:
        rmtree(path)
    except FileNotFoundError:
        pass
    os.makedirs(path)
age_max = 30
subjects_dir = cfg.get_subjects_dir(dataset_name)
subjects = cfg.get_subjects_list(dataset_name, 0, age_max)
os.environ['SUBJECTS_DIR'] = subjects_dir


def compute_src_fs(resolution):
    """Compute source space of fsaverage."""
    subject = "fsaverage"
    spacing = "ico%d" % resolution
    print("processing subject: %s" % subject)

    # Create the surface source space
    fname_src_fs = op.join(subjects_dir, subject, 'bem', '%s-%s-src.fif'
                           % (subject, spacing))
    if os.path.isfile(fname_src_fs):
        src_fs = mne.read_source_spaces(fname_src_fs)
    else:
        src_fs = mne.setup_source_space(subject=subject,
                                        spacing=spacing,
                                        subjects_dir=subjects_dir,
                                        add_dist=False)
    return src_fs


def compute_fwd(subject, src_fs, resolution=4, plot=False):
    """Morph source space of fsaverage to subject."""
    spacing = "ico%d" % resolution
    print("Processing subject %s" % subject)
    trans_fname = cfg.get_trans_fname(dataset_name, subject)
    raw_fname = cfg.get_raw_fname(dataset_name, subject)
    info = mne.io.read_info(raw_fname)
    picks = mne.pick_types(info, meg=True, exclude=[])
    info = mne.pick_info(info, picks)

    # morph src space
    src = mne.morph_source_spaces(src_fs, subject_to=subject,
                                  subjects_dir=subjects_dir)
    order = get_morph_src_mapping(src_fs, src, subjects_dir=subjects_dir,
                                  indices=False)[0]
    for v, hemi in zip(order, ["lh", "rh"]):
        vertno_name = save_dir + "vertno/%s-%s-morphed-%s-vrt.npy" %\
            (subject, spacing, hemi)
        np.save(vertno_name, np.array(list(v.values())))

    # old version with hacked morph-spaces in mne
    # vertno = np.stack([s['vertno'] for s in src], axis=0)
    # for v, hemi in zip(vertno, ["lh", "rh"]):
    #     vertno_name = save_dir + "vertno/%s-%s-morphed-%s-vrt.npy" %\
    #         (subject, spacing, hemi)
    #     np.save(vertno_name, v)
    bem_fname = cfg.get_bem_fname(dataset_name, subject)
    bem = mne.read_bem_solution(bem_fname)
    fwd = mne.make_forward_solution(info, trans=trans_fname, src=src, bem=bem,
                                    meg=True, eeg=False, mindist=2.,
                                    n_jobs=2)
    fwd_fname = save_dir + "bem/%s-ico4-fwd.fif" % (subject)
    mne.write_forward_solution(fwd_fname, fwd, overwrite=True)


if __name__ == "__main__":

    resolution = 4
    n_subjects = len(subjects)
    spacing = "ico%d" % resolution

    ##########
    # COMPUTE FILTERED FORWARDS
    src_fname = save_dir + \
        "bem/fsaverage-%s-morphed-src.fif" % spacing
    src_fs = compute_src_fs(resolution)
    mne.write_source_spaces(src_fname, src_fs, overwrite=True)

    parallel, run_func, _ = parallel_func(compute_fwd, n_jobs=len(subjects))
    Xs = parallel(run_func(subject, src_fs, resolution)
                  for subject in subjects)
    subject = "fsaverage"
    spacing = "ico4"
    fs_vert = np.arange(2562)
    for h in ["lh", "rh"]:
        fs_vert_fname = save_dir + "vertno/%s-%s-morphed-%s-vrt.npy" %\
            (subject, spacing, h)
        np.save(fs_vert_fname, fs_vert)

    ############
    # COMPUTE ALL ELIMINATED VERTICES IN THE SAME REFERENCE (MORPHED ORDER)

    idx_out = [[], []]
    idx_out_per_sub = [[], []]
    common_vertices = np.ones((2, 2562))
    verts_morphed = [[], []]
    vertices_out = [[], []]
    n_reject = [[], []]
    all_data_lh = {}
    for i, h in enumerate(["lh", "rh"]):
        for j, sub in enumerate(subjects):
            vert_fname = save_dir + "vertno/%s-ico4-morphed-%s-vrt.npy" %\
                (sub, h)
            all_verts = np.load(vert_fname)
            if i == 0:
                all_data_lh[j] = all_verts
            verts_morphed[i].append(all_verts)
            fwd_fname = save_dir + "bem/%s-%s-fwd.fif" % \
                (sub, spacing)
            fwd = mne.read_forward_solution(fwd_fname)
            src_filtered = fwd['src']
            verts = src_filtered[i]["vertno"]
            stars = np.array(list(set(all_verts) - set(verts)))
            n_reject[i].append(len(stars))
            idx = get_multiple_indices(stars, all_verts)
            idx_out[i].extend(idx)
            idx_out_per_sub[i].append(idx)
            vertices_out[i].append(set(stars))
        stars_all = np.unique(idx_out[i])
        if len(stars_all):
            common_vertices[i, stars_all] = 0
        for j, sub in enumerate(subjects + ["fsaverage"]):
            vert_fname = save_dir + "vertno/%s-ico4-morphed-%s-vrt.npy" % (sub, h)
            vert_ordered = np.load(vert_fname)
            vert_filtered = vert_ordered[common_vertices[i].astype(bool)]
            vert_fname = save_dir + \
                "vertno/%s-ico4-filtered-%s-vrt.npy" % (sub, h)
            np.save(vert_fname, vert_filtered)

    #############
    # ELIMINATE VERTICES FROM ALL LEADFIELD MATRIX IN A (WRONG) ORDERED
    # VERTNO AND THEN PERMUTE BACK TO THE CORRECT MORPHED ORDER

    n_sources = common_vertices.sum(1).astype(int)
    for j, sub in enumerate(subjects):
        fwd_fname = save_dir + "bem/%s-%s-fwd.fif" % \
            (sub, spacing)
        fwd = mne.read_forward_solution(fwd_fname)
        src = fwd['src']
        verts_filtered = []
        for sr, h in zip(src, ["lh", "rh"]):
            vert_fname = save_dir + "vertno/%s-ico4-filtered-%s-vrt.npy" % \
                (sub, h)
            vertidx = np.load(vert_fname)
            verts_filtered.append(vertidx)
            sr['vertno'] = vertidx
            sr['nuse'] = len(sr['vertno'])
            sr['inuse'][:] = 0
            sr['inuse'][sr['vertno']] = 1
        # no need to save filtered-src
        # src_fname_f = save_dir + \
        #     "bem/%s-ico4-filtered-src.fif" % (sub)
        # mne.write_source_spaces(src_fname_f, src, overwrite=True)
        fwd_fname = save_dir + "bem/%s-%s-fwd.fif" % \
            (sub, spacing)
        fwd = mne.read_forward_solution(fwd_fname)
        # fwd['src'] = src
        fwd = mne.convert_forward_solution(fwd, surf_ori=True,
                                           force_fixed=True,
                                           use_cps=True)

        n_l = fwd['src'][0]['nuse']
        n_r = fwd['src'][1]['nuse']
        # CLEAN UP BAD VERTICES OF OTHER SUBJECTS
        verts_fwd_l = set(fwd["src"][0]["vertno"])
        verts_fwd_r = set(fwd["src"][1]["vertno"])
        verts_out_l = list(verts_fwd_l - set(verts_filtered[0]))
        verts_out_r = list(verts_fwd_r - set(verts_filtered[1]))

        indx_l = get_multiple_indices(verts_out_l, list(fwd["src"][0]["vertno"]))
        indx_r = get_multiple_indices(verts_out_r, list(fwd["src"][1]["vertno"]))

        mask_l = np.ones(n_l).astype(bool)
        mask_r = np.ones(n_r).astype(bool)
        mask_l[indx_l] = False
        mask_r[indx_r] = False
        mask = np.r_[mask_l, mask_r]
        leadfield = fwd['sol']['data'][:, mask]

        # PERMUTE LEADFIELD COLUMNS SO AS THEY MATCH ACROSS SUBJECTS
        permutation_l = np.argsort(np.argsort(verts_filtered[0]))
        permutation_r = np.argsort(np.argsort(verts_filtered[1])) + n_sources[0]
        permutation = np.r_[permutation_l, permutation_r]
        leadfield = leadfield[:, permutation]
        print("Leadfield shape: ", leadfield.shape)

        assert n_sources[0] == mask_l.sum()
        assert n_sources[1] == mask_r.sum()

        # restric left / right hemispheres and save
        leadfield_lh = leadfield[:, :n_sources[0]]
        leadfield_rh = leadfield[:, n_sources[0]:]
        lfs = [leadfield_lh, leadfield_rh, leadfield]

        for i, (leadfield, hemi) in enumerate(zip(lfs, ["lh", "rh", "lrh"])):
            leadfield_fname = save_dir + "leadfields/X_%s_%s_ico4.npy" % \
                (sub, hemi)
            np.save(leadfield_fname, leadfield)

    # UPDATE FSAVERAGE FILES
    for i, (h, src) in enumerate(zip(["lh", "rh"], src_fs)):
        fs_vert = np.arange(2562)
        fs_vert = fs_vert[common_vertices[i].astype(bool)]
        fs_vert_fname = save_dir + "vertno/%s-%s-filtered-%s-vrt.npy" %\
            (subject, spacing, h)
        np.save(fs_vert_fname, fs_vert)

        src['vertno'] = fs_vert
        src['nuse'] = len(src['vertno'])
        src['inuse'][:] = 0
        src['inuse'][src['vertno']] = 1
    src_fname = save_dir + \
        "bem/fsaverage-%s-filtered-src.fif" % spacing
    mne.write_source_spaces(src_fname, src_fs, overwrite=True)
