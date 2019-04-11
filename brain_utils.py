import numpy as np
import mne


def save_as_stc(fname, data, subject, vertices_subject=None):
    brainpath = "data/"
    if vertices_subject is None:
        vertices_subject = subject
    vertices = np.load(brainpath + "vertices/vertices_%s.npy"
                       % vertices_subject)
    stc = dict()
    stc['tmin'] = 0.
    stc['tstep'] = 1.
    stc['vertices'] = vertices
    stc['data'] = data
    mne.source_estimate._write_stc(fname, **stc)


def sum_labels(names, subject="fsaverage", subjects_dir=None):
    labels = []
    for name in names:
        label_fname = subjects_dir + '%s/label/%s.label' % (subject, name)
        label_fs = mne.read_label(label_fname, subject=subject)
        labels.append(label_fs)
    for label in labels:
        label_fs += label
    return label_fs


def get_multiple_indices(values, search_list):
    indices = []
    for v in values:
        indices.extend(np.where(search_list == v)[0].tolist())
    return indices


def get_indices_label(label=None, subject="fsaverage", subjects_dir=None,
                      spacing="ico4", hemi="lh", mode="filtered"):
    vert_fname = subjects_dir + "%s/bem/%s-%s-%s-%s-vrt.npy" %\
        (subject, subject, spacing, mode, hemi)
    all_verts = np.load(vert_fname)
    values = label.get_vertices_used(all_verts)
    indices = get_multiple_indices(values, all_verts)
    return indices


def plot_labels(labels, subject, t=""):
    from surfer import Brain
    from mayavi import mlab
    subjects_dir = 'data/subjects'
    surface = 'inflated'
    hemi = 'lh'
    brainpath = 'data/'
    brain = Brain(subject, hemi, surface, subjects_dir=subjects_dir)
    for label in labels:
        brain.add_label(label, borders=True, color="indianred")
        brain.add_label(label, borders=False, color='yellow')
    brain.hide_colorbar()
    legend = 'Labels'
    legend = legend + ' ' * (14 - len(legend))
    mlab.text(0.1, 0.77, t)
    brain.save_image(brainpath + 'labels/simu_labels_%s_%s.png' %
                     (subject, surface))
