import os.path as op
import mne
from mayavi import mlab
from surfer import Brain

meg_path = "~/Dropbox/neuro_transport/code/mtw_experiments/meg/camcan/"
meg_path = op.expanduser(meg_path)

subjects_dir = meg_path + "subjects/"
labels_path = meg_path + "label/"
subject = "fsaverage"
hemi = "lh"

colors = [(1, 0, 0), (0, 0, 1), (0, 1, 1), (1, 1, 0),
          (1, 0, 1), (0.5, 0.2, 0.8), (0.2, 0.9, 0.3),
          (1., 0.5, 0.2), (0.5, 1, 0.9), (1, 0.5, 0.2)]
labels_raw = mne.read_labels_from_annot("fsaverage", "aparc.a2009s",
                                        subjects_dir=subjects_dir)
annot_dict_full = {}
for l in labels_raw:
    annot_dict_full[l.name] = l

sulci_names = ["S_front_middle",
               "S_precentral-sup-part",
               "S_oc_sup_and_transversal",
               "S_orbital-H_Shaped",
               "S_circular_insula_inf"]
gyri_names = ["G_rectus",
              "G_front_inf-Opercular",
              "G_oc-temp_lat-fusifor",
              "G_parietal_sup",
              "G_front_inf-Orbital"]
mix_names = ["S_oc_sup_and_transversal",
             "G_front_inf-Opercular",
             "S_precentral-sup-part",
             "S_front_middle",
             "G_pariet_inf-Supramar",
             "S_occipital_ant",
             "S_orbital_lateral",
             "G_temp_sup-G_T_transv",
             "G_and_S_paracentral"
             ]
label_lists = [sulci_names, gyri_names, mix_names]
label_types = ["sulci", "gyri", "any"]


for label_type, label_names in zip(label_types, label_lists):
    f = mlab.figure(size=(400, 400))

    brain = Brain(subject, hemi, 'inflated', background="white",
                  subjects_dir=subjects_dir, figure=f,
                  views="lateral")
    surf = brain.geo[hemi]
    for name, color in zip(label_names[:5], colors):
        ll = annot_dict_full[name + "-%s" % hemi]
        mlab.points3d(surf.x[ll.vertices],
                      surf.y[ll.vertices],
                      surf.z[ll.vertices],
                      color=tuple(color),
                      scale_factor=1)
    # brain.add_label(ll, color=color, borders=False, alpha=1.)
    mlab.savefig(labels_path + "labels-%s-%s.pdf" %
                 (label_type, hemi))
