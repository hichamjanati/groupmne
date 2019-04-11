import os
import os.path as op
import mne
import numpy as np
import pickle

from brain_utils import get_multiple_indices
import config as cfg


dataset_name = "camcan"
save_dir = "~/data/%s/" % dataset_name
save_dir = op.expanduser(save_dir)

subjects_dir = cfg.get_subjects_dir(dataset_name)
os.environ['SUBJECTS_DIR'] = subjects_dir
labels_path = subjects_dir + "fsaverage/label/"

fname = "aparca2009s"
labels_fname = subjects_dir + "fsaverage/label/%s.pkl" % fname

labels_raw = mne.read_labels_from_annot("fsaverage", "aparc.a2009s",
                                        subjects_dir=subjects_dir)
labels_dict = {}
annot_dict = {}
for l in labels_raw:
    labels_dict[l.name] = l
    ll = l.morph(subject_to="fsaverage", grade=4)
    annot_dict[ll.name] = ll

labels_dict_fname = save_dir + "label/%s.pkl" % fname
with open(labels_dict_fname, "wb") as ff:
    pickle.dump(labels_dict, ff)

fname = "aparca2009s"
labels_fname = subjects_dir + "fsaverage/label/%s.pkl" % fname

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
resolution = 4
spacing = "ico%d" % resolution
for hemi in ["lh", "rh"]:
    hemi_bool = int(hemi == "rh")
    subject = "fsaverage"
    vert_fname = save_dir + "vertno/%s-%s-filtered-%s-vrt.npy" %\
        (subject, spacing, hemi)
    vertno = np.load(vert_fname)
    colors = [(1, 0, 0), (0, 0, 1), (0, 1, 1), (1, 1, 0),
              (1, 0, 1), (0.5, 0.2, 0.8), (0.2, 0.9, 0.3),
              (1., 0.5, 0.2), (0.5, 1, 0.9), (1, 0.5, 0.2)]
    for label_type, label_names in zip(label_types, label_lists):
        n_labels = len(label_names)
        labels = np.zeros((n_labels, len(vertno)))
        for i, name in enumerate(label_names):
            ll = annot_dict[name + "-%s" % hemi]
            indices = get_multiple_indices(ll.vertices, vertno)
            labels[i, indices] = 1
        labels_fname = save_dir + "label/labels-%s-%s.npy" % \
            (label_type, hemi)
        np.save(labels_fname, labels)
