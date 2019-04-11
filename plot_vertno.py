from surfer import Brain
from mayavi import mlab

import os
import os.path as op
import numpy as np

dataset = "camcan"
meg_path = "~/Dropbox/neuro_transport/code/mtw_experiments/meg/%s/" % dataset
meg_path = op.expanduser(meg_path)

subjects_dir = meg_path + "subjects/"
os.environ['SUBJECTS_DIR'] = subjects_dir
vertno_dir = meg_path + "vertno/"

subjects = ["fsaverage", "CC110033"]
hemi = "lh"
sp = "ico4"
sources = np.arange(15)
for sub in subjects:
    brain = Brain(sub, hemi, "inflated", background="white")
    surf = brain.geo[hemi]
    vertno = np.load(vertno_dir + "%s-%s-filtered-%s-vrt.npy" %
                     (sub, sp, hemi))
    indx = vertno[sources]
    mlab.points3d(surf.x[indx], surf.y[indx], surf.z[indx],
                  color=(1, 0, 0), scale_factor=5)
    mlab.savefig(vertno_dir + "%s-%s.png" % (sub, hemi))
