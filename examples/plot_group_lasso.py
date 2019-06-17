"""
Multi-subject joint source localization with multi-task models
==============================================================

The aim of this tutorial is to show how to leverage functional similarity
across subjects to improve source localization. For that purpose we use the
the high frequency SEF MEG dataset of (Nurminen et al., 2017) which provides
MEG and MRI data for two subjects.
"""

# Author: Hicham Janati (hicham.janati@inria.fr)
#
# License: BSD (3-clause)

import mne
import os
import os.path as op
from mne.parallel import parallel_func
from mne.datasets import hf_sef
from matplotlib import pyplot as plt

from groupmne import group_model
from groupmne.inverse import compute_group_inverse

print('hello world')
