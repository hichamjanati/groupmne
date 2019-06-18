.. project-template documentation master file, created by
   sphinx-quickstart on Mon Jan 18 14:44:12 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


multi-subject source localization with MNE
==========================================
*version 0.0.1*

Description:
------------

This small package helps perform EEG and MEG source localization jointly on many
subjects. The inverse problems of all subjects are solved congruently using a
binding regularization.


Installation
------------

On a working `mne-python <https://github.com/mne-tools/mne-python>`_ environment:

::

   git clone https://github.com/hichamjanati/groupmne
   cd groupmne
   python setup.py develop

Otherwise, we recommend creating this minimal `conda env <https://raw.githubusercontent.com/hichamjanati/groupmne/master/environment.yml>`_

::

   conda env create --file environment.yml
   conda activate groupmne-env
   git clone https://github.com/hichamjanati/groupmne
   cd groupmne
   python setup.py develop


Documentation
-------------

For examples, see the `User Guide <auto_examples/index.html>`_.

Contact:
--------
Please contact hicham.janati@inria.fr for any bug encountered / any further information.
