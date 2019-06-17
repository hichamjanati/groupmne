
|Travis|_ |AppVeyor|_ |Codecov|_ |CircleCI|_ |ReadTheDocs|_

.. |Travis| image:: https://travis-ci.com/hichamjanati/groupmne.svg?branch=master
.. _Travis: https://travis-ci.com/hichamjanati/groupmne

.. |AppVeyor| image:: https://ci.appveyor.com/api/projects/status/l7g6vywwwuyha49l?svg=true
.. _AppVeyor: https://ci.appveyor.com/project/hichamjanati/groupmne

.. |Codecov| image:: https://codecov.io/gh/hichamjanati/groupmne/branch/master/graph/badge.svg
.. _Codecov: https://codecov.io/gh/hichamjanati/groupmne

.. |CircleCI| image:: https://circleci.com/gh/hichamjanati/groupmne.svg?style=svg
.. _CircleCI: https://circleci.com/gh/hichamjanati/groupmne/tree/master

.. |ReadTheDocs| image:: https://readthedocs.org/projects/groupmne/badge/?version=latest
.. _ReadTheDocs: https://groupmne.readthedocs.io/en/latest/



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

For examples, see `the groupmne webpage <https://groupmne.readthedocs.io/en/latest/>`_.

Contact:
--------
Please contact hicham.janati@inria.fr for any bug encountered / any further information.
