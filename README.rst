
|Travis|_ |AppVeyor|_ |Codecov|_

.. |Travis| image:: https://travis-ci.com/hichamjanati/groupmne.svg?branch=master
.. _Travis: https://travis-ci.com/hichamjanati/groupmne

.. |AppVeyor| image:: https://ci.appveyor.com/api/projects/status/l7g6vywwwuyha49l?svg=true
.. _AppVeyor: https://ci.appveyor.com/project/hichamjanati/groupmne

.. |Codecov| image:: https://codecov.io/gh/hichamjanati/groupmne/branch/master/graph/badge.svg
.. _Codecov: https://codecov.io/gh/hichamjanati/groupmne


multi-subject source localization with MNE
==========================================

*version 0.0.1*

Description:
------------

GroupMNE provides off-the-shelf functions to perform EEG and MEG source
localization jointly on many subjects. The inverse problems of all subjects are
solved congruently using a binding regularization.


Installation
------------

On a working `mne-python <https://mne.tools/stable/install/mne_python.html#installing-python>`_ environment:

.. code-block:: bash

    pip install -U mutar
    pip install -U https://api.github.com/repos/hichamjanati/groupmne/zipball/master


GroupMNE uses under the hood the `MUTAR package <https://hichamjanati.github.io/mutar/>`_
to solve the underlying multi-task optimization problems.

Documentation
-------------

For examples, see `the groupmne webpage <https://hichamjanati.github.io/groupmne/>`_.
