"""Multi-subject source localization with MNE."""

from . import group_model, inverse, utils
from .group_model import compute_fwd, prepare_fwds
from .inverse import compute_group_inverse
from ._version import __version__


__all__ = ["group_model", "inverse",
           "compute_group_inverse", "compute_fwd",
           "prepare_fwds", "__version__", "utils"]
