"""Multi-subject source localization with MNE."""

from . import group_model, inverse, utils
from .group_model import compute_gains, get_src_reference, compute_fwd
from .inverse import InverseOperator
from ._version import __version__


__all__ = ["group_model", "inverse", "compute_gains",
           "get_src_reference", "InverseOperator", "compute_fwd",
           "__version__", "utils"]
