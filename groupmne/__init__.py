from . import group_model, inverse
from .group_model import (compute_gains, compute_inv_data, get_src_reference,
                          compute_fwd)
from .inverse import compute_group_inverse


__all__ = ["group_model", "inverse", "compute_gains", "compute_inv_data",
           "get_src_reference", "compute_group_inverse", "compute_fwd"]
