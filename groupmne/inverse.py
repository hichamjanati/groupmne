"""
Multi-subject inverse problem.

This module fits multi-subject inverse problem models on real or simulated
M-EEG data.
"""


import numpy as np
# from joblib import Parallel, delayed
from mutar import DirtyModel, IndLasso, ReMTW, MTW, GroupLasso, IndRewLasso

from . import utils
from .solvers import _gl_wrapper


def _coefs_to_stcs(coefs, group_info):
    stcs = []
    hemi = group_info["hemi"]
    vertices_lh = group_info["vertno_lh"]
    vertices_rh = group_info["vertno_rh"]
    subjects = group_info["subjects"]
    for i, (v_l, v_r, subject) in enumerate(zip(vertices_lh, vertices_rh,
                                                subjects)):
        if hemi == "lh":
            v = [v_l, []]
        elif hemi == "rh":
            v = [[], v_r]
        else:
            v = [v_l, v_r]
        stc = utils._make_stc(coefs[:, :, i].T, v, tmin=group_info["tmin"],
                              tstep=group_info["tstep"], subject=subject)
        stcs.append(stc)
    return stcs


def _method_to_estimator(method):
    if method == "lasso":
        return IndLasso
    elif method == "relasso":
        return IndRewLasso
    elif method == "mtw":
        return MTW
    elif method == "remtw":
        return ReMTW
    elif method == "dirty":
        return DirtyModel
    elif method == "grouplasso":
        return GroupLasso
    else:
        raise ValueError("Method %s not recognized." % method)


def _method_to_str(method):
    if method == "lasso":
        return "mutar.IndLasso"
    if method == "relasso":
        return "mutar.IndRewLasso"
    elif method == "mtw":
        return "mutar.MTW"
    elif method == "remtw":
        return "mutar.ReMTW"
    elif method == "dirty":
        return "mutar.DirtyModel"
    elif method == "grouplasso":
        return "mutar.GroupLasso"
    else:
        raise ValueError("Method %s not recognized." % method)


def _check_estimator_params(method, estimator_kwargs, gains_scaled, meeg):
    n_subjects, n_channels, n_times = meeg.shape
    n_features = gains_scaled.shape[-1]

    # alpha is necessary for all models
    if "alpha" not in estimator_kwargs.keys():
        raise ValueError("The method %s requires the `alpha` hyperparameter."
                         "See the documentation of %s for further detail."
                         % (method, _method_to_str(method)))

    # beta is necessary for dirty and ot models
    if method not in ["lasso", "grouplasso", "relasso"]:
        if "beta" not in estimator_kwargs.keys():
            raise ValueError("The method %s requires the `beta`"
                             " hyperparameter."
                             "See the documentation of %s for further detail."
                             % (method, _method_to_str(method)))

    # ground metric and ot hyperparameters for ot models
    if method in ["mtw", "remtw"]:
        if "M" not in estimator_kwargs.keys():
            raise ValueError("The method %s requires the OT ground metric `M`"
                             " corresponding to the geodesic distance matrix "
                             "over the cortical mantle. "
                             "See the documentation of %s for further detail."
                             % (method, _method_to_str(method)))
        else:
            M = estimator_kwargs["M"]
            if len(M) != n_features or len(M.T) != n_features:
                raise ValueError("The ground metric M must be an array"
                                 "(n_features, n_features); got (%s, %s)"
                                 % M.shape)
            if M.min() < 0.:
                raise ValueError("The ground metric M must be non-negative"
                                 "got M.min() = %s"
                                 % M.min())
        if "gamma" not in estimator_kwargs.keys():
            gamma = estimator_kwargs["M"].max()
            estimator_kwargs["gamma"] = gamma
        if "epsilon" not in estimator_kwargs.keys():
            epsilon = 10. / n_features
            estimator_kwargs["epsilon"] = epsilon

    xty = np.array([g.T.dot(m) for g, m in zip(gains_scaled, meeg)])

    # rescale l12 norm penalty
    if method in ["grouplasso", "dirty"]:
        # take L2 over subjects and max over time and space
        # alphamax (n_times)
        alphamax = np.linalg.norm(xty, axis=0).max() / n_channels
        estimator_kwargs["alpha"] *= alphamax

    # rescale l1 norm penalty
    elif method in ["dirty", "mtw", "remtw", "lasso", "relasso"]:
        betamax = abs(xty).max() / n_channels
        if method in ["lasso", "relasso"]:
            alpha_ = betamax * np.ones(n_subjects)
            estimator_kwargs["alpha"] *= alpha_
        else:
            estimator_kwargs["beta"] *= betamax
    return estimator_kwargs


def compute_time_dependent_gl(gains, meeg, group_info, depth=0.9,
                              return_stc=True, **estimator_kwargs):
    """Solves a Group Lasso over subjects, space and time."""
    n_subjects, n_channels, n_times = meeg.shape
    norms = np.linalg.norm(gains, axis=1) ** depth
    gains_scaled = gains / norms[:, None, :]
    estimator_kwargs = _check_estimator_params("grouplasso", estimator_kwargs,
                                               gains_scaled, meeg)
    # estimator_kwargs["alpha"] = estimator_kwargs["alpha"].max()
    meeg = np.swapaxes(meeg, 1, 2).reshape(-1, n_channels)
    gains_scaled = np.tile(gains, (n_times, 1, 1))
    coefs, residuals, loss, dg = _gl_wrapper(gains_scaled, meeg,
                                             **estimator_kwargs)
    coefs = coefs.reshape(-1, n_subjects, n_times).T
    coefs = np.swapaxes(coefs, 1, 2)

    # re-normalize coefs and change units to nAm
    coefs = np.array(coefs) * 1e9 / norms.T[None, :, :]
    log = dict(residuals=residuals, loss=loss, dg=dg)
    if return_stc:
        stcs = _coefs_to_stcs(coefs, group_info)
        return stcs, log

    return coefs, log


def compute_group_inverse(gains, meeg, group_info, method="grouplasso",
                          depth=0.9, return_stc=True,
                          time_independent=True, verbose=True,
                          **estimator_kwargs):
    """Solves the joint inverse problem for source localization.

    Parameters
    ----------
    gains: ndarray, shape (n_subjects, n_channels, n_sources).
        Forward data, returned by
        `group_model.compute_gains` or `group_model.compute_inv_data`.
    meeg : ndarray, shape (n_subjects, n_channels, n_times)
        M-EEG data
    group_info : dict
        The measurement info.
    method : str
        Inverse problem model to use. For now, only "grouplasso" is
        supported. The group-lasso solver promotes source estimates with
        overlapping active vertices across subjects. Each time point is
        treated independently.
    depth : float in (0, 1)
        Depth weighting. If 1, no normalization is done.
    return_stc : bool, (optional, default True)
        If true, source estimates are returned as stc objects, array otherwise.
    time_independent : bool, (optional, default false)
        If True, each time point is solved independently. By default,
        the group lasso is applied on the time and subjects axes.
    estimator_kwargs : dict
        additional arguments passed to the solver.

    Returns
    -------
    estimates: instance of SourceEstimates | array
        The estimated sources as an array if `return_stc` is False.
    log : dict
        Some info about the convergence.

    """
    if method not in ["grouplasso", "dirty", "mtw", "remtw", "lasso",
                      "relasso"]:
        raise ValueError("%s is not a valid method. `method` must be one of "
                         "'grouplasso', 'dirty', 'mtw', 'remtw', 'lasso', "
                         "'relasso'" % method)
    if method != "grouplasso" and not time_independent:
        raise ValueError("%s is not feasible as a time dependent method."
                         "Use Group Lasso for an L2 over the time axis or"
                         " set `time_independent` to `True`." % method)
    if method == "grouplasso" and not time_independent:
        return compute_time_dependent_gl(gains, meeg, group_info, depth,
                                         return_stc, **estimator_kwargs)
    else:
        estimator = _method_to_estimator(method)
        n_features = gains.shape[-1]
        n_subjects, n_channels, n_times = meeg.shape
        norms = np.linalg.norm(gains, axis=1) ** depth
        gains_scaled = gains / norms[:, None, :]
        estimator_kwargs = _check_estimator_params(method, estimator_kwargs,
                                                   gains_scaled, meeg)
        coefs = np.empty((n_times, n_features, n_subjects))
        # residuals = np.empty((n_times, n_subjects, n_channels))

    for t in range(n_times):
        if verbose:
            print("Solving for time point {} / {}".format(t + 1, n_times))
        estim = estimator(fit_intercept=False,
                          normalize=False, **estimator_kwargs)
        estim.fit(gains_scaled, meeg[:, :, t])
        coefs[t] = estim.coef_
        # residuals[:, :, t] = estim.residuals_
    # re-normalize coefs and change units to nAm
    coefs = np.array(coefs) * 1e9 / norms.T[None, :, :]
    log = dict()
    if return_stc:
        stcs = _coefs_to_stcs(coefs, group_info)
        return stcs, log

    return coefs, log
