"""
Multi-subject inverse problem.

This module fits multi-subject inverse problem models on real or simulated
M-EEG data.
"""


import numpy as np

import mne

from mutar import DirtyModel, IndLasso, ReMTW, MTW, GroupLasso, IndRewLasso

from . import utils
from .solvers import _gl_wrapper
from .utils import _compute_ground_metric


def _coefs_to_stcs(coefs, group_info, tmin, tstep):
    stcs = []
    vertices_lh = group_info["vertno_lh"]
    vertices_rh = group_info["vertno_rh"]
    subjects = group_info["subjects"]
    for i, (v_l, v_r, subject) in enumerate(zip(vertices_lh, vertices_rh,
                                                subjects)):
        v = [v_l, v_r]
        stc = utils._make_stc(coefs[:, :, i].T, v, tmin=tmin,
                              tstep=tstep, subject=subject)
        stcs.append(stc)
    return stcs


def _method_to_solver(method):
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


def _check_evokeds(evokeds):
    times = evokeds[0].times
    for ii, evoked in enumerate(evokeds[1:]):
        current_times = evoked.times
        if times.shape != current_times.shape:
            raise ValueError("Subject number %d has a times array with a "
                             "different length. Please provide evokeds data "
                             "with the same shape." % (ii + 1))
        if not (times == current_times).all():
            raise ValueError("Subject number %d has a times array with "
                             "different time coordinates. Please provide "
                             "evokeds data with the same times "
                             "array." % (ii + 1))
        times = current_times.copy()


def _get_common_sel(evokeds, noise_covs, fwds):
    selections = []
    for ev, cov, fwd in zip(evokeds, noise_covs, fwds):
        all_channels = fwd["sol"]["row_names"]
        ch_names = utils._get_channels(fwd, noise_cov=cov, evoked=ev)
        sel = utils._ch_names_to_sel(all_channels, ch_names)
        selections.append(sel)
    sel = set(selections[0]).intersection(*selections[1:])
    sel = list(sel)
    return sel


def _whiten_data(fwds, evokeds, noise_covs, depth):
    """Whiten the evokeds."""
    n_subjects = len(fwds)
    sel = _get_common_sel(evokeds, noise_covs, fwds)
    meeg_w = []
    gains_w = []
    for ii in range(n_subjects):
        ev = evokeds[ii]
        all_channels = list(np.array(fwds[ii]["sol"]["row_names"])[sel])
        W, _ = mne.cov.compute_whitener(noise_covs[ii], ev.info, all_channels,
                                        pca=False, verbose=False)
        W = W[sel, :][:, sel]
        gain = fwds[ii]["sol_group"]["data"][sel]
        gains_w.append((ev.nave) ** 0.5 * W.dot(gain))
        meeg_w.append((ev.nave) ** 0.5 * W.dot(ev.data[sel]))

    meeg_data = np.stack(meeg_w, axis=0)
    gains_w = np.stack(gains_w, axis=0)

    weights = np.linalg.norm(gains_w, axis=1) ** depth
    gains_w = gains_w / weights[:, None, :]
    return gains_w, meeg_data, weights


def _check_solver(method, spatiotemporal):
    if method not in ["grouplasso", "dirty", "mtw", "remtw", "lasso",
                      "relasso"]:
        raise ValueError("%s is not a valid method. `method` must be one "
                         "of 'grouplasso', 'dirty', 'mtw', 'remtw',"
                         " 'lasso', 'relasso'" % method)
    if method != "grouplasso" and spatiotemporal:
        raise ValueError("%s is not feasible as a time dependent method."
                         "Use Group Lasso for an L2 over the time axis or"
                         " set `spatiotemporal` to `True`." % method)


def _check_solver_params(fwds, method, solver_kwargs, gains_scaled, meeg,
                         spatiotemporal):
    n_subjects, n_channels, n_times = meeg.shape
    n_features = gains_scaled.shape[-1]

    # alpha is necessary for all models
    if "alpha" not in solver_kwargs.keys():
        solver_kwargs["alpha"] = 0.2
    # beta is necessary for dirty and ot models
    if method not in ["lasso", "grouplasso", "relasso"]:
        if "beta" not in solver_kwargs.keys():
            solver_kwargs["beta"] = 0.2
    # ground metric and ot hyperparameters for ot models
    if method in ["mtw", "remtw"]:
        if "M" not in solver_kwargs.keys():
            print("Computing OT ground metric ...")
            src_ref = fwds[0]["sol_group"]["src_ref"]
            _group_info = fwds[0]["sol_group"]["group_info"]
            M = _compute_ground_metric(src_ref, _group_info)
            solver_kwargs["M"] = M
        else:
            M = solver_kwargs["M"]
        if len(M) != n_features or len(M.T) != n_features:
            raise ValueError("The ground metric M must be an array"
                             "(%s, %s); got (%s, %s)"
                             % (n_features, n_features, *M.shape))
        if M.min() < 0.:
            raise ValueError("The ground metric M must be non-negative"
                             "got M.min() = %s"
                             % M.min())
        if "gamma" not in solver_kwargs.keys():
            gamma = solver_kwargs["M"].max()
            solver_kwargs["gamma"] = gamma
        if "epsilon" not in solver_kwargs.keys():
            epsilon = 10. / n_features
            solver_kwargs["epsilon"] = epsilon

    xty = np.array([g.T.dot(m) for g, m in zip(gains_scaled, meeg)])

    # rescale l12 norm penalty
    if method in ["grouplasso", "dirty"]:
        if not spatiotemporal:
            alphamax = np.linalg.norm(xty, axis=0).max() / n_channels
            solver_kwargs["alpha"] *= alphamax
    # rescale l1 norm penalty
    if method in ["dirty", "mtw", "remtw", "lasso", "relasso"]:
        betamax = abs(xty).max() / n_channels
        if method in ["lasso", "relasso"]:
            alpha_ = betamax * np.ones(n_subjects)
            solver_kwargs["alpha"] *= alpha_
        else:
            solver_kwargs["beta"] *= betamax
    return solver_kwargs


def _apply_solver(gains_scaled, meeg, method, spatiotemporal, verbose,
                  **solver_kwargs):
    """Apply time independent solver."""
    n_subjects, n_channels, n_times = meeg.shape

    if spatiotemporal:
        meeg = np.swapaxes(meeg, 1, 2).reshape(-1, n_channels)
        gains_scaled = np.tile(gains_scaled, (n_times, 1, 1))
        gty = np.array([g.T.dot(m) for g, m in zip(gains_scaled, meeg)])
        alphamax = np.linalg.norm(gty, axis=0).max(axis=0)
        solver_kwargs["alpha"] *= alphamax / n_channels
        coefs, residuals, loss, dg = _gl_wrapper(gains_scaled, meeg,
                                                 **solver_kwargs)
        coefs = coefs.reshape(-1, n_subjects, n_times).T
        coefs = np.swapaxes(coefs, 1, 2)
        log = dict(dualgap=dg, loss=loss, residuals=residuals)
    else:
        solver = _method_to_solver(method)
        n_features = gains_scaled.shape[-1]
        n_subjects, n_channels, n_times = meeg.shape
        coefs = np.empty((n_times, n_features, n_subjects))

        for t in range(n_times):
            if verbose:
                print("Solving for time point {} / {}".format(t + 1, n_times))
            estim = solver(fit_intercept=False, normalize=False,
                           **solver_kwargs)
            estim.fit(gains_scaled, meeg[:, :, t])
            assert estim.coef_.shape == (n_features, n_subjects)
            coefs[t] = estim.coef_

        log = dict()

    return coefs, log


def _check_fwds(fwds):
    """Check whether fwds were prepared."""
    for fwd in fwds:
        if "sol_group" not in fwd.keys():
            raise ValueError("`groupmne.prepare_fwds` must be called before "
                             "to compute a group inverse.")


def compute_group_inverse(fwds, evokeds, noise_covs, method="grouplasso",
                          depth=0.8, spatiotemporal=False, verbose=True,
                          **solver_kwargs):
    """Compute inverse solution for a group of subjects.

    Parameters
    ----------
    fwds: list of `mne.Forward`.
        Forward soluton of each subject.
    evokeds: list of `mne.Evokeds`
        Evoked object of each subject.
    noise_covs: list of `mne.Covariance`
        Noise covariance of each subject.
    method: str
        Model used for the joint prior. Must be one of ('lasso', 'relasso',
        'grouplasso', 'dirty', 'mtw', 'remtw').
    depth: float.
        How to weight (or normalize) the forward using a depth prior.
        If float (default 0.8), it acts as the depth weighting exponent (exp)
        to use, which must be between 0 and 1. None is equivalent to 0,
        meaning no depth weighting is performed.
    spatiotemporal: boolean.
        If True, apply a spatiotemporal prior on the source estimates.
        Only for method = `grouplasso`.
    solvers_kwargs: additional keyword arguments passed to the solver.

    Returns
    -------
    stcs: list of `mne.SourceEstimates`.
        Source estimates.

    """
    if len(evokeds) != len(fwds):
        raise ValueError("The number of evokeds is not equal to the number "
                         "of forwards.")
    _check_solver(method, spatiotemporal)
    _check_evokeds(evokeds)
    _check_fwds(fwds)
    gains, meeg, weights = _whiten_data(fwds, evokeds, noise_covs, depth)
    # Check hyperparameters for all models and rescale them to 0-1
    solver_kwargs = _check_solver_params(fwds, method, solver_kwargs,
                                         gains, meeg,
                                         spatiotemporal)

    stc_data, log = _apply_solver(gains, meeg, method,
                                  spatiotemporal, verbose=verbose,
                                  **solver_kwargs)
    # re-scale coefs and change units to nAm
    stc_data = np.array(stc_data) * 1e9 / weights.T[None, :, :]
    tmin = evokeds[0].times[0]
    tstep = evokeds[0].times[1] - tmin
    stcs = _coefs_to_stcs(stc_data, fwds[0]["sol_group"]["group_info"],
                          tmin=tmin, tstep=tstep)
    return stcs
