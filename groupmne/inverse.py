"""
Multi-subject inverse problem.

This module fits multi-subject inverse problem models on real or simulated
M-EEG data.
"""


import numpy as np

import mne

from mutar import DirtyModel, IndLasso, ReMTW, MTW, GroupLasso, IndRewLasso

from . import utils
from .group_model import _group_filtering
from .solvers import _gl_wrapper
from .utils import compute_ground_metric


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


def _prepare_inv_data(fwds, src_ref, noise_covs, ch_type="grad"):
    """Compute aligned gain matrices of the group of subjects.

    Parameters
    ----------
    fwds : list
        The forward operators computed on the morphed source
        space `src_ref`.
    src_ref : instance of SourceSpace instance
        Reference source model.
    noise_covs : list of instances of Covariance
        The noise covariances, one element for each subject.
    ch_type : str (default "grad")
        Type of channels used for source reconstruction. Can be one
        of ("mag", "grad", "eeg"). Using more than one type of channels is not
        yet supported.

    Returns
    -------
    gains: ndarray, shape (n_subjects, n_channels, n_sources)
        The gain matrices.
    M: ndarray, shape (n_subjects, n_channels, n_times)
        M-EEG data.
    group_info: dict
        Group information (channels, alignments maps across subjects)

    """
    if len(fwds) != len(noise_covs):
        raise ValueError("""The length of `fwds` and `noise_cov_s`
                         do not match.""")
    gains, group_info = _group_filtering(fwds, src_ref, noise_covs=noise_covs)
    info = fwds[0]["info"]
    ch_names = group_info["ch_names"]
    sel = utils._filter_channels(info, ch_names, ch_type)
    group_info["ch_filter"] = True
    group_info["sel"] = sel
    gains = gains[:, sel, :]
    whiteners = []
    for i, (noise_cov, gain) \
            in enumerate(zip(noise_covs, gains)):
        whitener, _ = mne.cov.compute_whitener(noise_cov, info,
                                               sel, pca=False)
        whiteners.append(whitener)

    group_info["hemi"] = "both"
    return gains, group_info, whiteners


class InverseOperator(object):
    """InverseOperator class to represent the inverse problem data.

    Parameters
    ----------

    fwds: list of `mne.Forward`.
        Forward soluton of each subject.
    noise_covs: list of `mne.Covariance`
        Noise covariance of each subject.
    src_ref: instance of `mne.SourceSpaces`.
        Source space to be used as a reference for the group study. `src_ref`
        will be morphed to the individual surface of each subject to have
        aligned sources.
    ch_type: str.
        Type of channels to use. Must be one of "grad", "mag" or "eeg".
    depth: float.
        How to weight (or normalize) the forward using a depth prior.
        If float (default 0.8), it acts as the depth weighting exponent (exp)
        to use, which must be between 0 and 1. None is equivalent to 0,
        meaning no depth weighting is performed.
    loose: float.
        Value that weights the source variances of the dipole components that
        are parallel (tangential) to the cortical surface. If loose is 0 then
        the solution is computed with fixed orientation (default).

    Attributes
    ----------
    stcs_: list of `mne.SourceEstimates`.
        Source estimates.
    """

    def __init__(self, fwds, noise_covs, src_ref, ch_type="grad", depth=0.9,
                 verbose=True):

        self.fwds = fwds
        self.noise_covs = noise_covs
        self.src_ref = src_ref
        self.ch_type = ch_type
        self.depth = depth
        self.verbose = verbose
        self.method = None
        self.solver_kwargs = dict()
        self._fitted = False
        self._compute_group_model()

    def _compute_group_model(self):
        """Compute aligned forwards and whitener operators."""
        self.n_subjects = len(self.fwds)
        gains, group_info, whiteners = \
            _prepare_inv_data(self.fwds, self.noise_covs, self.src_ref,
                              self.ch_type)
        self._gains = gains
        self._group_info = group_info
        self._whiteners = whiteners

    def _check_evokeds(self, evokeds, tmin, tmax):
        self.tmin, self.tmax = tmin, tmax
        if self.n_subjects != len(evokeds):
            raise ValueError("The number of evokeds is not equal to the "
                             "number of subjects. Expected %d, got %d."
                             % (self.n_subjects, len(evokeds)))
        times = np.array([ev.times for ev in evokeds])
        all_equal = np.diff(times, axis=0)
        if (all_equal != 0.).any():
            raise ValueError("The Evokeds have different time coordinates.")

        self._evokeds = []
        for i in range(self.n_subjects):
            ev = evokeds[i].copy()
            ev.crop(self.tmin, self.tmax)
            self._evokeds.append(ev)

    def _whiten_data(self):
        """Whiten the evokeds."""
        meeg_w = []
        gains_w = []
        sel = self._group_info["sel"]
        for i in range(self.n_subjects):
            ev = self._evokeds[i]
            W = self._whiteners[i]
            gains_w.append((ev.nave) ** 0.5 * W.dot(self._gains[i]))
            meeg_w.append((ev.nave) ** 0.5 * W.dot(ev.data[sel]))

        meeg_data = np.stack(meeg_w, axis=0)
        gains_w = np.stack(gains_w, axis=0)
        return meeg_data, gains_w

    def _scale_data(self):
        weights = np.linalg.norm(self._gains_w, axis=1) ** self.depth
        gains_scaled = self._gains_w / weights[:, None, :]
        return gains_scaled, weights

    def _check_solver(self, method, time_independent, **solver_kwargs):
        self.time_independent = time_independent
        self.method = method
        if method not in ["grouplasso", "dirty", "mtw", "remtw", "lasso",
                          "relasso"]:
            raise ValueError("%s is not a valid method. `method` must be one "
                             "of 'grouplasso', 'dirty', 'mtw', 'remtw',"
                             " 'lasso', 'relasso'" % method)
        if method != "grouplasso" and not time_independent:
            raise ValueError("%s is not feasible as a time dependent method."
                             "Use Group Lasso for an L2 over the time axis or"
                             " set `time_independent` to `True`." % method)

    def _check_solver_params(self, method, solver_kwargs, gains_scaled, meeg):
        n_subjects, n_channels, n_times = meeg.shape
        n_features = gains_scaled.shape[-1]

        # alpha is necessary for all models
        if "alpha" not in solver_kwargs.keys():
            raise ValueError("The method %s requires the `alpha`"
                             " hyperparameter."
                             "See the documentation of %s for further detail."
                             % (method, _method_to_str(method)))

        # beta is necessary for dirty and ot models
        if method not in ["lasso", "grouplasso", "relasso"]:
            if "beta" not in solver_kwargs.keys():
                raise ValueError("The method %s requires the `beta`"
                                 " hyperparameter."
                                 "See the documentation of %s for further "
                                 "detail."
                                 % (method, _method_to_str(method)))

        # ground metric and ot hyperparameters for ot models
        if method in ["mtw", "remtw"]:
            if "M" not in solver_kwargs.keys():
                print("Computing OT ground metric ...")
                M = compute_ground_metric(self.src_ref, self._group_info)
                solver_kwargs["M"] = M
            else:
                M = solver_kwargs["M"]
            if len(M) != n_features or len(M.T) != n_features:
                raise ValueError("The ground metric M must be an array"
                                 "(n_features, n_features); got (%s, %s)"
                                 % M.shape)
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
            # take L2 over subjects and max over time and space
            # alphamax (n_times)
            alphamax = np.linalg.norm(xty, axis=0).max() / n_channels
            solver_kwargs["alpha"] *= alphamax

        # rescale l1 norm penalty
        elif method in ["dirty", "mtw", "remtw", "lasso", "relasso"]:
            betamax = abs(xty).max() / n_channels
            if method in ["lasso", "relasso"]:
                alpha_ = betamax * np.ones(n_subjects)
                solver_kwargs["alpha"] *= alpha_
            else:
                solver_kwargs["beta"] *= betamax
        return solver_kwargs

    def solve(self, evokeds, method="grouplasso", time_independent=True,
              tmin=None, tmax=None, verbose=True, **solver_kwargs):
        """Solves the inverse problem jointly.

        Parameters
        ----------

        evokeds: list of mne.Evokeds instances.
        method: str.
            Method to use to compute the source estimates. Must be one of
            ("grouplasso", "dirty", "mtw", "remtw", "lasso", "relasso").
        time_independent: bool, (optional, default True)
            If True, each time point is solved independently.
        tmin: float.
            Starting time point of the selected evoked response window.
        tmax: flat
            Ending time point of the selected evoked response window.
        solver_kwargs : dict
            additional arguments passed to the solver.
        """
        self.tmin, self.tmax = tmin, tmax
        self._check_solver(method, time_independent, solver_kwargs)

        gains, meeg_data = self._whiten_data()
        gains_scaled, weights = self._rescale_data(gains)

        self._check_evokeds(evokeds, tmin, tmax)
        self.solver_kwargs = self._check_solver_params(method, solver_kwargs,
                                                       gains_scaled, meeg_data)
        stc_data, log = _apply_solver(gains_scaled, self._meeg, method,
                                      time_independent, verbose=verbose,
                                      **self.solver_kwargs)
        self._stc_data = stc_data
        # re-scale coefs and change units to nAm
        stc_data = np.array(stc_data) * 1e9 / weights.T[None, :, :]
        stcs = _coefs_to_stcs(stc_data, self._group_info)
        self.stcs_ = stcs
        self.log_ = log
        return stcs


def _apply_solver(gains_scaled, meeg, method, time_independent, verbose,
                  **solver_kwargs):
    """Apply time independent solver."""
    n_subjects, n_channels, n_times = meeg.shape

    if not time_independent:
        meeg = np.swapaxes(meeg, 1, 2).reshape(-1, n_channels)
        gains_scaled = np.tile(gains_scaled, (n_times, 1, 1))
        coefs, residuals, loss, dg = _gl_wrapper(gains_scaled, meeg,
                                                 **solver_kwargs)
        coefs = coefs.reshape(-1, n_subjects, n_times).T
        coefs = np.swapaxes(coefs, 1, 2)
    else:
        solver = _method_to_solver(method)
        n_features = gains_scaled.shape[-1]
        n_subjects, n_channels, n_times = meeg.shape
        coefs = np.empty((n_times, n_features, n_subjects))
    # residuals = np.empty((n_times, n_subjects, n_channels))

    for t in range(n_times):
        if verbose:
            print("Solving for time point {} / {}".format(t + 1, n_times))
        estim = solver(fit_intercept=False, normalize=False, **solver_kwargs)
        estim.fit(gains_scaled, meeg[:, :, t])
        coefs[t] = estim.coef_

    log = dict()

    return coefs, log
