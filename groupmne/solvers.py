"""Solvers for multitask group-stl with a simplex constraint."""
import warnings
import numpy as np
import numba as nb
from numba import (jit, float64, int64, boolean)


@jit(float64[:](float64[::1, :, :]), nopython=True, cache=True)
def _lipschitz(X):
    """Compute lipschitz constants."""
    T, n, p = X.shape
    L = np.zeros(p)
    for j in range(p):
        maxl = 0.
        for k in range(T):
            li = 0.
            for i in range(n):
                li = li + X[k, i, j] ** 2
            if li > maxl:
                maxl = li
        L[j] = maxl
    return L


@jit(float64(float64[::1, :, :], float64[::1, :], float64[::1, :],
             float64[::1, :], float64),
     nopython=True, cache=True)
def _dualgap(X, theta, R, y, alpha):
    """Compute dual gap for multi task group lasso."""
    n_samples, n_tasks = R.T.shape
    n_features = X.shape[-1]
    dualnorm = 0.
    xtr = np.zeros_like(theta)
    for k in range(n_tasks):
        xtr[:, k] = X[k].T.dot(R[k])
    for j in range(n_features):
        dn = np.linalg.norm(xtr[j])
        if dn > dualnorm:
            dualnorm = dn
    R_norm = np.linalg.norm(R)
    if dualnorm < alpha:
        dg = R_norm ** 2
        const = 1.
    else:
        const = alpha / dualnorm
        A_norm = R_norm * const
        dg = 0.5 * (R_norm ** 2 + A_norm ** 2)
    dg += - const * (R * y).sum()
    for j in range(n_features):
        dg += alpha * np.linalg.norm(theta[j])
    return dg


@jit(float64(float64[::1, :], float64[::1, :], float64[::1, :], float64),
     nopython=True, cache=True)
def _mtlobjective(theta, R, y, alpha):
    """Compute objective function for multi task group lasso."""
    n_samples, n_tasks = R.T.shape
    n_features = theta.shape[0]
    obj = 0.
    for t in range(n_tasks):
        for n in range(n_samples):
            obj += R[t, n] ** 2
    obj *= 0.5
    for j in range(n_features):
        obj += alpha * np.linalg.norm(theta[j])
    return obj


output_type = nb.types.Tuple((float64[::1, :], float64[::1, :],
                              float64[:], float64, int64))


@jit(output_type(float64[::1, :, :], float64[::1, :],
                 float64[::1, :], float64[::1, :], float64, int64,
                 float64, boolean, boolean),
     nopython=True, cache=True)
def _gl_solver(X, y, theta, R, alpha, maxiter, tol, verbose, compute_obj):
    """Solve Multi-task group lasso."""
    n_tasks = len(X)
    n_samples, n_features = X[0].shape
    Ls = _lipschitz(X)
    dg_tol = tol * np.linalg.norm(y) ** 2
    loss = []
    flag = 0
    for i in range(maxiter):
        w_max = 0.0
        d_w_max = 0.0
        for j in range(n_features):
            # compute residual
            if Ls[j] == 0.:
                continue
            tmp = np.zeros(n_tasks)
            for t in range(n_tasks):
                for n in range(n_samples):
                    tmp[t] += X[t, n, j] * R[t, n]
            tmp /= Ls[j]
            tmp += theta[j]
            normtmp = np.linalg.norm(tmp)

            # l2 thresholding
            threshold = 0.
            if normtmp:
                threshold = max(1. - alpha / (Ls[j] * normtmp), 0.)
            tmp *= threshold
            if theta[j].any():
                for t in range(n_tasks):
                    R[t] += X[t, :, j] * theta[j, t]
            d_w_j = np.abs(theta[j] - tmp).max()
            d_w_max = max(d_w_max, d_w_j)
            w_max = max(w_max, np.abs(tmp).max())
            theta[j] = tmp

            if theta[j].any():
                for t in range(n_tasks):
                    R[t] -= X[t, :, j] * theta[j, t]

        if compute_obj:
            obj = _mtlobjective(theta, R, y, alpha)
            loss.append(obj)
        if (w_max == 0.0 or d_w_max / w_max < tol or
                i == maxiter - 1):
            dg = _dualgap(X, theta, R, y, alpha)
            if verbose:
                print("it:", i, "- duality gap: ", dg)
            if dg < dg_tol:
                break
    if i == maxiter - 1:
        flag = 1
    loss = np.array(loss)
    return theta, R, loss, dg, flag


def _gl_wrapper(X, y, alpha=0.1, maxiter=2000, tol=1e-5, verbose=False,
                computeobj=False):
    """Group lasso solver wrapper."""
    X = np.asfortranarray(X)
    y = np.asfortranarray(y)

    R = y.copy()
    n_tasks = len(X)
    n_samples, n_features = X[0].shape
    coefs0 = np.zeros((n_features, n_tasks))
    coefs0 = np.asfortranarray(coefs0)
    alpha_ = alpha * n_samples

    R = np.asfortranarray(R)
    theta, R, loss, dg, flag = _gl_solver(X, y, coefs0, R, alpha_,
                                          maxiter, tol, verbose, computeobj)
    theta = np.ascontiguousarray(theta)
    if flag:
        warnings.warn("Did not reach convergence threshold. Stopped with" +
                      "a duality gap = %f" % dg)
    return theta, R, loss, dg
