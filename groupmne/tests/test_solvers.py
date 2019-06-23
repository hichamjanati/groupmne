import numpy as np
from groupmne.solvers import gl_wrapper


def test_solver_gl():
    seed = 42
    rnd = np.random.RandomState(seed)
    n_features = 10
    n_samples = 5
    n_subjects = 2
    X = rnd.randn(n_subjects, n_samples, n_features)
    y = rnd.randn(n_subjects, n_samples)
    xty = np.array([x.T.dot(yi) for x, yi in zip(X, y)])
    alphamax = np.linalg.norm(xty, axis=0).max()
    tol = 1e-4
    tol_dg = tol * np.linalg.norm(y) ** 2

    a = 0.1
    alpha = a * alphamax
    theta, R, loss, dg = gl_wrapper(X, y, alpha=alpha, maxiter=2000,
                                    tol=tol_dg, verbose=False,
                                    computeobj=True)
    assert np.diff(loss).max() <= 0.
    assert (theta.any(axis=1) == theta.all(axis=1)).all()

    a = 1.1
    alpha = a * alphamax
    theta, R, loss, dg = gl_wrapper(X, y, alpha=alpha, maxiter=2000,
                                    tol=tol_dg, verbose=False,
                                    computeobj=True)
    assert abs(theta).max() == 0.
