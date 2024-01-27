import numpy as np
import pandas as pd
import scipy


def y_next(B, X, epsilon, c=None):
    res = B @ X + epsilon
    if c is not None:
        res += c
    return res


def X_next(theta, A, X, nu):
    res = theta - A @ theta + A @ X
    if isinstance(nu, (pd.Series, np.ndarray)) or nu:
        res += nu
    return res


def simulate(X_init, Q, H, B, theta, A, n_obs, seed=42, c=None):
    if Q is not None:
        seed_nu = np.random.RandomState((seed, 1))
        nu = scipy.stats.multivariate_normal(cov=Q, seed=seed_nu)
    else:
        nu = None
    if H is not None:
        seed_epsilon = np.random.RandomState((seed, 2))
        epsilon = scipy.stats.multivariate_normal(cov=H, seed=seed_epsilon)
    else:
        epsilon = None

    if X_init is None:
        X_init = theta
    xs = [X_init]
    ys = []

    for i in range(n_obs):
        ys.append(y_next(B, xs[-1], pd.Series(epsilon.rvs() if epsilon is not None else 0, index=B.index),
                         c=c))
        xs.append(X_next(theta, A, xs[-1], nu.rvs() if nu is not None else 0))

    xs = xs[:-1]

    return pd.DataFrame(xs), pd.DataFrame(ys)
