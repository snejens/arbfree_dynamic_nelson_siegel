import numpy as np
import pandas as pd
from numba import jit

from .utils import numpyify, numpyify_single


@jit(nopython=True)
def Ft_numba(B, Pt, H):
    return B @ Pt @ B.T + H


@jit(nopython=True)
def xtt_numba(xt, Pt, B, vt, Finv):
    return xt + Pt @ B.T @ Finv @ vt


@jit(nopython=True)
def Ptt_numba(Pt, B, Finv):
    return Pt - Pt @ B.T @ Finv @ B @ Pt


@jit(nopython=True)
def xpred_numba(A, xtt, theta):
    return A @ xtt + theta - A @ theta


@jit(nopython=True)
def Ppred_numba(A, Ptt, Q):
    return A @ Ptt @ A.T + Q


@jit(nopython=True)
def x0_numba(A, theta):
    # res = np.linalg.solve(np.eye(3) - A, theta)
    return theta


@jit(nopython=True)
def P0_numba(A, Q):
    n = Q.shape[0]
    res = np.linalg.solve(np.eye(n ** 2) - np.kron(A, A), Q.reshape(-1)).reshape(n, n)
    return res


def Ft(B, Pt, H):
    return Ft_numba(*numpyify(B, Pt, H))


def xtt(xt, Pt, B, vt, Finv):
    return xtt_numba(*numpyify(xt, Pt, B, vt, Finv))


def Ptt(Pt, B, Finv):
    return Ptt_numba(*numpyify(Pt, B, Finv))


def xpred(A, xtt, theta):
    return xpred_numba(*numpyify(A, xtt, theta))


def Ppred(A, Ptt, Q):
    return Ppred_numba(*numpyify(A, Ptt, Q))


def x0(A, theta):
    return x0_numba(*numpyify(A, theta))


def P0(A, Q):
    return P0_numba(*numpyify(A, Q))


@jit(nopython=True)
def _kalman_filter_inner(synth_yields, theta, A, H, Q, B,
                         exclude_first_observations_for_loglikelihood=0,
                         c=None):  # , no_tqdm=False):
    # print(synth_yields, theta, A, H, Q)
    exclude_first_observations_for_loglikelihood = int(exclude_first_observations_for_loglikelihood)
    assert len(synth_yields) > exclude_first_observations_for_loglikelihood >= 0

    x = np.zeros((len(synth_yields) + 1, Q.shape[0]), dtype=A.dtype)
    x[0] = x0_numba(A, theta)
    P = np.zeros((len(synth_yields) + 1, *Q.shape), dtype=Q.dtype)
    P[0] = P0_numba(A, Q)
    F = np.zeros((len(synth_yields), *H.shape), dtype=H.dtype)
    v = np.zeros((len(synth_yields), B.shape[0]), dtype=B.dtype)
    loglikelihood_contribution = np.zeros(len(synth_yields) - exclude_first_observations_for_loglikelihood,
                                          dtype=B.dtype)

    # assert spectral_radius(A) < 1 + 1e-9
    assert np.max(H - H.T) < 1e-9
    assert np.max(Q - Q.T) < 1e-9
    assert np.all(np.linalg.eigvalsh(Q) > -1e-9)
    assert np.all(np.linalg.eigvalsh(H) > -1e-9)

    # it = synth_yields.iterrows()
    # if not no_tqdm:
    #    it = tqdm(it, total=len(synth_yields))
    # for t, y in it:
    for t, y in enumerate(synth_yields):
        xt, Pt = x[t], P[t]
        Ft_ = Ft_numba(B, Pt, H)
        vt_ = y - B @ xt
        if c is not None:
            vt_ -= c
        if exclude_first_observations_for_loglikelihood <= t:
            if not np.isfinite(Ft_).all():
                _debug_output(A, Ft_, H, Q, x)
                return None, -1, None, None, None, None, None
            signdetFt_, logdetFt_ = np.linalg.slogdet(Ft_)  # np.log(np.linalg.det(Ft_))
            if signdetFt_ <= 0:
                _debug_output(A, Ft_, H, Q, x)
                return None, -1, None, None, None, None, None
            if not np.isfinite(logdetFt_):
                _debug_output(A, Ft_, H, Q, x)
                return None, -1, None, None, None, None, None
            # Finv_times_vt_ = np.linalg.solve(Ft_, vt_)
        # try:
        Finv = np.linalg.inv(Ft_)
        # except np.linalg.LinAlgError:
        #    _debug_output(A, Ft_, H, Q, x)
        #    return None, -1, None, None, None, None, None
        if exclude_first_observations_for_loglikelihood <= t:
            new_loglikelihood_contribution = logdetFt_ + vt_.T @ Finv @ vt_
            # if new_loglikelihood_contribution <= 1e-9:
            #    return None, -np.inf, None, None, None, None, None
            loglikelihood_contribution[
                t - exclude_first_observations_for_loglikelihood] = new_loglikelihood_contribution
        xtt_, Ptt_ = xtt_numba(xt, Pt, B, vt_, Finv), Ptt_numba(Pt, B, Finv)
        x[t + 1] = xpred_numba(A, xtt_, theta)
        P[t + 1] = Ppred_numba(A, Ptt_, Q)
        F[t] = Ft_
        v[t] = vt_

    loglikelihood = -Q.shape[0] * len(synth_yields) / 2 * np.log(2 * np.pi) - 1 / 2 * np.sum(loglikelihood_contribution)
    # x, x_next = pd.DataFrame(x[:-1], index=synth_yields.index), x[-1]
    x, x_next = x[:-1], x[-1]

    return x, loglikelihood, loglikelihood_contribution, P, F, v, x_next


@jit(nopython=True)
def _debug_output(A, Ft_, H, Q, x):
    print("len(x) ==", len(x))
    try:
        print("\neigvals(Ft) is")
        print(np.linalg.eigvals(Ft_))
    except:
        pass
    print("\nFt is")
    print(Ft_)
    print("\nA is")
    print(A)
    print("\nH is")
    print(H)
    print("\nQ is")
    print(Q)
    print()


def kalman_filter(synth_yields, theta, A, H, Q, B, exclude_first_observations_for_loglikelihood=0,
                  c=None):  # , no_tqdm=False):
    first_args_numpy = numpyify(synth_yields, theta, A, H, Q)
    x, loglikelihood, loglikelihood_contribution, P, F, v, x_next = _kalman_filter_inner(
        *first_args_numpy, B=numpyify_single(B),
        exclude_first_observations_for_loglikelihood=exclude_first_observations_for_loglikelihood,
        c=c)  # , no_tqdm=no_tqdm)
    if x is not None and isinstance(synth_yields, pd.DataFrame):
        x = pd.DataFrame(x, index=synth_yields.index)
    return x, loglikelihood, loglikelihood_contribution, P, F, v, x_next
