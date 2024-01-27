import numpy as np
import pandas as pd
import scipy
import sympy
from matplotlib import pyplot as plt
from numba import vectorize, jit
import logging

from .utils import is_lower_triangular

logger = logging.getLogger(__name__)

beta_0, beta_1, beta_2, beta_svensson = sympy.var(
    'beta_0 beta_1 beta_2 beta_svensson', real=True)
lambda_, lambda_svensson, T = sympy.var(
    'lambda_ lambda_svensson T', real=True, positive=True)

nss_formula = beta_0 + beta_1 * ((1 - sympy.exp(-lambda_ * T)) / (lambda_ * T)) + beta_2 * (
        (1 - sympy.exp(-lambda_ * T)) / (lambda_ * T) - sympy.exp(-lambda_ * T)) + beta_svensson * (
                      (1 - sympy.exp(-lambda_svensson * T)) / (lambda_svensson * T) - sympy.exp(
                  -lambda_svensson * T))
nss_formula_no_svensson = nss_formula.subs({"beta_svensson": 0})
diff_vars = [sympy.diff(nss_formula, var) for var in [beta_0, beta_1, beta_2, lambda_, beta_svensson, lambda_svensson]]
diff_vars_no_svensson = [sympy.diff(nss_formula_no_svensson, var) for var in [beta_0, beta_1, beta_2, lambda_]]
ns1 = (1 - sympy.exp(-lambda_ * T)) / (lambda_ * T)
ns2 = (1 - sympy.exp(-lambda_ * T)) / (lambda_ * T) - sympy.exp(-lambda_ * T)

B = sympy.Matrix((-T,
                  -(1 - sympy.exp(-lambda_ * T)) / (lambda_),
                  -(1 - sympy.exp(-lambda_svensson * T)) / (lambda_svensson),
                  T * sympy.exp(-lambda_ * T) - (1 - sympy.exp(-lambda_ * T)) / (lambda_),
                  T * sympy.exp(-lambda_svensson * T) - (1 - sympy.exp(-lambda_svensson * T)) / (lambda_svensson),
                  ))

B_ns = sympy.Matrix((-T,
                     -(1 - sympy.exp(-lambda_ * T)) / (lambda_),
                     T * sympy.exp(-lambda_ * T) - (1 - sympy.exp(-lambda_ * T)) / (lambda_)
                     ))

T1, T2 = sympy.var("T1 T2", positive=True, real=True)
B_integrals = sympy.integrate(B.subs({T: T2 - T1}), T1)
B_ns_integrals = sympy.Matrix([B_integrals[i] for i in (0, 1, 3)])

KQ = sympy.Matrix([[0, 0, 0, 0, 0],
                   [0, lambda_, 0, -lambda_, 0],
                   [0, 0, lambda_svensson, 0, -lambda_svensson],
                   [0, 0, 0, lambda_, 0],
                   [0, 0, 0, 0, lambda_svensson]])

KQ_ns = sympy.Matrix([[0, 0, 0],
                      [0, lambda_, -lambda_],
                      [0, 0, lambda_]])


def C(t1, t2, K_times_theta, Sigma, lambda_value, lambda_svensson_value=None):
    """Returns C(t1, t2) (=C(0, t2-t1))"""
    try:
        if K_times_theta is None and is_lower_triangular(Sigma):
            n = Sigma.shape[0]
            if n == 3:
                return C_quick3(t2 - t1, lambda_value,
                                Sigma[np.tril_indices(Sigma.shape[0])])
            else:
                return C_quick5(t2 - t1, lambda_value, lambda_svensson_value, Sigma[np.tril_indices(Sigma.shape[0])])
    except:
        logger.exception("in quick computation of C")
    logger.warning("Computing C slowly...")

    return C_actual(t1, t2, K_times_theta,
                    *parameters_for_C_actual(t1, t2, Sigma, lambda_value, lambda_svensson_value))


def parameters_for_C_actual(t1, t2, Sigma, lambda_value, lambda_svensson_value):
    assert Sigma.shape[0] in (3, 5) and Sigma.shape[0] == Sigma.shape[1]
    substitutions = {T1: t1, T2: t2, lambda_: lambda_value}
    if lambda_svensson_value is not None:
        substitutions[lambda_svensson] = lambda_svensson_value
    B_integrals2 = (B_integrals if Sigma.shape[0] == 5 else B_ns_integrals).subs(
        substitutions)
    substitutions = {T: t2 - T, lambda_: lambda_value}
    if lambda_svensson_value is not None:
        substitutions[lambda_svensson] = lambda_svensson_value
    B2 = (B if Sigma.shape[0] == 5 else B_ns).subs(
        substitutions)
    return B_integrals2, (Sigma.T @ B2 @ B2.T @ Sigma).diagonal()


def C_actual(t1, t2, K_times_theta, B_integrals2, matrix_diagonal):
    """Returns C(t1, t2)"""
    if K_times_theta is None:
        summands = []
    else:
        summands = [K_times_theta_entry * B_integral
                    for K_times_theta_entry, B_integral in zip(K_times_theta, B_integrals2)]
    final_summand_summands = [sympy.integrate(diagonal_entry, (T, t1, t2))
                              for diagonal_entry in matrix_diagonal]
    final_summand = 1 / 2 * sum(final_summand_summands)
    summands.append(final_summand)
    result = sum(summands)
    if not result.has_free():
        try:
            result = float(result)
        except TypeError:
            pass
    return result


def symbolic_objective(df, non_svensson=False):
    vars = [beta_0, beta_1, beta_2, lambda_, beta_svensson, lambda_svensson]
    if non_svensson:
        nss_formula2 = nss_formula.subs({"beta_svensson": 0})
        vars = vars[:-2]
    else:
        nss_formula2 = nss_formula

    main = sympy.Add(*((nss_formula2.subs({T: t}) - yield_) ** 2 for t, yield_ in
                       zip(df["TTM_precise"].dt.days.values / 365.25, df["yield"])))

    diffs = [sympy.Add(*((2 * (nss_formula2.subs({T: t}) - yield_) * diff_var.subs({T: t})
                          for t, yield_ in zip(df["TTM_precise"].dt.days.values / 365.25, df["yield"]))))
             for diff_var in (diff_vars_no_svensson if non_svensson else diff_vars)]

    return main, diffs, vars


def ns_yield(T, beta_0, beta_1, beta_2, lambda_, beta_svensson=0, lambda_svensson=1):
    """with beta_3 != 0 it becomes Svensson"""
    return beta_0 + beta_1 * ((1 - np.exp(-lambda_ * T)) / (lambda_ * T)) + beta_2 * (
            (1 - np.exp(-lambda_ * T)) / (lambda_ * T) - np.exp(-lambda_ * T)) + beta_svensson * (
            (1 - np.exp(-lambda_svensson * T)) / (lambda_ * T) - np.exp(-lambda_svensson * T))


v_ns_yield = vectorize(nopython=True)(ns_yield)


def ns_price(T, beta_0, beta_1, beta_2, lambda_, beta_svensson=0, lambda_svensson=1):
    """with beta_3 != 0 it becomes Svensson"""
    return np.exp(-T * ns_yield(T, beta_0, beta_1, beta_2,
                                lambda_, beta_svensson, lambda_svensson))


def ns_plot(up_to, beta_0, beta_1, beta_2, lambda_,
            beta_svensson=0, lambda_svensson=1, n=60, df=None,
            plot_prices: bool = False):
    x = np.linspace(0.001, up_to, n)
    if plot_prices:
        y = np.array([ns_price(single_x, beta_0, beta_1, beta_2, lambda_, beta_svensson, lambda_svensson)
                      for single_x in x])
    else:
        y = v_ns_yield(x, beta_0, beta_1, beta_2, lambda_, beta_svensson, lambda_svensson)
    # plt.ylim(-0.5/100, 8/100)
    if df is not None:
        plt.scatter(df["TTM_precise"].dt.days / 365.25, df["yield"], c="r")
    plt.axhline(0, color="k", linestyle="--")
    return plt.plot(x, y)


def deviation_arr(df, arr):
    if len(arr) == 4:  # non-svensson
        arr = np.concatenate((arr, np.array([0., 0.])))
    return deviation(df, *arr)


def deviation(df, beta_0, beta_1, beta_2, lambda_, beta_svensson=0, lambda_svensson=1):
    return pd.Series(
        v_ns_yield(df["TTM_precise"].dt.days.values / 365.25, beta_0, beta_1, beta_2, lambda_, beta_svensson,
                   lambda_svensson) - df["yield"].values, index=df.index)


def objective_ns_fit(x, df):
    return (deviation_arr(df, x) ** 2).sum()


def ns_fit(df, non_svensson=False):
    if True:
        sol = scipy.optimize.minimize(objective_ns_fit, x0=np.array([.5] * (4 if non_svensson else 6)), args=(df,))
    else:
        objective_formula, objective_jac, vars = symbolic_objective(df, non_svensson=non_svensson)
        objective_lambdified = sympy.lambdify(vars, objective_formula)
        objective2 = lambda x: objective_lambdified(*x)
        jac_lambdified = [sympy.lambdify(vars, j) for j in objective_jac]
        jac = lambda x: np.array([j(*x) for j in jac_lambdified])
        sol = scipy.optimize.minimize(objective2, x0=np.array([.5] * (4 if non_svensson else 6)),
                                      jac=jac)
    return sol


dl_lambda = .0609


def B_matrix(lambda_value, grid, lambda_svensson_value=None):
    subs = {lambda_: lambda_value}
    if lambda_svensson_value is not None:
        subs_svensson = {lambda_: lambda_svensson_value}

    def row(t):
        subs[T] = t
        if lambda_svensson_value is not None:
            subs_svensson[T] = t
            return pd.Series([1,
                              ns1.subs(subs), ns1.subs(subs_svensson),
                              ns2.subs(subs), ns2.subs(subs_svensson)])
        return pd.Series([1,
                          ns1.subs(subs),
                          ns2.subs(subs)])

    return pd.DataFrame([row(tau) for tau in grid], index=grid).astype(float)


def C_vector(lambda_, grid, K_times_theta, Sigma, lambda_svensson_value=None):
    return pd.Series([-C(0, tau, K_times_theta, Sigma, lambda_, lambda_svensson_value) / tau
                      for tau in grid], index=grid)


@jit(nopython=True, fastmath=True, cache=True, nogil=True)
def C_quick3(tau, lambda_, sigma):
    # generated using str(sympy.separatevars(sympy.simplify(C(0, tau, None, Sigma, lambda_, lambda_svensson)))),
    # where Sigma is symbolic
    exp = np.exp
    sigma_00, sigma_10, sigma_11, sigma_20, sigma_21, sigma_22 = sigma
    if sigma_10 == 0 and sigma_20 == 0 and sigma_21 == 0:
        return 2.0 * (0.0833333333333333 * lambda_ ** 3 * sigma_00 ** 2 * tau ** 3 * exp(
            2 * lambda_ * tau) - 0.125 * lambda_ ** 2 * sigma_22 ** 2 * tau ** 2 + 0.25 * lambda_ * sigma_11 ** 2 * tau * exp(
            2 * lambda_ * tau) + 0.25 * lambda_ * sigma_22 ** 2 * tau * exp(
            2 * lambda_ * tau) + 0.5 * lambda_ * sigma_22 ** 2 * tau * exp(
            lambda_ * tau) - 0.375 * lambda_ * sigma_22 ** 2 * tau - 0.375 * sigma_11 ** 2 * exp(
            2 * lambda_ * tau) + 0.5 * sigma_11 ** 2 * exp(
            lambda_ * tau) - 0.125 * sigma_11 ** 2 - 0.6875 * sigma_22 ** 2 * exp(
            2 * lambda_ * tau) + 1.0 * sigma_22 ** 2 * exp(lambda_ * tau) - 0.3125 * sigma_22 ** 2) * exp(
            -2 * lambda_ * tau) / lambda_ ** 3
    # return (0.166666666666667*lambda_**3*sigma_00**2*tau**3*exp(2*lambda_*tau) - 0.5*lambda_**2*sigma_00*tau**2*(lambda_*sigma_00*tau + sigma_10 + sigma_20)*exp(2*lambda_*tau) - 0.25*lambda_**2*sigma_20**2*tau**2 - 0.25*lambda_**2*sigma_21**2*tau**2 - 0.5*lambda_*sigma_10*sigma_20*tau - 0.5*lambda_*sigma_11*sigma_21*tau - 0.75*lambda_*sigma_20**2*tau - 0.75*lambda_*sigma_21**2*tau + 0.5*lambda_*tau*(lambda_**2*sigma_00**2*tau**2 + 2*lambda_*sigma_00*sigma_10*tau + 2*lambda_*sigma_00*sigma_20*tau + sigma_10**2 + 2*sigma_10*sigma_20 + sigma_11**2 + 2*sigma_11*sigma_21 + sigma_20**2 + sigma_21**2 + sigma_22**2)*exp(2*lambda_*tau) - 0.25*sigma_10**2 - 0.75*sigma_10*sigma_20 - 0.25*sigma_11**2 - 0.75*sigma_11*sigma_21 - 0.625*sigma_20**2 - 0.625*sigma_21**2 + sigma_22**2*(1.0*lambda_*tau + 2.0)*exp(lambda_*tau) - sigma_22**2*(0.25*lambda_**2*tau**2 + 0.75*lambda_*tau + 0.625) - 1.375*sigma_22**2*exp(2*lambda_*tau) + (1.0*lambda_*sigma_11*sigma_21*tau + 1.0*lambda_*sigma_21**2*tau + 1.0*sigma_11**2 + 3.0*sigma_11*sigma_21 + 2.0*sigma_21**2)*exp(lambda_*tau) - (1.0*sigma_00*sigma_10 + 3.0*sigma_00*sigma_20 + 0.75*sigma_10**2 + 2.25*sigma_10*sigma_20 + 0.75*sigma_11**2 + 2.25*sigma_11*sigma_21 + 1.375*sigma_20**2 + 1.375*sigma_21**2)*exp(2*lambda_*tau) + (1.0*lambda_**2*sigma_00*sigma_20*tau**2 + 1.0*lambda_*sigma_00*sigma_10*tau + 3.0*lambda_*sigma_00*sigma_20*tau + 1.0*lambda_*sigma_10*sigma_20*tau + 1.0*lambda_*sigma_20**2*tau + 1.0*sigma_00*sigma_10 + 3.0*sigma_00*sigma_20 + 1.0*sigma_10**2 + 3.0*sigma_10*sigma_20 + 2.0*sigma_20**2)*exp(lambda_*tau))*exp(-2*lambda_*tau)/lambda_**3
    return 3.0 * (0.0555555555555556 * lambda_ ** 3 * sigma_00 ** 2 * tau ** 3 * exp(
        2 * lambda_ * tau) + 0.166666666666667 * lambda_ ** 2 * sigma_00 * sigma_10 * tau ** 2 * exp(
        2 * lambda_ * tau) + 0.166666666666667 * lambda_ ** 2 * sigma_00 * sigma_20 * tau ** 2 * exp(
        2 * lambda_ * tau) + 0.333333333333333 * lambda_ ** 2 * sigma_00 * sigma_20 * tau ** 2 * exp(
        lambda_ * tau) - 0.0833333333333333 * lambda_ ** 2 * sigma_20 ** 2 * tau ** 2 - 0.0833333333333333 * lambda_ ** 2 * sigma_21 ** 2 * tau ** 2 - 0.0833333333333333 * lambda_ ** 2 * sigma_22 ** 2 * tau ** 2 + 0.333333333333333 * lambda_ * sigma_00 * sigma_10 * tau * exp(
        lambda_ * tau) + 1.0 * lambda_ * sigma_00 * sigma_20 * tau * exp(
        lambda_ * tau) + 0.166666666666667 * lambda_ * sigma_10 ** 2 * tau * exp(
        2 * lambda_ * tau) + 0.333333333333333 * lambda_ * sigma_10 * sigma_20 * tau * exp(
        2 * lambda_ * tau) + 0.333333333333333 * lambda_ * sigma_10 * sigma_20 * tau * exp(
        lambda_ * tau) - 0.166666666666667 * lambda_ * sigma_10 * sigma_20 * tau + 0.166666666666667 * lambda_ * sigma_11 ** 2 * tau * exp(
        2 * lambda_ * tau) + 0.333333333333333 * lambda_ * sigma_11 * sigma_21 * tau * exp(
        2 * lambda_ * tau) + 0.333333333333333 * lambda_ * sigma_11 * sigma_21 * tau * exp(
        lambda_ * tau) - 0.166666666666667 * lambda_ * sigma_11 * sigma_21 * tau + 0.166666666666667 * lambda_ * sigma_20 ** 2 * tau * exp(
        2 * lambda_ * tau) + 0.333333333333333 * lambda_ * sigma_20 ** 2 * tau * exp(
        lambda_ * tau) - 0.25 * lambda_ * sigma_20 ** 2 * tau + 0.166666666666667 * lambda_ * sigma_21 ** 2 * tau * exp(
        2 * lambda_ * tau) + 0.333333333333333 * lambda_ * sigma_21 ** 2 * tau * exp(
        lambda_ * tau) - 0.25 * lambda_ * sigma_21 ** 2 * tau + 0.166666666666667 * lambda_ * sigma_22 ** 2 * tau * exp(
        2 * lambda_ * tau) + 0.333333333333333 * lambda_ * sigma_22 ** 2 * tau * exp(
        lambda_ * tau) - 0.25 * lambda_ * sigma_22 ** 2 * tau - 0.333333333333333 * sigma_00 * sigma_10 * exp(
        2 * lambda_ * tau) + 0.333333333333333 * sigma_00 * sigma_10 * exp(
        lambda_ * tau) - 1.0 * sigma_00 * sigma_20 * exp(2 * lambda_ * tau) + 1.0 * sigma_00 * sigma_20 * exp(
        lambda_ * tau) - 0.25 * sigma_10 ** 2 * exp(2 * lambda_ * tau) + 0.333333333333333 * sigma_10 ** 2 * exp(
        lambda_ * tau) - 0.0833333333333333 * sigma_10 ** 2 - 0.75 * sigma_10 * sigma_20 * exp(
        2 * lambda_ * tau) + 1.0 * sigma_10 * sigma_20 * exp(
        lambda_ * tau) - 0.25 * sigma_10 * sigma_20 - 0.25 * sigma_11 ** 2 * exp(
        2 * lambda_ * tau) + 0.333333333333333 * sigma_11 ** 2 * exp(
        lambda_ * tau) - 0.0833333333333333 * sigma_11 ** 2 - 0.75 * sigma_11 * sigma_21 * exp(
        2 * lambda_ * tau) + 1.0 * sigma_11 * sigma_21 * exp(
        lambda_ * tau) - 0.25 * sigma_11 * sigma_21 - 0.458333333333333 * sigma_20 ** 2 * exp(
        2 * lambda_ * tau) + 0.666666666666667 * sigma_20 ** 2 * exp(
        lambda_ * tau) - 0.208333333333333 * sigma_20 ** 2 - 0.458333333333333 * sigma_21 ** 2 * exp(
        2 * lambda_ * tau) + 0.666666666666667 * sigma_21 ** 2 * exp(
        lambda_ * tau) - 0.208333333333333 * sigma_21 ** 2 - 0.458333333333333 * sigma_22 ** 2 * exp(
        2 * lambda_ * tau) + 0.666666666666667 * sigma_22 ** 2 * exp(
        lambda_ * tau) - 0.208333333333333 * sigma_22 ** 2) * exp(-2 * lambda_ * tau) / lambda_ ** 3


@jit(nopython=True, fastmath=True, cache=True, nogil=True)
def C_quick5(tau, lambda_, lambda_svensson, sigma):
    # generated using str(sympy.separatevars(sympy.simplify(C(0, tau, None, Sigma, lambda_, lambda_svensson)))),
    # where Sigma is symbolic
    exp = np.exp
    sigma_00, sigma_10, sigma_11, sigma_20, sigma_21, sigma_22, sigma_30, sigma_31, sigma_32, sigma_33, \
        sigma_40, sigma_41, sigma_42, sigma_43, sigma_44 = sigma
    if (sigma_10 == 0 and sigma_20 == 0 and sigma_21 == 0 and sigma_30 == 0 and sigma_31 == 0 and sigma_32 == 0
            and sigma_40 == 0 and sigma_41 == 0 and sigma_42 == 0 and sigma_43 == 0):
        return 2.0 * (0.0833333333333333 * lambda_ ** 3 * lambda_svensson ** 3 * sigma_00 ** 2 * tau ** 3 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) - 0.125 * lambda_ ** 3 * lambda_svensson ** 2 * sigma_44 ** 2 * tau ** 2 * exp(
            2 * lambda_ * tau) + 0.25 * lambda_ ** 3 * lambda_svensson * sigma_22 ** 2 * tau * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.25 * lambda_ ** 3 * lambda_svensson * sigma_44 ** 2 * tau * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.5 * lambda_ ** 3 * lambda_svensson * sigma_44 ** 2 * tau * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.375 * lambda_ ** 3 * lambda_svensson * sigma_44 ** 2 * tau * exp(
            2 * lambda_ * tau) - 0.375 * lambda_ ** 3 * sigma_22 ** 2 * exp(2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.5 * lambda_ ** 3 * sigma_22 ** 2 * exp(2 * lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.125 * lambda_ ** 3 * sigma_22 ** 2 * exp(
            2 * lambda_ * tau) - 0.6875 * lambda_ ** 3 * sigma_44 ** 2 * exp(2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 1.0 * lambda_ ** 3 * sigma_44 ** 2 * exp(2 * lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.3125 * lambda_ ** 3 * sigma_44 ** 2 * exp(
            2 * lambda_ * tau) - 0.125 * lambda_ ** 2 * lambda_svensson ** 3 * sigma_33 ** 2 * tau ** 2 * exp(
            2 * lambda_svensson * tau) + 0.25 * lambda_ * lambda_svensson ** 3 * sigma_11 ** 2 * tau * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.25 * lambda_ * lambda_svensson ** 3 * sigma_33 ** 2 * tau * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.5 * lambda_ * lambda_svensson ** 3 * sigma_33 ** 2 * tau * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) - 0.375 * lambda_ * lambda_svensson ** 3 * sigma_33 ** 2 * tau * exp(
            2 * lambda_svensson * tau) - 0.375 * lambda_svensson ** 3 * sigma_11 ** 2 * exp(2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.5 * lambda_svensson ** 3 * sigma_11 ** 2 * exp(lambda_ * tau) * exp(
            2 * lambda_svensson * tau) - 0.125 * lambda_svensson ** 3 * sigma_11 ** 2 * exp(
            2 * lambda_svensson * tau) - 0.6875 * lambda_svensson ** 3 * sigma_33 ** 2 * exp(2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 1.0 * lambda_svensson ** 3 * sigma_33 ** 2 * exp(lambda_ * tau) * exp(
            2 * lambda_svensson * tau) - 0.3125 * lambda_svensson ** 3 * sigma_33 ** 2 * exp(
            2 * lambda_svensson * tau)) * exp(-2 * lambda_ * tau) * exp(-2 * lambda_svensson * tau) / (
                    lambda_ ** 3 * lambda_svensson ** 3)

    # return (0.5*lambda_**3*lambda_svensson*sigma_44**2*tau*(lambda_**2 + 2*lambda_*lambda_svensson + lambda_svensson**2)*(lambda_**3 + 3*lambda_**2*lambda_svensson + 3*lambda_*lambda_svensson**2 + lambda_svensson**3)*exp(tau*(5*lambda_ + 7*lambda_svensson)) + lambda_**3*sigma_44**2*(lambda_**2 + 2*lambda_*lambda_svensson + lambda_svensson**2)*(lambda_**3 + 3*lambda_**2*lambda_svensson + 3*lambda_*lambda_svensson**2 + lambda_svensson**3)*(-0.25*lambda_svensson**2*tau**2 - 0.75*lambda_svensson*tau + (1.0*lambda_svensson*tau + 2.0)*exp(lambda_svensson*tau) - 0.625)*exp(5*tau*(lambda_ + lambda_svensson)) - 1.375*lambda_**3*sigma_44**2*(lambda_**2 + 2*lambda_*lambda_svensson + lambda_svensson**2)*(lambda_**3 + 3*lambda_**2*lambda_svensson + 3*lambda_*lambda_svensson**2 + lambda_svensson**3)*exp(tau*(5*lambda_ + 7*lambda_svensson)) + 0.125*lambda_*(lambda_**2 + 2*lambda_*lambda_svensson + lambda_svensson**2)*(-sigma_40*(-2*lambda_**5*lambda_svensson**2*sigma_00*tau**2 - 4*lambda_**5*lambda_svensson*sigma_20*tau - 4*lambda_**5*lambda_svensson*sigma_40*tau + 12*lambda_**5*sigma_00 + 9*lambda_**5*sigma_20 + 11*lambda_**5*sigma_40 - 6*lambda_**4*lambda_svensson**3*sigma_00*tau**2 - 4*lambda_**4*lambda_svensson**2*sigma_10*tau - 12*lambda_**4*lambda_svensson**2*sigma_20*tau - 4*lambda_**4*lambda_svensson**2*sigma_30*tau - 12*lambda_**4*lambda_svensson**2*sigma_40*tau + 36*lambda_**4*lambda_svensson*sigma_00 + 8*lambda_**4*lambda_svensson*sigma_10 + 27*lambda_**4*lambda_svensson*sigma_20 + 8*lambda_**4*lambda_svensson*sigma_30 + 33*lambda_**4*lambda_svensson*sigma_40 - 6*lambda_**3*lambda_svensson**4*sigma_00*tau**2 - 12*lambda_**3*lambda_svensson**3*sigma_10*tau - 12*lambda_**3*lambda_svensson**3*sigma_20*tau - 12*lambda_**3*lambda_svensson**3*sigma_30*tau - 12*lambda_**3*lambda_svensson**3*sigma_40*tau + 36*lambda_**3*lambda_svensson**2*sigma_00 + 24*lambda_**3*lambda_svensson**2*sigma_10 + 27*lambda_**3*lambda_svensson**2*sigma_20 + 24*lambda_**3*lambda_svensson**2*sigma_30 + 33*lambda_**3*lambda_svensson**2*sigma_40 - 2*lambda_**2*lambda_svensson**5*sigma_00*tau**2 - 12*lambda_**2*lambda_svensson**4*sigma_10*tau - 4*lambda_**2*lambda_svensson**4*sigma_20*tau - 12*lambda_**2*lambda_svensson**4*sigma_30*tau - 4*lambda_**2*lambda_svensson**4*sigma_40*tau + 12*lambda_**2*lambda_svensson**3*sigma_00 + 24*lambda_**2*lambda_svensson**3*sigma_10 + 9*lambda_**2*lambda_svensson**3*sigma_20 + 24*lambda_**2*lambda_svensson**3*sigma_30 + 11*lambda_**2*lambda_svensson**3*sigma_40 - 4*lambda_*lambda_svensson**5*sigma_10*tau - 4*lambda_*lambda_svensson**5*sigma_30*tau + 12*lambda_*lambda_svensson**4*sigma_10 + 24*lambda_*lambda_svensson**4*sigma_30 + 4*lambda_svensson**5*sigma_10 + 8*lambda_svensson**5*sigma_30)*exp(tau*(lambda_ + 2*lambda_svensson)) + sigma_40*(4*lambda_**5*lambda_svensson**2*sigma_00*tau**2*exp(tau*(lambda_ + lambda_svensson)) - 2*lambda_**5*lambda_svensson**2*sigma_40*tau**2*exp(lambda_*tau) + 12*lambda_**5*lambda_svensson*sigma_00*tau*exp(tau*(lambda_ + lambda_svensson)) - 2*lambda_**5*lambda_svensson*sigma_20*tau*exp(lambda_*tau) + 4*lambda_**5*lambda_svensson*sigma_20*tau*exp(tau*(lambda_ + lambda_svensson)) - 6*lambda_**5*lambda_svensson*sigma_40*tau*exp(lambda_*tau) + 8*lambda_**5*lambda_svensson*sigma_40*tau*exp(tau*(lambda_ + lambda_svensson)) + 12*lambda_**5*sigma_00*exp(tau*(lambda_ + lambda_svensson)) - 3*lambda_**5*sigma_20*exp(lambda_*tau) + 12*lambda_**5*sigma_20*exp(tau*(lambda_ + lambda_svensson)) - 5*lambda_**5*sigma_40*exp(lambda_*tau) + 16*lambda_**5*sigma_40*exp(tau*(lambda_ + lambda_svensson)) + 12*lambda_**4*lambda_svensson**3*sigma_00*tau**2*exp(tau*(lambda_ + lambda_svensson)) - 4*lambda_**4*lambda_svensson**3*sigma_30*tau**2*exp(lambda_svensson*tau) - 6*lambda_**4*lambda_svensson**3*sigma_40*tau**2*exp(lambda_*tau) + 36*lambda_**4*lambda_svensson**2*sigma_00*tau*exp(tau*(lambda_ + lambda_svensson)) + 4*lambda_**4*lambda_svensson**2*sigma_10*tau*exp(tau*(lambda_ + lambda_svensson)) - 6*lambda_**4*lambda_svensson**2*sigma_20*tau*exp(lambda_*tau) + 12*lambda_**4*lambda_svensson**2*sigma_20*tau*exp(tau*(lambda_ + lambda_svensson)) + 4*lambda_**4*lambda_svensson**2*sigma_30*tau*exp(2*lambda_svensson*tau) - 4*lambda_**4*lambda_svensson**2*sigma_30*tau*exp(lambda_svensson*tau) + 4*lambda_**4*lambda_svensson**2*sigma_30*tau*exp(tau*(lambda_ + lambda_svensson)) - 18*lambda_**4*lambda_svensson**2*sigma_40*tau*exp(lambda_*tau) + 24*lambda_**4*lambda_svensson**2*sigma_40*tau*exp(tau*(lambda_ + lambda_svensson)) + 36*lambda_**4*lambda_svensson*sigma_00*exp(tau*(lambda_ + lambda_svensson)) + 8*lambda_**4*lambda_svensson*sigma_10*exp(tau*(lambda_ + lambda_svensson)) - 9*lambda_**4*lambda_svensson*sigma_20*exp(lambda_*tau) + 36*lambda_**4*lambda_svensson*sigma_20*exp(tau*(lambda_ + lambda_svensson)) + 8*lambda_**4*lambda_svensson*sigma_30*exp(tau*(lambda_ + lambda_svensson)) - 15*lambda_**4*lambda_svensson*sigma_40*exp(lambda_*tau) + 48*lambda_**4*lambda_svensson*sigma_40*exp(tau*(lambda_ + lambda_svensson)) + 12*lambda_**3*lambda_svensson**4*sigma_00*tau**2*exp(tau*(lambda_ + lambda_svensson)) - 8*lambda_**3*lambda_svensson**4*sigma_30*tau**2*exp(lambda_svensson*tau) - 6*lambda_**3*lambda_svensson**4*sigma_40*tau**2*exp(lambda_*tau) + 36*lambda_**3*lambda_svensson**3*sigma_00*tau*exp(tau*(lambda_ + lambda_svensson)) - 4*lambda_**3*lambda_svensson**3*sigma_10*tau*exp(lambda_svensson*tau) + 12*lambda_**3*lambda_svensson**3*sigma_10*tau*exp(tau*(lambda_ + lambda_svensson)) - 6*lambda_**3*lambda_svensson**3*sigma_20*tau*exp(lambda_*tau) + 12*lambda_**3*lambda_svensson**3*sigma_20*tau*exp(tau*(lambda_ + lambda_svensson)) + 12*lambda_**3*lambda_svensson**3*sigma_30*tau*exp(2*lambda_svensson*tau) - 20*lambda_**3*lambda_svensson**3*sigma_30*tau*exp(lambda_svensson*tau) + 12*lambda_**3*lambda_svensson**3*sigma_30*tau*exp(tau*(lambda_ + lambda_svensson)) - 18*lambda_**3*lambda_svensson**3*sigma_40*tau*exp(lambda_*tau) + 24*lambda_**3*lambda_svensson**3*sigma_40*tau*exp(tau*(lambda_ + lambda_svensson)) + 36*lambda_**3*lambda_svensson**2*sigma_00*exp(tau*(lambda_ + lambda_svensson)) + 4*lambda_**3*lambda_svensson**2*sigma_10*exp(2*lambda_svensson*tau) - 4*lambda_**3*lambda_svensson**2*sigma_10*exp(lambda_svensson*tau) + 24*lambda_**3*lambda_svensson**2*sigma_10*exp(tau*(lambda_ + lambda_svensson)) - 9*lambda_**3*lambda_svensson**2*sigma_20*exp(lambda_*tau) + 36*lambda_**3*lambda_svensson**2*sigma_20*exp(tau*(lambda_ + lambda_svensson)) + 8*lambda_**3*lambda_svensson**2*sigma_30*exp(2*lambda_svensson*tau) - 8*lambda_**3*lambda_svensson**2*sigma_30*exp(lambda_svensson*tau) + 24*lambda_**3*lambda_svensson**2*sigma_30*exp(tau*(lambda_ + lambda_svensson)) - 15*lambda_**3*lambda_svensson**2*sigma_40*exp(lambda_*tau) + 48*lambda_**3*lambda_svensson**2*sigma_40*exp(tau*(lambda_ + lambda_svensson)) + 4*lambda_**2*lambda_svensson**5*sigma_00*tau**2*exp(tau*(lambda_ + lambda_svensson)) - 4*lambda_**2*lambda_svensson**5*sigma_30*tau**2*exp(lambda_svensson*tau) - 2*lambda_**2*lambda_svensson**5*sigma_40*tau**2*exp(lambda_*tau) + 12*lambda_**2*lambda_svensson**4*sigma_00*tau*exp(tau*(lambda_ + lambda_svensson)) - 8*lambda_**2*lambda_svensson**4*sigma_10*tau*exp(lambda_svensson*tau) + 12*lambda_**2*lambda_svensson**4*sigma_10*tau*exp(tau*(lambda_ + lambda_svensson)) - 2*lambda_**2*lambda_svensson**4*sigma_20*tau*exp(lambda_*tau) + 4*lambda_**2*lambda_svensson**4*sigma_20*tau*exp(tau*(lambda_ + lambda_svensson)) + 12*lambda_**2*lambda_svensson**4*sigma_30*tau*exp(2*lambda_svensson*tau) - 20*lambda_**2*lambda_svensson**4*sigma_30*tau*exp(lambda_svensson*tau) + 12*lambda_**2*lambda_svensson**4*sigma_30*tau*exp(tau*(lambda_ + lambda_svensson)) - 6*lambda_**2*lambda_svensson**4*sigma_40*tau*exp(lambda_*tau) + 8*lambda_**2*lambda_svensson**4*sigma_40*tau*exp(tau*(lambda_ + lambda_svensson)) + 12*lambda_**2*lambda_svensson**3*sigma_00*exp(tau*(lambda_ + lambda_svensson)) + 12*lambda_**2*lambda_svensson**3*sigma_10*exp(2*lambda_svensson*tau) - 12*lambda_**2*lambda_svensson**3*sigma_10*exp(lambda_svensson*tau) + 24*lambda_**2*lambda_svensson**3*sigma_10*exp(tau*(lambda_ + lambda_svensson)) - 3*lambda_**2*lambda_svensson**3*sigma_20*exp(lambda_*tau) + 12*lambda_**2*lambda_svensson**3*sigma_20*exp(tau*(lambda_ + lambda_svensson)) + 24*lambda_**2*lambda_svensson**3*sigma_30*exp(2*lambda_svensson*tau) - 24*lambda_**2*lambda_svensson**3*sigma_30*exp(lambda_svensson*tau) + 24*lambda_**2*lambda_svensson**3*sigma_30*exp(tau*(lambda_ + lambda_svensson)) - 5*lambda_**2*lambda_svensson**3*sigma_40*exp(lambda_*tau) + 16*lambda_**2*lambda_svensson**3*sigma_40*exp(tau*(lambda_ + lambda_svensson)) - 4*lambda_*lambda_svensson**5*sigma_10*tau*exp(lambda_svensson*tau) + 4*lambda_*lambda_svensson**5*sigma_10*tau*exp(tau*(lambda_ + lambda_svensson)) + 4*lambda_*lambda_svensson**5*sigma_30*tau*exp(2*lambda_svensson*tau) - 4*lambda_*lambda_svensson**5*sigma_30*tau*exp(lambda_svensson*tau) + 4*lambda_*lambda_svensson**5*sigma_30*tau*exp(tau*(lambda_ + lambda_svensson)) + 12*lambda_*lambda_svensson**4*sigma_10*exp(2*lambda_svensson*tau) - 8*lambda_*lambda_svensson**4*sigma_10*exp(lambda_svensson*tau) + 8*lambda_*lambda_svensson**4*sigma_10*exp(tau*(lambda_ + lambda_svensson)) + 24*lambda_*lambda_svensson**4*sigma_30*exp(2*lambda_svensson*tau) - 8*lambda_*lambda_svensson**4*sigma_30*exp(lambda_svensson*tau) + 8*lambda_*lambda_svensson**4*sigma_30*exp(tau*(lambda_ + lambda_svensson)) + 4*lambda_svensson**5*sigma_10*exp(2*lambda_svensson*tau) + 8*lambda_svensson**5*sigma_30*exp(2*lambda_svensson*tau)) - sigma_41*(-4*lambda_**5*lambda_svensson*sigma_21*tau - 4*lambda_**5*lambda_svensson*sigma_41*tau + 9*lambda_**5*sigma_21 + 11*lambda_**5*sigma_41 - 4*lambda_**4*lambda_svensson**2*sigma_11*tau - 12*lambda_**4*lambda_svensson**2*sigma_21*tau - 4*lambda_**4*lambda_svensson**2*sigma_31*tau - 12*lambda_**4*lambda_svensson**2*sigma_41*tau + 8*lambda_**4*lambda_svensson*sigma_11 + 27*lambda_**4*lambda_svensson*sigma_21 + 8*lambda_**4*lambda_svensson*sigma_31 + 33*lambda_**4*lambda_svensson*sigma_41 - 12*lambda_**3*lambda_svensson**3*sigma_11*tau - 12*lambda_**3*lambda_svensson**3*sigma_21*tau - 12*lambda_**3*lambda_svensson**3*sigma_31*tau - 12*lambda_**3*lambda_svensson**3*sigma_41*tau + 24*lambda_**3*lambda_svensson**2*sigma_11 + 27*lambda_**3*lambda_svensson**2*sigma_21 + 24*lambda_**3*lambda_svensson**2*sigma_31 + 33*lambda_**3*lambda_svensson**2*sigma_41 - 12*lambda_**2*lambda_svensson**4*sigma_11*tau - 4*lambda_**2*lambda_svensson**4*sigma_21*tau - 12*lambda_**2*lambda_svensson**4*sigma_31*tau - 4*lambda_**2*lambda_svensson**4*sigma_41*tau + 24*lambda_**2*lambda_svensson**3*sigma_11 + 9*lambda_**2*lambda_svensson**3*sigma_21 + 24*lambda_**2*lambda_svensson**3*sigma_31 + 11*lambda_**2*lambda_svensson**3*sigma_41 - 4*lambda_*lambda_svensson**5*sigma_11*tau - 4*lambda_*lambda_svensson**5*sigma_31*tau + 12*lambda_*lambda_svensson**4*sigma_11 + 24*lambda_*lambda_svensson**4*sigma_31 + 4*lambda_svensson**5*sigma_11 + 8*lambda_svensson**5*sigma_31)*exp(tau*(lambda_ + 2*lambda_svensson)) + sigma_41*(-2*lambda_**5*lambda_svensson**2*sigma_41*tau**2*exp(lambda_*tau) - 2*lambda_**5*lambda_svensson*sigma_21*tau*exp(lambda_*tau) + 4*lambda_**5*lambda_svensson*sigma_21*tau*exp(tau*(lambda_ + lambda_svensson)) - 6*lambda_**5*lambda_svensson*sigma_41*tau*exp(lambda_*tau) + 8*lambda_**5*lambda_svensson*sigma_41*tau*exp(tau*(lambda_ + lambda_svensson)) - 3*lambda_**5*sigma_21*exp(lambda_*tau) + 12*lambda_**5*sigma_21*exp(tau*(lambda_ + lambda_svensson)) - 5*lambda_**5*sigma_41*exp(lambda_*tau) + 16*lambda_**5*sigma_41*exp(tau*(lambda_ + lambda_svensson)) - 4*lambda_**4*lambda_svensson**3*sigma_31*tau**2*exp(lambda_svensson*tau) - 6*lambda_**4*lambda_svensson**3*sigma_41*tau**2*exp(lambda_*tau) + 4*lambda_**4*lambda_svensson**2*sigma_11*tau*exp(tau*(lambda_ + lambda_svensson)) - 6*lambda_**4*lambda_svensson**2*sigma_21*tau*exp(lambda_*tau) + 12*lambda_**4*lambda_svensson**2*sigma_21*tau*exp(tau*(lambda_ + lambda_svensson)) + 4*lambda_**4*lambda_svensson**2*sigma_31*tau*exp(2*lambda_svensson*tau) - 4*lambda_**4*lambda_svensson**2*sigma_31*tau*exp(lambda_svensson*tau) + 4*lambda_**4*lambda_svensson**2*sigma_31*tau*exp(tau*(lambda_ + lambda_svensson)) - 18*lambda_**4*lambda_svensson**2*sigma_41*tau*exp(lambda_*tau) + 24*lambda_**4*lambda_svensson**2*sigma_41*tau*exp(tau*(lambda_ + lambda_svensson)) + 8*lambda_**4*lambda_svensson*sigma_11*exp(tau*(lambda_ + lambda_svensson)) - 9*lambda_**4*lambda_svensson*sigma_21*exp(lambda_*tau) + 36*lambda_**4*lambda_svensson*sigma_21*exp(tau*(lambda_ + lambda_svensson)) + 8*lambda_**4*lambda_svensson*sigma_31*exp(tau*(lambda_ + lambda_svensson)) - 15*lambda_**4*lambda_svensson*sigma_41*exp(lambda_*tau) + 48*lambda_**4*lambda_svensson*sigma_41*exp(tau*(lambda_ + lambda_svensson)) - 8*lambda_**3*lambda_svensson**4*sigma_31*tau**2*exp(lambda_svensson*tau) - 6*lambda_**3*lambda_svensson**4*sigma_41*tau**2*exp(lambda_*tau) - 4*lambda_**3*lambda_svensson**3*sigma_11*tau*exp(lambda_svensson*tau) + 12*lambda_**3*lambda_svensson**3*sigma_11*tau*exp(tau*(lambda_ + lambda_svensson)) - 6*lambda_**3*lambda_svensson**3*sigma_21*tau*exp(lambda_*tau) + 12*lambda_**3*lambda_svensson**3*sigma_21*tau*exp(tau*(lambda_ + lambda_svensson)) + 12*lambda_**3*lambda_svensson**3*sigma_31*tau*exp(2*lambda_svensson*tau) - 20*lambda_**3*lambda_svensson**3*sigma_31*tau*exp(lambda_svensson*tau) + 12*lambda_**3*lambda_svensson**3*sigma_31*tau*exp(tau*(lambda_ + lambda_svensson)) - 18*lambda_**3*lambda_svensson**3*sigma_41*tau*exp(lambda_*tau) + 24*lambda_**3*lambda_svensson**3*sigma_41*tau*exp(tau*(lambda_ + lambda_svensson)) + 4*lambda_**3*lambda_svensson**2*sigma_11*exp(2*lambda_svensson*tau) - 4*lambda_**3*lambda_svensson**2*sigma_11*exp(lambda_svensson*tau) + 24*lambda_**3*lambda_svensson**2*sigma_11*exp(tau*(lambda_ + lambda_svensson)) - 9*lambda_**3*lambda_svensson**2*sigma_21*exp(lambda_*tau) + 36*lambda_**3*lambda_svensson**2*sigma_21*exp(tau*(lambda_ + lambda_svensson)) + 8*lambda_**3*lambda_svensson**2*sigma_31*exp(2*lambda_svensson*tau) - 8*lambda_**3*lambda_svensson**2*sigma_31*exp(lambda_svensson*tau) + 24*lambda_**3*lambda_svensson**2*sigma_31*exp(tau*(lambda_ + lambda_svensson)) - 15*lambda_**3*lambda_svensson**2*sigma_41*exp(lambda_*tau) + 48*lambda_**3*lambda_svensson**2*sigma_41*exp(tau*(lambda_ + lambda_svensson)) - 4*lambda_**2*lambda_svensson**5*sigma_31*tau**2*exp(lambda_svensson*tau) - 2*lambda_**2*lambda_svensson**5*sigma_41*tau**2*exp(lambda_*tau) - 8*lambda_**2*lambda_svensson**4*sigma_11*tau*exp(lambda_svensson*tau) + 12*lambda_**2*lambda_svensson**4*sigma_11*tau*exp(tau*(lambda_ + lambda_svensson)) - 2*lambda_**2*lambda_svensson**4*sigma_21*tau*exp(lambda_*tau) + 4*lambda_**2*lambda_svensson**4*sigma_21*tau*exp(tau*(lambda_ + lambda_svensson)) + 12*lambda_**2*lambda_svensson**4*sigma_31*tau*exp(2*lambda_svensson*tau) - 20*lambda_**2*lambda_svensson**4*sigma_31*tau*exp(lambda_svensson*tau) + 12*lambda_**2*lambda_svensson**4*sigma_31*tau*exp(tau*(lambda_ + lambda_svensson)) - 6*lambda_**2*lambda_svensson**4*sigma_41*tau*exp(lambda_*tau) + 8*lambda_**2*lambda_svensson**4*sigma_41*tau*exp(tau*(lambda_ + lambda_svensson)) + 12*lambda_**2*lambda_svensson**3*sigma_11*exp(2*lambda_svensson*tau) - 12*lambda_**2*lambda_svensson**3*sigma_11*exp(lambda_svensson*tau) + 24*lambda_**2*lambda_svensson**3*sigma_11*exp(tau*(lambda_ + lambda_svensson)) - 3*lambda_**2*lambda_svensson**3*sigma_21*exp(lambda_*tau) + 12*lambda_**2*lambda_svensson**3*sigma_21*exp(tau*(lambda_ + lambda_svensson)) + 24*lambda_**2*lambda_svensson**3*sigma_31*exp(2*lambda_svensson*tau) - 24*lambda_**2*lambda_svensson**3*sigma_31*exp(lambda_svensson*tau) + 24*lambda_**2*lambda_svensson**3*sigma_31*exp(tau*(lambda_ + lambda_svensson)) - 5*lambda_**2*lambda_svensson**3*sigma_41*exp(lambda_*tau) + 16*lambda_**2*lambda_svensson**3*sigma_41*exp(tau*(lambda_ + lambda_svensson)) - 4*lambda_*lambda_svensson**5*sigma_11*tau*exp(lambda_svensson*tau) + 4*lambda_*lambda_svensson**5*sigma_11*tau*exp(tau*(lambda_ + lambda_svensson)) + 4*lambda_*lambda_svensson**5*sigma_31*tau*exp(2*lambda_svensson*tau) - 4*lambda_*lambda_svensson**5*sigma_31*tau*exp(lambda_svensson*tau) + 4*lambda_*lambda_svensson**5*sigma_31*tau*exp(tau*(lambda_ + lambda_svensson)) + 12*lambda_*lambda_svensson**4*sigma_11*exp(2*lambda_svensson*tau) - 8*lambda_*lambda_svensson**4*sigma_11*exp(lambda_svensson*tau) + 8*lambda_*lambda_svensson**4*sigma_11*exp(tau*(lambda_ + lambda_svensson)) + 24*lambda_*lambda_svensson**4*sigma_31*exp(2*lambda_svensson*tau) - 8*lambda_*lambda_svensson**4*sigma_31*exp(lambda_svensson*tau) + 8*lambda_*lambda_svensson**4*sigma_31*exp(tau*(lambda_ + lambda_svensson)) + 4*lambda_svensson**5*sigma_11*exp(2*lambda_svensson*tau) + 8*lambda_svensson**5*sigma_31*exp(2*lambda_svensson*tau)) - sigma_42*(-4*lambda_**5*lambda_svensson*sigma_22*tau - 4*lambda_**5*lambda_svensson*sigma_42*tau + 9*lambda_**5*sigma_22 + 11*lambda_**5*sigma_42 - 12*lambda_**4*lambda_svensson**2*sigma_22*tau - 4*lambda_**4*lambda_svensson**2*sigma_32*tau - 12*lambda_**4*lambda_svensson**2*sigma_42*tau + 27*lambda_**4*lambda_svensson*sigma_22 + 8*lambda_**4*lambda_svensson*sigma_32 + 33*lambda_**4*lambda_svensson*sigma_42 - 12*lambda_**3*lambda_svensson**3*sigma_22*tau - 12*lambda_**3*lambda_svensson**3*sigma_32*tau - 12*lambda_**3*lambda_svensson**3*sigma_42*tau + 27*lambda_**3*lambda_svensson**2*sigma_22 + 24*lambda_**3*lambda_svensson**2*sigma_32 + 33*lambda_**3*lambda_svensson**2*sigma_42 - 4*lambda_**2*lambda_svensson**4*sigma_22*tau - 12*lambda_**2*lambda_svensson**4*sigma_32*tau - 4*lambda_**2*lambda_svensson**4*sigma_42*tau + 9*lambda_**2*lambda_svensson**3*sigma_22 + 24*lambda_**2*lambda_svensson**3*sigma_32 + 11*lambda_**2*lambda_svensson**3*sigma_42 - 4*lambda_*lambda_svensson**5*sigma_32*tau + 24*lambda_*lambda_svensson**4*sigma_32 + 8*lambda_svensson**5*sigma_32)*exp(tau*(lambda_ + 2*lambda_svensson)) + sigma_42*(-2*lambda_**5*lambda_svensson**2*sigma_42*tau**2*exp(lambda_*tau) - 2*lambda_**5*lambda_svensson*sigma_22*tau*exp(lambda_*tau) + 4*lambda_**5*lambda_svensson*sigma_22*tau*exp(tau*(lambda_ + lambda_svensson)) - 6*lambda_**5*lambda_svensson*sigma_42*tau*exp(lambda_*tau) + 8*lambda_**5*lambda_svensson*sigma_42*tau*exp(tau*(lambda_ + lambda_svensson)) - 3*lambda_**5*sigma_22*exp(lambda_*tau) + 12*lambda_**5*sigma_22*exp(tau*(lambda_ + lambda_svensson)) - 5*lambda_**5*sigma_42*exp(lambda_*tau) + 16*lambda_**5*sigma_42*exp(tau*(lambda_ + lambda_svensson)) - 4*lambda_**4*lambda_svensson**3*sigma_32*tau**2*exp(lambda_svensson*tau) - 6*lambda_**4*lambda_svensson**3*sigma_42*tau**2*exp(lambda_*tau) - 6*lambda_**4*lambda_svensson**2*sigma_22*tau*exp(lambda_*tau) + 12*lambda_**4*lambda_svensson**2*sigma_22*tau*exp(tau*(lambda_ + lambda_svensson)) + 4*lambda_**4*lambda_svensson**2*sigma_32*tau*exp(2*lambda_svensson*tau) - 4*lambda_**4*lambda_svensson**2*sigma_32*tau*exp(lambda_svensson*tau) + 4*lambda_**4*lambda_svensson**2*sigma_32*tau*exp(tau*(lambda_ + lambda_svensson)) - 18*lambda_**4*lambda_svensson**2*sigma_42*tau*exp(lambda_*tau) + 24*lambda_**4*lambda_svensson**2*sigma_42*tau*exp(tau*(lambda_ + lambda_svensson)) - 9*lambda_**4*lambda_svensson*sigma_22*exp(lambda_*tau) + 36*lambda_**4*lambda_svensson*sigma_22*exp(tau*(lambda_ + lambda_svensson)) + 8*lambda_**4*lambda_svensson*sigma_32*exp(tau*(lambda_ + lambda_svensson)) - 15*lambda_**4*lambda_svensson*sigma_42*exp(lambda_*tau) + 48*lambda_**4*lambda_svensson*sigma_42*exp(tau*(lambda_ + lambda_svensson)) - 8*lambda_**3*lambda_svensson**4*sigma_32*tau**2*exp(lambda_svensson*tau) - 6*lambda_**3*lambda_svensson**4*sigma_42*tau**2*exp(lambda_*tau) - 6*lambda_**3*lambda_svensson**3*sigma_22*tau*exp(lambda_*tau) + 12*lambda_**3*lambda_svensson**3*sigma_22*tau*exp(tau*(lambda_ + lambda_svensson)) + 12*lambda_**3*lambda_svensson**3*sigma_32*tau*exp(2*lambda_svensson*tau) - 20*lambda_**3*lambda_svensson**3*sigma_32*tau*exp(lambda_svensson*tau) + 12*lambda_**3*lambda_svensson**3*sigma_32*tau*exp(tau*(lambda_ + lambda_svensson)) - 18*lambda_**3*lambda_svensson**3*sigma_42*tau*exp(lambda_*tau) + 24*lambda_**3*lambda_svensson**3*sigma_42*tau*exp(tau*(lambda_ + lambda_svensson)) - 9*lambda_**3*lambda_svensson**2*sigma_22*exp(lambda_*tau) + 36*lambda_**3*lambda_svensson**2*sigma_22*exp(tau*(lambda_ + lambda_svensson)) + 8*lambda_**3*lambda_svensson**2*sigma_32*exp(2*lambda_svensson*tau) - 8*lambda_**3*lambda_svensson**2*sigma_32*exp(lambda_svensson*tau) + 24*lambda_**3*lambda_svensson**2*sigma_32*exp(tau*(lambda_ + lambda_svensson)) - 15*lambda_**3*lambda_svensson**2*sigma_42*exp(lambda_*tau) + 48*lambda_**3*lambda_svensson**2*sigma_42*exp(tau*(lambda_ + lambda_svensson)) - 4*lambda_**2*lambda_svensson**5*sigma_32*tau**2*exp(lambda_svensson*tau) - 2*lambda_**2*lambda_svensson**5*sigma_42*tau**2*exp(lambda_*tau) - 2*lambda_**2*lambda_svensson**4*sigma_22*tau*exp(lambda_*tau) + 4*lambda_**2*lambda_svensson**4*sigma_22*tau*exp(tau*(lambda_ + lambda_svensson)) + 12*lambda_**2*lambda_svensson**4*sigma_32*tau*exp(2*lambda_svensson*tau) - 20*lambda_**2*lambda_svensson**4*sigma_32*tau*exp(lambda_svensson*tau) + 12*lambda_**2*lambda_svensson**4*sigma_32*tau*exp(tau*(lambda_ + lambda_svensson)) - 6*lambda_**2*lambda_svensson**4*sigma_42*tau*exp(lambda_*tau) + 8*lambda_**2*lambda_svensson**4*sigma_42*tau*exp(tau*(lambda_ + lambda_svensson)) - 3*lambda_**2*lambda_svensson**3*sigma_22*exp(lambda_*tau) + 12*lambda_**2*lambda_svensson**3*sigma_22*exp(tau*(lambda_ + lambda_svensson)) + 24*lambda_**2*lambda_svensson**3*sigma_32*exp(2*lambda_svensson*tau) - 24*lambda_**2*lambda_svensson**3*sigma_32*exp(lambda_svensson*tau) + 24*lambda_**2*lambda_svensson**3*sigma_32*exp(tau*(lambda_ + lambda_svensson)) - 5*lambda_**2*lambda_svensson**3*sigma_42*exp(lambda_*tau) + 16*lambda_**2*lambda_svensson**3*sigma_42*exp(tau*(lambda_ + lambda_svensson)) + 4*lambda_*lambda_svensson**5*sigma_32*tau*exp(2*lambda_svensson*tau) - 4*lambda_*lambda_svensson**5*sigma_32*tau*exp(lambda_svensson*tau) + 4*lambda_*lambda_svensson**5*sigma_32*tau*exp(tau*(lambda_ + lambda_svensson)) + 24*lambda_*lambda_svensson**4*sigma_32*exp(2*lambda_svensson*tau) - 8*lambda_*lambda_svensson**4*sigma_32*exp(lambda_svensson*tau) + 8*lambda_*lambda_svensson**4*sigma_32*exp(tau*(lambda_ + lambda_svensson)) + 8*lambda_svensson**5*sigma_32*exp(2*lambda_svensson*tau)) - sigma_43*(-4*lambda_**5*lambda_svensson*sigma_43*tau + 11*lambda_**5*sigma_43 - 4*lambda_**4*lambda_svensson**2*sigma_33*tau - 12*lambda_**4*lambda_svensson**2*sigma_43*tau + 8*lambda_**4*lambda_svensson*sigma_33 + 33*lambda_**4*lambda_svensson*sigma_43 - 12*lambda_**3*lambda_svensson**3*sigma_33*tau - 12*lambda_**3*lambda_svensson**3*sigma_43*tau + 24*lambda_**3*lambda_svensson**2*sigma_33 + 33*lambda_**3*lambda_svensson**2*sigma_43 - 12*lambda_**2*lambda_svensson**4*sigma_33*tau - 4*lambda_**2*lambda_svensson**4*sigma_43*tau + 24*lambda_**2*lambda_svensson**3*sigma_33 + 11*lambda_**2*lambda_svensson**3*sigma_43 - 4*lambda_*lambda_svensson**5*sigma_33*tau + 24*lambda_*lambda_svensson**4*sigma_33 + 8*lambda_svensson**5*sigma_33)*exp(tau*(lambda_ + 2*lambda_svensson)) + sigma_43*(-2*lambda_**5*lambda_svensson**2*sigma_43*tau**2*exp(lambda_*tau) - 6*lambda_**5*lambda_svensson*sigma_43*tau*exp(lambda_*tau) + 8*lambda_**5*lambda_svensson*sigma_43*tau*exp(tau*(lambda_ + lambda_svensson)) - 5*lambda_**5*sigma_43*exp(lambda_*tau) + 16*lambda_**5*sigma_43*exp(tau*(lambda_ + lambda_svensson)) - 4*lambda_**4*lambda_svensson**3*sigma_33*tau**2*exp(lambda_svensson*tau) - 6*lambda_**4*lambda_svensson**3*sigma_43*tau**2*exp(lambda_*tau) + 4*lambda_**4*lambda_svensson**2*sigma_33*tau*exp(2*lambda_svensson*tau) - 4*lambda_**4*lambda_svensson**2*sigma_33*tau*exp(lambda_svensson*tau) + 4*lambda_**4*lambda_svensson**2*sigma_33*tau*exp(tau*(lambda_ + lambda_svensson)) - 18*lambda_**4*lambda_svensson**2*sigma_43*tau*exp(lambda_*tau) + 24*lambda_**4*lambda_svensson**2*sigma_43*tau*exp(tau*(lambda_ + lambda_svensson)) + 8*lambda_**4*lambda_svensson*sigma_33*exp(tau*(lambda_ + lambda_svensson)) - 15*lambda_**4*lambda_svensson*sigma_43*exp(lambda_*tau) + 48*lambda_**4*lambda_svensson*sigma_43*exp(tau*(lambda_ + lambda_svensson)) - 8*lambda_**3*lambda_svensson**4*sigma_33*tau**2*exp(lambda_svensson*tau) - 6*lambda_**3*lambda_svensson**4*sigma_43*tau**2*exp(lambda_*tau) + 12*lambda_**3*lambda_svensson**3*sigma_33*tau*exp(2*lambda_svensson*tau) - 20*lambda_**3*lambda_svensson**3*sigma_33*tau*exp(lambda_svensson*tau) + 12*lambda_**3*lambda_svensson**3*sigma_33*tau*exp(tau*(lambda_ + lambda_svensson)) - 18*lambda_**3*lambda_svensson**3*sigma_43*tau*exp(lambda_*tau) + 24*lambda_**3*lambda_svensson**3*sigma_43*tau*exp(tau*(lambda_ + lambda_svensson)) + 8*lambda_**3*lambda_svensson**2*sigma_33*exp(2*lambda_svensson*tau) - 8*lambda_**3*lambda_svensson**2*sigma_33*exp(lambda_svensson*tau) + 24*lambda_**3*lambda_svensson**2*sigma_33*exp(tau*(lambda_ + lambda_svensson)) - 15*lambda_**3*lambda_svensson**2*sigma_43*exp(lambda_*tau) + 48*lambda_**3*lambda_svensson**2*sigma_43*exp(tau*(lambda_ + lambda_svensson)) - 4*lambda_**2*lambda_svensson**5*sigma_33*tau**2*exp(lambda_svensson*tau) - 2*lambda_**2*lambda_svensson**5*sigma_43*tau**2*exp(lambda_*tau) + 12*lambda_**2*lambda_svensson**4*sigma_33*tau*exp(2*lambda_svensson*tau) - 20*lambda_**2*lambda_svensson**4*sigma_33*tau*exp(lambda_svensson*tau) + 12*lambda_**2*lambda_svensson**4*sigma_33*tau*exp(tau*(lambda_ + lambda_svensson)) - 6*lambda_**2*lambda_svensson**4*sigma_43*tau*exp(lambda_*tau) + 8*lambda_**2*lambda_svensson**4*sigma_43*tau*exp(tau*(lambda_ + lambda_svensson)) + 24*lambda_**2*lambda_svensson**3*sigma_33*exp(2*lambda_svensson*tau) - 24*lambda_**2*lambda_svensson**3*sigma_33*exp(lambda_svensson*tau) + 24*lambda_**2*lambda_svensson**3*sigma_33*exp(tau*(lambda_ + lambda_svensson)) - 5*lambda_**2*lambda_svensson**3*sigma_43*exp(lambda_*tau) + 16*lambda_**2*lambda_svensson**3*sigma_43*exp(tau*(lambda_ + lambda_svensson)) + 4*lambda_*lambda_svensson**5*sigma_33*tau*exp(2*lambda_svensson*tau) - 4*lambda_*lambda_svensson**5*sigma_33*tau*exp(lambda_svensson*tau) + 4*lambda_*lambda_svensson**5*sigma_33*tau*exp(tau*(lambda_ + lambda_svensson)) + 24*lambda_*lambda_svensson**4*sigma_33*exp(2*lambda_svensson*tau) - 8*lambda_*lambda_svensson**4*sigma_33*exp(lambda_svensson*tau) + 8*lambda_*lambda_svensson**4*sigma_33*exp(tau*(lambda_ + lambda_svensson)) + 8*lambda_svensson**5*sigma_33*exp(2*lambda_svensson*tau)))*exp(tau*(4*lambda_ + 5*lambda_svensson)) + 0.125*lambda_*(lambda_**3 + 3*lambda_**2*lambda_svensson + 3*lambda_*lambda_svensson**2 + lambda_svensson**3)*(-sigma_20*(-2*lambda_**4*lambda_svensson**2*sigma_00*tau**2 - 4*lambda_**4*lambda_svensson*sigma_20*tau - 4*lambda_**4*lambda_svensson*sigma_40*tau + 4*lambda_**4*sigma_00 + 6*lambda_**4*sigma_20 + 9*lambda_**4*sigma_40 - 4*lambda_**3*lambda_svensson**3*sigma_00*tau**2 - 4*lambda_**3*lambda_svensson**2*sigma_10*tau - 8*lambda_**3*lambda_svensson**2*sigma_20*tau - 4*lambda_**3*lambda_svensson**2*sigma_30*tau - 8*lambda_**3*lambda_svensson**2*sigma_40*tau + 8*lambda_**3*lambda_svensson*sigma_00 + 4*lambda_**3*lambda_svensson*sigma_10 + 12*lambda_**3*lambda_svensson*sigma_20 + 4*lambda_**3*lambda_svensson*sigma_30 + 18*lambda_**3*lambda_svensson*sigma_40 - 2*lambda_**2*lambda_svensson**4*sigma_00*tau**2 - 8*lambda_**2*lambda_svensson**3*sigma_10*tau - 4*lambda_**2*lambda_svensson**3*sigma_20*tau - 8*lambda_**2*lambda_svensson**3*sigma_30*tau - 4*lambda_**2*lambda_svensson**3*sigma_40*tau + 4*lambda_**2*lambda_svensson**2*sigma_00 + 8*lambda_**2*lambda_svensson**2*sigma_10 + 6*lambda_**2*lambda_svensson**2*sigma_20 + 8*lambda_**2*lambda_svensson**2*sigma_30 + 9*lambda_**2*lambda_svensson**2*sigma_40 - 4*lambda_*lambda_svensson**4*sigma_10*tau - 4*lambda_*lambda_svensson**4*sigma_30*tau + 8*lambda_*lambda_svensson**3*sigma_10 + 16*lambda_*lambda_svensson**3*sigma_30 + 4*lambda_svensson**4*sigma_10 + 8*lambda_svensson**4*sigma_30)*exp(tau*(lambda_ + 2*lambda_svensson)) + sigma_20*(4*lambda_**4*lambda_svensson*sigma_00*tau*exp(tau*(lambda_ + lambda_svensson)) - 2*lambda_**4*lambda_svensson*sigma_40*tau*exp(lambda_*tau) + 4*lambda_**4*lambda_svensson*sigma_40*tau*exp(tau*(lambda_ + lambda_svensson)) + 4*lambda_**4*sigma_00*exp(tau*(lambda_ + lambda_svensson)) - 2*lambda_**4*sigma_20*exp(lambda_*tau) + 8*lambda_**4*sigma_20*exp(tau*(lambda_ + lambda_svensson)) - 3*lambda_**4*sigma_40*exp(lambda_*tau) + 12*lambda_**4*sigma_40*exp(tau*(lambda_ + lambda_svensson)) + 8*lambda_**3*lambda_svensson**2*sigma_00*tau*exp(tau*(lambda_ + lambda_svensson)) + 4*lambda_**3*lambda_svensson**2*sigma_30*tau*exp(2*lambda_svensson*tau) - 4*lambda_**3*lambda_svensson**2*sigma_30*tau*exp(lambda_svensson*tau) - 4*lambda_**3*lambda_svensson**2*sigma_40*tau*exp(lambda_*tau) + 8*lambda_**3*lambda_svensson**2*sigma_40*tau*exp(tau*(lambda_ + lambda_svensson)) + 8*lambda_**3*lambda_svensson*sigma_00*exp(tau*(lambda_ + lambda_svensson)) + 4*lambda_**3*lambda_svensson*sigma_10*exp(tau*(lambda_ + lambda_svensson)) - 4*lambda_**3*lambda_svensson*sigma_20*exp(lambda_*tau) + 16*lambda_**3*lambda_svensson*sigma_20*exp(tau*(lambda_ + lambda_svensson)) + 4*lambda_**3*lambda_svensson*sigma_30*exp(tau*(lambda_ + lambda_svensson)) - 6*lambda_**3*lambda_svensson*sigma_40*exp(lambda_*tau) + 24*lambda_**3*lambda_svensson*sigma_40*exp(tau*(lambda_ + lambda_svensson)) + 4*lambda_**2*lambda_svensson**3*sigma_00*tau*exp(tau*(lambda_ + lambda_svensson)) + 8*lambda_**2*lambda_svensson**3*sigma_30*tau*exp(2*lambda_svensson*tau) - 4*lambda_**2*lambda_svensson**3*sigma_30*tau*exp(lambda_svensson*tau) - 2*lambda_**2*lambda_svensson**3*sigma_40*tau*exp(lambda_*tau) + 4*lambda_**2*lambda_svensson**3*sigma_40*tau*exp(tau*(lambda_ + lambda_svensson)) + 4*lambda_**2*lambda_svensson**2*sigma_00*exp(tau*(lambda_ + lambda_svensson)) + 4*lambda_**2*lambda_svensson**2*sigma_10*exp(2*lambda_svensson*tau) - 4*lambda_**2*lambda_svensson**2*sigma_10*exp(lambda_svensson*tau) + 8*lambda_**2*lambda_svensson**2*sigma_10*exp(tau*(lambda_ + lambda_svensson)) - 2*lambda_**2*lambda_svensson**2*sigma_20*exp(lambda_*tau) + 8*lambda_**2*lambda_svensson**2*sigma_20*exp(tau*(lambda_ + lambda_svensson)) + 8*lambda_**2*lambda_svensson**2*sigma_30*exp(2*lambda_svensson*tau) - 8*lambda_**2*lambda_svensson**2*sigma_30*exp(lambda_svensson*tau) + 8*lambda_**2*lambda_svensson**2*sigma_30*exp(tau*(lambda_ + lambda_svensson)) - 3*lambda_**2*lambda_svensson**2*sigma_40*exp(lambda_*tau) + 12*lambda_**2*lambda_svensson**2*sigma_40*exp(tau*(lambda_ + lambda_svensson)) + 4*lambda_*lambda_svensson**4*sigma_30*tau*exp(2*lambda_svensson*tau) + 8*lambda_*lambda_svensson**3*sigma_10*exp(2*lambda_svensson*tau) - 4*lambda_*lambda_svensson**3*sigma_10*exp(lambda_svensson*tau) + 4*lambda_*lambda_svensson**3*sigma_10*exp(tau*(lambda_ + lambda_svensson)) + 16*lambda_*lambda_svensson**3*sigma_30*exp(2*lambda_svensson*tau) - 4*lambda_*lambda_svensson**3*sigma_30*exp(lambda_svensson*tau) + 4*lambda_*lambda_svensson**3*sigma_30*exp(tau*(lambda_ + lambda_svensson)) + 4*lambda_svensson**4*sigma_10*exp(2*lambda_svensson*tau) + 8*lambda_svensson**4*sigma_30*exp(2*lambda_svensson*tau)) - sigma_21*(-4*lambda_**4*lambda_svensson*sigma_21*tau - 4*lambda_**4*lambda_svensson*sigma_41*tau + 6*lambda_**4*sigma_21 + 9*lambda_**4*sigma_41 - 4*lambda_**3*lambda_svensson**2*sigma_11*tau - 8*lambda_**3*lambda_svensson**2*sigma_21*tau - 4*lambda_**3*lambda_svensson**2*sigma_31*tau - 8*lambda_**3*lambda_svensson**2*sigma_41*tau + 4*lambda_**3*lambda_svensson*sigma_11 + 12*lambda_**3*lambda_svensson*sigma_21 + 4*lambda_**3*lambda_svensson*sigma_31 + 18*lambda_**3*lambda_svensson*sigma_41 - 8*lambda_**2*lambda_svensson**3*sigma_11*tau - 4*lambda_**2*lambda_svensson**3*sigma_21*tau - 8*lambda_**2*lambda_svensson**3*sigma_31*tau - 4*lambda_**2*lambda_svensson**3*sigma_41*tau + 8*lambda_**2*lambda_svensson**2*sigma_11 + 6*lambda_**2*lambda_svensson**2*sigma_21 + 8*lambda_**2*lambda_svensson**2*sigma_31 + 9*lambda_**2*lambda_svensson**2*sigma_41 - 4*lambda_*lambda_svensson**4*sigma_11*tau - 4*lambda_*lambda_svensson**4*sigma_31*tau + 8*lambda_*lambda_svensson**3*sigma_11 + 16*lambda_*lambda_svensson**3*sigma_31 + 4*lambda_svensson**4*sigma_11 + 8*lambda_svensson**4*sigma_31)*exp(tau*(lambda_ + 2*lambda_svensson)) + sigma_21*(-2*lambda_**4*lambda_svensson*sigma_41*tau*exp(lambda_*tau) + 4*lambda_**4*lambda_svensson*sigma_41*tau*exp(tau*(lambda_ + lambda_svensson)) - 2*lambda_**4*sigma_21*exp(lambda_*tau) + 8*lambda_**4*sigma_21*exp(tau*(lambda_ + lambda_svensson)) - 3*lambda_**4*sigma_41*exp(lambda_*tau) + 12*lambda_**4*sigma_41*exp(tau*(lambda_ + lambda_svensson)) + 4*lambda_**3*lambda_svensson**2*sigma_31*tau*exp(2*lambda_svensson*tau) - 4*lambda_**3*lambda_svensson**2*sigma_31*tau*exp(lambda_svensson*tau) - 4*lambda_**3*lambda_svensson**2*sigma_41*tau*exp(lambda_*tau) + 8*lambda_**3*lambda_svensson**2*sigma_41*tau*exp(tau*(lambda_ + lambda_svensson)) + 4*lambda_**3*lambda_svensson*sigma_11*exp(tau*(lambda_ + lambda_svensson)) - 4*lambda_**3*lambda_svensson*sigma_21*exp(lambda_*tau) + 16*lambda_**3*lambda_svensson*sigma_21*exp(tau*(lambda_ + lambda_svensson)) + 4*lambda_**3*lambda_svensson*sigma_31*exp(tau*(lambda_ + lambda_svensson)) - 6*lambda_**3*lambda_svensson*sigma_41*exp(lambda_*tau) + 24*lambda_**3*lambda_svensson*sigma_41*exp(tau*(lambda_ + lambda_svensson)) + 8*lambda_**2*lambda_svensson**3*sigma_31*tau*exp(2*lambda_svensson*tau) - 4*lambda_**2*lambda_svensson**3*sigma_31*tau*exp(lambda_svensson*tau) - 2*lambda_**2*lambda_svensson**3*sigma_41*tau*exp(lambda_*tau) + 4*lambda_**2*lambda_svensson**3*sigma_41*tau*exp(tau*(lambda_ + lambda_svensson)) + 4*lambda_**2*lambda_svensson**2*sigma_11*exp(2*lambda_svensson*tau) - 4*lambda_**2*lambda_svensson**2*sigma_11*exp(lambda_svensson*tau) + 8*lambda_**2*lambda_svensson**2*sigma_11*exp(tau*(lambda_ + lambda_svensson)) - 2*lambda_**2*lambda_svensson**2*sigma_21*exp(lambda_*tau) + 8*lambda_**2*lambda_svensson**2*sigma_21*exp(tau*(lambda_ + lambda_svensson)) + 8*lambda_**2*lambda_svensson**2*sigma_31*exp(2*lambda_svensson*tau) - 8*lambda_**2*lambda_svensson**2*sigma_31*exp(lambda_svensson*tau) + 8*lambda_**2*lambda_svensson**2*sigma_31*exp(tau*(lambda_ + lambda_svensson)) - 3*lambda_**2*lambda_svensson**2*sigma_41*exp(lambda_*tau) + 12*lambda_**2*lambda_svensson**2*sigma_41*exp(tau*(lambda_ + lambda_svensson)) + 4*lambda_*lambda_svensson**4*sigma_31*tau*exp(2*lambda_svensson*tau) + 8*lambda_*lambda_svensson**3*sigma_11*exp(2*lambda_svensson*tau) - 4*lambda_*lambda_svensson**3*sigma_11*exp(lambda_svensson*tau) + 4*lambda_*lambda_svensson**3*sigma_11*exp(tau*(lambda_ + lambda_svensson)) + 16*lambda_*lambda_svensson**3*sigma_31*exp(2*lambda_svensson*tau) - 4*lambda_*lambda_svensson**3*sigma_31*exp(lambda_svensson*tau) + 4*lambda_*lambda_svensson**3*sigma_31*exp(tau*(lambda_ + lambda_svensson)) + 4*lambda_svensson**4*sigma_11*exp(2*lambda_svensson*tau) + 8*lambda_svensson**4*sigma_31*exp(2*lambda_svensson*tau)) - sigma_22*(-4*lambda_**4*lambda_svensson*sigma_22*tau - 4*lambda_**4*lambda_svensson*sigma_42*tau + 6*lambda_**4*sigma_22 + 9*lambda_**4*sigma_42 - 8*lambda_**3*lambda_svensson**2*sigma_22*tau - 4*lambda_**3*lambda_svensson**2*sigma_32*tau - 8*lambda_**3*lambda_svensson**2*sigma_42*tau + 12*lambda_**3*lambda_svensson*sigma_22 + 4*lambda_**3*lambda_svensson*sigma_32 + 18*lambda_**3*lambda_svensson*sigma_42 - 4*lambda_**2*lambda_svensson**3*sigma_22*tau - 8*lambda_**2*lambda_svensson**3*sigma_32*tau - 4*lambda_**2*lambda_svensson**3*sigma_42*tau + 6*lambda_**2*lambda_svensson**2*sigma_22 + 8*lambda_**2*lambda_svensson**2*sigma_32 + 9*lambda_**2*lambda_svensson**2*sigma_42 - 4*lambda_*lambda_svensson**4*sigma_32*tau + 16*lambda_*lambda_svensson**3*sigma_32 + 8*lambda_svensson**4*sigma_32)*exp(tau*(lambda_ + 2*lambda_svensson)) + sigma_22*(-2*lambda_**4*lambda_svensson*sigma_42*tau*exp(lambda_*tau) + 4*lambda_**4*lambda_svensson*sigma_42*tau*exp(tau*(lambda_ + lambda_svensson)) - 2*lambda_**4*sigma_22*exp(lambda_*tau) + 8*lambda_**4*sigma_22*exp(tau*(lambda_ + lambda_svensson)) - 3*lambda_**4*sigma_42*exp(lambda_*tau) + 12*lambda_**4*sigma_42*exp(tau*(lambda_ + lambda_svensson)) + 4*lambda_**3*lambda_svensson**2*sigma_32*tau*exp(2*lambda_svensson*tau) - 4*lambda_**3*lambda_svensson**2*sigma_32*tau*exp(lambda_svensson*tau) - 4*lambda_**3*lambda_svensson**2*sigma_42*tau*exp(lambda_*tau) + 8*lambda_**3*lambda_svensson**2*sigma_42*tau*exp(tau*(lambda_ + lambda_svensson)) - 4*lambda_**3*lambda_svensson*sigma_22*exp(lambda_*tau) + 16*lambda_**3*lambda_svensson*sigma_22*exp(tau*(lambda_ + lambda_svensson)) + 4*lambda_**3*lambda_svensson*sigma_32*exp(tau*(lambda_ + lambda_svensson)) - 6*lambda_**3*lambda_svensson*sigma_42*exp(lambda_*tau) + 24*lambda_**3*lambda_svensson*sigma_42*exp(tau*(lambda_ + lambda_svensson)) + 8*lambda_**2*lambda_svensson**3*sigma_32*tau*exp(2*lambda_svensson*tau) - 4*lambda_**2*lambda_svensson**3*sigma_32*tau*exp(lambda_svensson*tau) - 2*lambda_**2*lambda_svensson**3*sigma_42*tau*exp(lambda_*tau) + 4*lambda_**2*lambda_svensson**3*sigma_42*tau*exp(tau*(lambda_ + lambda_svensson)) - 2*lambda_**2*lambda_svensson**2*sigma_22*exp(lambda_*tau) + 8*lambda_**2*lambda_svensson**2*sigma_22*exp(tau*(lambda_ + lambda_svensson)) + 8*lambda_**2*lambda_svensson**2*sigma_32*exp(2*lambda_svensson*tau) - 8*lambda_**2*lambda_svensson**2*sigma_32*exp(lambda_svensson*tau) + 8*lambda_**2*lambda_svensson**2*sigma_32*exp(tau*(lambda_ + lambda_svensson)) - 3*lambda_**2*lambda_svensson**2*sigma_42*exp(lambda_*tau) + 12*lambda_**2*lambda_svensson**2*sigma_42*exp(tau*(lambda_ + lambda_svensson)) + 4*lambda_*lambda_svensson**4*sigma_32*tau*exp(2*lambda_svensson*tau) + 16*lambda_*lambda_svensson**3*sigma_32*exp(2*lambda_svensson*tau) - 4*lambda_*lambda_svensson**3*sigma_32*exp(lambda_svensson*tau) + 4*lambda_*lambda_svensson**3*sigma_32*exp(tau*(lambda_ + lambda_svensson)) + 8*lambda_svensson**4*sigma_32*exp(2*lambda_svensson*tau)))*exp(tau*(4*lambda_ + 5*lambda_svensson)) + 0.125*lambda_svensson*(lambda_**2 + 2*lambda_*lambda_svensson + lambda_svensson**2)*(-sigma_30*(-2*lambda_**5*lambda_svensson**2*sigma_00*tau**2 - 4*lambda_**5*lambda_svensson*sigma_20*tau - 4*lambda_**5*lambda_svensson*sigma_40*tau + 4*lambda_**5*sigma_20 + 8*lambda_**5*sigma_40 - 6*lambda_**4*lambda_svensson**3*sigma_00*tau**2 - 4*lambda_**4*lambda_svensson**2*sigma_10*tau - 12*lambda_**4*lambda_svensson**2*sigma_20*tau - 4*lambda_**4*lambda_svensson**2*sigma_30*tau - 12*lambda_**4*lambda_svensson**2*sigma_40*tau + 12*lambda_**4*lambda_svensson*sigma_20 + 24*lambda_**4*lambda_svensson*sigma_40 - 6*lambda_**3*lambda_svensson**4*sigma_00*tau**2 - 12*lambda_**3*lambda_svensson**3*sigma_10*tau - 12*lambda_**3*lambda_svensson**3*sigma_20*tau - 12*lambda_**3*lambda_svensson**3*sigma_30*tau - 12*lambda_**3*lambda_svensson**3*sigma_40*tau + 12*lambda_**3*lambda_svensson**2*sigma_00 + 9*lambda_**3*lambda_svensson**2*sigma_10 + 24*lambda_**3*lambda_svensson**2*sigma_20 + 11*lambda_**3*lambda_svensson**2*sigma_30 + 24*lambda_**3*lambda_svensson**2*sigma_40 - 2*lambda_**2*lambda_svensson**5*sigma_00*tau**2 - 12*lambda_**2*lambda_svensson**4*sigma_10*tau - 4*lambda_**2*lambda_svensson**4*sigma_20*tau - 12*lambda_**2*lambda_svensson**4*sigma_30*tau - 4*lambda_**2*lambda_svensson**4*sigma_40*tau + 36*lambda_**2*lambda_svensson**3*sigma_00 + 27*lambda_**2*lambda_svensson**3*sigma_10 + 24*lambda_**2*lambda_svensson**3*sigma_20 + 33*lambda_**2*lambda_svensson**3*sigma_30 + 24*lambda_**2*lambda_svensson**3*sigma_40 - 4*lambda_*lambda_svensson**5*sigma_10*tau - 4*lambda_*lambda_svensson**5*sigma_30*tau + 36*lambda_*lambda_svensson**4*sigma_00 + 27*lambda_*lambda_svensson**4*sigma_10 + 8*lambda_*lambda_svensson**4*sigma_20 + 33*lambda_*lambda_svensson**4*sigma_30 + 8*lambda_*lambda_svensson**4*sigma_40 + 12*lambda_svensson**5*sigma_00 + 9*lambda_svensson**5*sigma_10 + 11*lambda_svensson**5*sigma_30)*exp(tau*(2*lambda_ + lambda_svensson)) + sigma_30*(4*lambda_**5*lambda_svensson**2*sigma_00*tau**2*exp(tau*(lambda_ + lambda_svensson)) - 2*lambda_**5*lambda_svensson**2*sigma_30*tau**2*exp(lambda_svensson*tau) - 4*lambda_**5*lambda_svensson**2*sigma_40*tau**2*exp(lambda_*tau) - 4*lambda_**5*lambda_svensson*sigma_20*tau*exp(lambda_*tau) + 4*lambda_**5*lambda_svensson*sigma_20*tau*exp(tau*(lambda_ + lambda_svensson)) + 4*lambda_**5*lambda_svensson*sigma_40*tau*exp(2*lambda_*tau) - 4*lambda_**5*lambda_svensson*sigma_40*tau*exp(lambda_*tau) + 4*lambda_**5*lambda_svensson*sigma_40*tau*exp(tau*(lambda_ + lambda_svensson)) + 4*lambda_**5*sigma_20*exp(2*lambda_*tau) + 8*lambda_**5*sigma_40*exp(2*lambda_*tau) + 12*lambda_**4*lambda_svensson**3*sigma_00*tau**2*exp(tau*(lambda_ + lambda_svensson)) - 6*lambda_**4*lambda_svensson**3*sigma_30*tau**2*exp(lambda_svensson*tau) - 8*lambda_**4*lambda_svensson**3*sigma_40*tau**2*exp(lambda_*tau) + 12*lambda_**4*lambda_svensson**2*sigma_00*tau*exp(tau*(lambda_ + lambda_svensson)) - 2*lambda_**4*lambda_svensson**2*sigma_10*tau*exp(lambda_svensson*tau) + 4*lambda_**4*lambda_svensson**2*sigma_10*tau*exp(tau*(lambda_ + lambda_svensson)) - 8*lambda_**4*lambda_svensson**2*sigma_20*tau*exp(lambda_*tau) + 12*lambda_**4*lambda_svensson**2*sigma_20*tau*exp(tau*(lambda_ + lambda_svensson)) - 6*lambda_**4*lambda_svensson**2*sigma_30*tau*exp(lambda_svensson*tau) + 8*lambda_**4*lambda_svensson**2*sigma_30*tau*exp(tau*(lambda_ + lambda_svensson)) + 12*lambda_**4*lambda_svensson**2*sigma_40*tau*exp(2*lambda_*tau) - 20*lambda_**4*lambda_svensson**2*sigma_40*tau*exp(lambda_*tau) + 12*lambda_**4*lambda_svensson**2*sigma_40*tau*exp(tau*(lambda_ + lambda_svensson)) + 12*lambda_**4*lambda_svensson*sigma_20*exp(2*lambda_*tau) - 8*lambda_**4*lambda_svensson*sigma_20*exp(lambda_*tau) + 8*lambda_**4*lambda_svensson*sigma_20*exp(tau*(lambda_ + lambda_svensson)) + 24*lambda_**4*lambda_svensson*sigma_40*exp(2*lambda_*tau) - 8*lambda_**4*lambda_svensson*sigma_40*exp(lambda_*tau) + 8*lambda_**4*lambda_svensson*sigma_40*exp(tau*(lambda_ + lambda_svensson)) + 12*lambda_**3*lambda_svensson**4*sigma_00*tau**2*exp(tau*(lambda_ + lambda_svensson)) - 6*lambda_**3*lambda_svensson**4*sigma_30*tau**2*exp(lambda_svensson*tau) - 4*lambda_**3*lambda_svensson**4*sigma_40*tau**2*exp(lambda_*tau) + 36*lambda_**3*lambda_svensson**3*sigma_00*tau*exp(tau*(lambda_ + lambda_svensson)) - 6*lambda_**3*lambda_svensson**3*sigma_10*tau*exp(lambda_svensson*tau) + 12*lambda_**3*lambda_svensson**3*sigma_10*tau*exp(tau*(lambda_ + lambda_svensson)) - 4*lambda_**3*lambda_svensson**3*sigma_20*tau*exp(lambda_*tau) + 12*lambda_**3*lambda_svensson**3*sigma_20*tau*exp(tau*(lambda_ + lambda_svensson)) - 18*lambda_**3*lambda_svensson**3*sigma_30*tau*exp(lambda_svensson*tau) + 24*lambda_**3*lambda_svensson**3*sigma_30*tau*exp(tau*(lambda_ + lambda_svensson)) + 12*lambda_**3*lambda_svensson**3*sigma_40*tau*exp(2*lambda_*tau) - 20*lambda_**3*lambda_svensson**3*sigma_40*tau*exp(lambda_*tau) + 12*lambda_**3*lambda_svensson**3*sigma_40*tau*exp(tau*(lambda_ + lambda_svensson)) + 12*lambda_**3*lambda_svensson**2*sigma_00*exp(tau*(lambda_ + lambda_svensson)) - 3*lambda_**3*lambda_svensson**2*sigma_10*exp(lambda_svensson*tau) + 12*lambda_**3*lambda_svensson**2*sigma_10*exp(tau*(lambda_ + lambda_svensson)) + 12*lambda_**3*lambda_svensson**2*sigma_20*exp(2*lambda_*tau) - 12*lambda_**3*lambda_svensson**2*sigma_20*exp(lambda_*tau) + 24*lambda_**3*lambda_svensson**2*sigma_20*exp(tau*(lambda_ + lambda_svensson)) - 5*lambda_**3*lambda_svensson**2*sigma_30*exp(lambda_svensson*tau) + 16*lambda_**3*lambda_svensson**2*sigma_30*exp(tau*(lambda_ + lambda_svensson)) + 24*lambda_**3*lambda_svensson**2*sigma_40*exp(2*lambda_*tau) - 24*lambda_**3*lambda_svensson**2*sigma_40*exp(lambda_*tau) + 24*lambda_**3*lambda_svensson**2*sigma_40*exp(tau*(lambda_ + lambda_svensson)) + 4*lambda_**2*lambda_svensson**5*sigma_00*tau**2*exp(tau*(lambda_ + lambda_svensson)) - 2*lambda_**2*lambda_svensson**5*sigma_30*tau**2*exp(lambda_svensson*tau) + 36*lambda_**2*lambda_svensson**4*sigma_00*tau*exp(tau*(lambda_ + lambda_svensson)) - 6*lambda_**2*lambda_svensson**4*sigma_10*tau*exp(lambda_svensson*tau) + 12*lambda_**2*lambda_svensson**4*sigma_10*tau*exp(tau*(lambda_ + lambda_svensson)) + 4*lambda_**2*lambda_svensson**4*sigma_20*tau*exp(tau*(lambda_ + lambda_svensson)) - 18*lambda_**2*lambda_svensson**4*sigma_30*tau*exp(lambda_svensson*tau) + 24*lambda_**2*lambda_svensson**4*sigma_30*tau*exp(tau*(lambda_ + lambda_svensson)) + 4*lambda_**2*lambda_svensson**4*sigma_40*tau*exp(2*lambda_*tau) - 4*lambda_**2*lambda_svensson**4*sigma_40*tau*exp(lambda_*tau) + 4*lambda_**2*lambda_svensson**4*sigma_40*tau*exp(tau*(lambda_ + lambda_svensson)) + 36*lambda_**2*lambda_svensson**3*sigma_00*exp(tau*(lambda_ + lambda_svensson)) - 9*lambda_**2*lambda_svensson**3*sigma_10*exp(lambda_svensson*tau) + 36*lambda_**2*lambda_svensson**3*sigma_10*exp(tau*(lambda_ + lambda_svensson)) + 4*lambda_**2*lambda_svensson**3*sigma_20*exp(2*lambda_*tau) - 4*lambda_**2*lambda_svensson**3*sigma_20*exp(lambda_*tau) + 24*lambda_**2*lambda_svensson**3*sigma_20*exp(tau*(lambda_ + lambda_svensson)) - 15*lambda_**2*lambda_svensson**3*sigma_30*exp(lambda_svensson*tau) + 48*lambda_**2*lambda_svensson**3*sigma_30*exp(tau*(lambda_ + lambda_svensson)) + 8*lambda_**2*lambda_svensson**3*sigma_40*exp(2*lambda_*tau) - 8*lambda_**2*lambda_svensson**3*sigma_40*exp(lambda_*tau) + 24*lambda_**2*lambda_svensson**3*sigma_40*exp(tau*(lambda_ + lambda_svensson)) + 12*lambda_*lambda_svensson**5*sigma_00*tau*exp(tau*(lambda_ + lambda_svensson)) - 2*lambda_*lambda_svensson**5*sigma_10*tau*exp(lambda_svensson*tau) + 4*lambda_*lambda_svensson**5*sigma_10*tau*exp(tau*(lambda_ + lambda_svensson)) - 6*lambda_*lambda_svensson**5*sigma_30*tau*exp(lambda_svensson*tau) + 8*lambda_*lambda_svensson**5*sigma_30*tau*exp(tau*(lambda_ + lambda_svensson)) + 36*lambda_*lambda_svensson**4*sigma_00*exp(tau*(lambda_ + lambda_svensson)) - 9*lambda_*lambda_svensson**4*sigma_10*exp(lambda_svensson*tau) + 36*lambda_*lambda_svensson**4*sigma_10*exp(tau*(lambda_ + lambda_svensson)) + 8*lambda_*lambda_svensson**4*sigma_20*exp(tau*(lambda_ + lambda_svensson)) - 15*lambda_*lambda_svensson**4*sigma_30*exp(lambda_svensson*tau) + 48*lambda_*lambda_svensson**4*sigma_30*exp(tau*(lambda_ + lambda_svensson)) + 8*lambda_*lambda_svensson**4*sigma_40*exp(tau*(lambda_ + lambda_svensson)) + 12*lambda_svensson**5*sigma_00*exp(tau*(lambda_ + lambda_svensson)) - 3*lambda_svensson**5*sigma_10*exp(lambda_svensson*tau) + 12*lambda_svensson**5*sigma_10*exp(tau*(lambda_ + lambda_svensson)) - 5*lambda_svensson**5*sigma_30*exp(lambda_svensson*tau) + 16*lambda_svensson**5*sigma_30*exp(tau*(lambda_ + lambda_svensson))) - sigma_31*(-4*lambda_**5*lambda_svensson*sigma_21*tau - 4*lambda_**5*lambda_svensson*sigma_41*tau + 4*lambda_**5*sigma_21 + 8*lambda_**5*sigma_41 - 4*lambda_**4*lambda_svensson**2*sigma_11*tau - 12*lambda_**4*lambda_svensson**2*sigma_21*tau - 4*lambda_**4*lambda_svensson**2*sigma_31*tau - 12*lambda_**4*lambda_svensson**2*sigma_41*tau + 12*lambda_**4*lambda_svensson*sigma_21 + 24*lambda_**4*lambda_svensson*sigma_41 - 12*lambda_**3*lambda_svensson**3*sigma_11*tau - 12*lambda_**3*lambda_svensson**3*sigma_21*tau - 12*lambda_**3*lambda_svensson**3*sigma_31*tau - 12*lambda_**3*lambda_svensson**3*sigma_41*tau + 9*lambda_**3*lambda_svensson**2*sigma_11 + 24*lambda_**3*lambda_svensson**2*sigma_21 + 11*lambda_**3*lambda_svensson**2*sigma_31 + 24*lambda_**3*lambda_svensson**2*sigma_41 - 12*lambda_**2*lambda_svensson**4*sigma_11*tau - 4*lambda_**2*lambda_svensson**4*sigma_21*tau - 12*lambda_**2*lambda_svensson**4*sigma_31*tau - 4*lambda_**2*lambda_svensson**4*sigma_41*tau + 27*lambda_**2*lambda_svensson**3*sigma_11 + 24*lambda_**2*lambda_svensson**3*sigma_21 + 33*lambda_**2*lambda_svensson**3*sigma_31 + 24*lambda_**2*lambda_svensson**3*sigma_41 - 4*lambda_*lambda_svensson**5*sigma_11*tau - 4*lambda_*lambda_svensson**5*sigma_31*tau + 27*lambda_*lambda_svensson**4*sigma_11 + 8*lambda_*lambda_svensson**4*sigma_21 + 33*lambda_*lambda_svensson**4*sigma_31 + 8*lambda_*lambda_svensson**4*sigma_41 + 9*lambda_svensson**5*sigma_11 + 11*lambda_svensson**5*sigma_31)*exp(tau*(2*lambda_ + lambda_svensson)) + sigma_31*(-2*lambda_**5*lambda_svensson**2*sigma_31*tau**2*exp(lambda_svensson*tau) - 4*lambda_**5*lambda_svensson**2*sigma_41*tau**2*exp(lambda_*tau) - 4*lambda_**5*lambda_svensson*sigma_21*tau*exp(lambda_*tau) + 4*lambda_**5*lambda_svensson*sigma_21*tau*exp(tau*(lambda_ + lambda_svensson)) + 4*lambda_**5*lambda_svensson*sigma_41*tau*exp(2*lambda_*tau) - 4*lambda_**5*lambda_svensson*sigma_41*tau*exp(lambda_*tau) + 4*lambda_**5*lambda_svensson*sigma_41*tau*exp(tau*(lambda_ + lambda_svensson)) + 4*lambda_**5*sigma_21*exp(2*lambda_*tau) + 8*lambda_**5*sigma_41*exp(2*lambda_*tau) - 6*lambda_**4*lambda_svensson**3*sigma_31*tau**2*exp(lambda_svensson*tau) - 8*lambda_**4*lambda_svensson**3*sigma_41*tau**2*exp(lambda_*tau) - 2*lambda_**4*lambda_svensson**2*sigma_11*tau*exp(lambda_svensson*tau) + 4*lambda_**4*lambda_svensson**2*sigma_11*tau*exp(tau*(lambda_ + lambda_svensson)) - 8*lambda_**4*lambda_svensson**2*sigma_21*tau*exp(lambda_*tau) + 12*lambda_**4*lambda_svensson**2*sigma_21*tau*exp(tau*(lambda_ + lambda_svensson)) - 6*lambda_**4*lambda_svensson**2*sigma_31*tau*exp(lambda_svensson*tau) + 8*lambda_**4*lambda_svensson**2*sigma_31*tau*exp(tau*(lambda_ + lambda_svensson)) + 12*lambda_**4*lambda_svensson**2*sigma_41*tau*exp(2*lambda_*tau) - 20*lambda_**4*lambda_svensson**2*sigma_41*tau*exp(lambda_*tau) + 12*lambda_**4*lambda_svensson**2*sigma_41*tau*exp(tau*(lambda_ + lambda_svensson)) + 12*lambda_**4*lambda_svensson*sigma_21*exp(2*lambda_*tau) - 8*lambda_**4*lambda_svensson*sigma_21*exp(lambda_*tau) + 8*lambda_**4*lambda_svensson*sigma_21*exp(tau*(lambda_ + lambda_svensson)) + 24*lambda_**4*lambda_svensson*sigma_41*exp(2*lambda_*tau) - 8*lambda_**4*lambda_svensson*sigma_41*exp(lambda_*tau) + 8*lambda_**4*lambda_svensson*sigma_41*exp(tau*(lambda_ + lambda_svensson)) - 6*lambda_**3*lambda_svensson**4*sigma_31*tau**2*exp(lambda_svensson*tau) - 4*lambda_**3*lambda_svensson**4*sigma_41*tau**2*exp(lambda_*tau) - 6*lambda_**3*lambda_svensson**3*sigma_11*tau*exp(lambda_svensson*tau) + 12*lambda_**3*lambda_svensson**3*sigma_11*tau*exp(tau*(lambda_ + lambda_svensson)) - 4*lambda_**3*lambda_svensson**3*sigma_21*tau*exp(lambda_*tau) + 12*lambda_**3*lambda_svensson**3*sigma_21*tau*exp(tau*(lambda_ + lambda_svensson)) - 18*lambda_**3*lambda_svensson**3*sigma_31*tau*exp(lambda_svensson*tau) + 24*lambda_**3*lambda_svensson**3*sigma_31*tau*exp(tau*(lambda_ + lambda_svensson)) + 12*lambda_**3*lambda_svensson**3*sigma_41*tau*exp(2*lambda_*tau) - 20*lambda_**3*lambda_svensson**3*sigma_41*tau*exp(lambda_*tau) + 12*lambda_**3*lambda_svensson**3*sigma_41*tau*exp(tau*(lambda_ + lambda_svensson)) - 3*lambda_**3*lambda_svensson**2*sigma_11*exp(lambda_svensson*tau) + 12*lambda_**3*lambda_svensson**2*sigma_11*exp(tau*(lambda_ + lambda_svensson)) + 12*lambda_**3*lambda_svensson**2*sigma_21*exp(2*lambda_*tau) - 12*lambda_**3*lambda_svensson**2*sigma_21*exp(lambda_*tau) + 24*lambda_**3*lambda_svensson**2*sigma_21*exp(tau*(lambda_ + lambda_svensson)) - 5*lambda_**3*lambda_svensson**2*sigma_31*exp(lambda_svensson*tau) + 16*lambda_**3*lambda_svensson**2*sigma_31*exp(tau*(lambda_ + lambda_svensson)) + 24*lambda_**3*lambda_svensson**2*sigma_41*exp(2*lambda_*tau) - 24*lambda_**3*lambda_svensson**2*sigma_41*exp(lambda_*tau) + 24*lambda_**3*lambda_svensson**2*sigma_41*exp(tau*(lambda_ + lambda_svensson)) - 2*lambda_**2*lambda_svensson**5*sigma_31*tau**2*exp(lambda_svensson*tau) - 6*lambda_**2*lambda_svensson**4*sigma_11*tau*exp(lambda_svensson*tau) + 12*lambda_**2*lambda_svensson**4*sigma_11*tau*exp(tau*(lambda_ + lambda_svensson)) + 4*lambda_**2*lambda_svensson**4*sigma_21*tau*exp(tau*(lambda_ + lambda_svensson)) - 18*lambda_**2*lambda_svensson**4*sigma_31*tau*exp(lambda_svensson*tau) + 24*lambda_**2*lambda_svensson**4*sigma_31*tau*exp(tau*(lambda_ + lambda_svensson)) + 4*lambda_**2*lambda_svensson**4*sigma_41*tau*exp(2*lambda_*tau) - 4*lambda_**2*lambda_svensson**4*sigma_41*tau*exp(lambda_*tau) + 4*lambda_**2*lambda_svensson**4*sigma_41*tau*exp(tau*(lambda_ + lambda_svensson)) - 9*lambda_**2*lambda_svensson**3*sigma_11*exp(lambda_svensson*tau) + 36*lambda_**2*lambda_svensson**3*sigma_11*exp(tau*(lambda_ + lambda_svensson)) + 4*lambda_**2*lambda_svensson**3*sigma_21*exp(2*lambda_*tau) - 4*lambda_**2*lambda_svensson**3*sigma_21*exp(lambda_*tau) + 24*lambda_**2*lambda_svensson**3*sigma_21*exp(tau*(lambda_ + lambda_svensson)) - 15*lambda_**2*lambda_svensson**3*sigma_31*exp(lambda_svensson*tau) + 48*lambda_**2*lambda_svensson**3*sigma_31*exp(tau*(lambda_ + lambda_svensson)) + 8*lambda_**2*lambda_svensson**3*sigma_41*exp(2*lambda_*tau) - 8*lambda_**2*lambda_svensson**3*sigma_41*exp(lambda_*tau) + 24*lambda_**2*lambda_svensson**3*sigma_41*exp(tau*(lambda_ + lambda_svensson)) - 2*lambda_*lambda_svensson**5*sigma_11*tau*exp(lambda_svensson*tau) + 4*lambda_*lambda_svensson**5*sigma_11*tau*exp(tau*(lambda_ + lambda_svensson)) - 6*lambda_*lambda_svensson**5*sigma_31*tau*exp(lambda_svensson*tau) + 8*lambda_*lambda_svensson**5*sigma_31*tau*exp(tau*(lambda_ + lambda_svensson)) - 9*lambda_*lambda_svensson**4*sigma_11*exp(lambda_svensson*tau) + 36*lambda_*lambda_svensson**4*sigma_11*exp(tau*(lambda_ + lambda_svensson)) + 8*lambda_*lambda_svensson**4*sigma_21*exp(tau*(lambda_ + lambda_svensson)) - 15*lambda_*lambda_svensson**4*sigma_31*exp(lambda_svensson*tau) + 48*lambda_*lambda_svensson**4*sigma_31*exp(tau*(lambda_ + lambda_svensson)) + 8*lambda_*lambda_svensson**4*sigma_41*exp(tau*(lambda_ + lambda_svensson)) - 3*lambda_svensson**5*sigma_11*exp(lambda_svensson*tau) + 12*lambda_svensson**5*sigma_11*exp(tau*(lambda_ + lambda_svensson)) - 5*lambda_svensson**5*sigma_31*exp(lambda_svensson*tau) + 16*lambda_svensson**5*sigma_31*exp(tau*(lambda_ + lambda_svensson))) - sigma_32*(-4*lambda_**5*lambda_svensson*sigma_22*tau - 4*lambda_**5*lambda_svensson*sigma_42*tau + 4*lambda_**5*sigma_22 + 8*lambda_**5*sigma_42 - 12*lambda_**4*lambda_svensson**2*sigma_22*tau - 4*lambda_**4*lambda_svensson**2*sigma_32*tau - 12*lambda_**4*lambda_svensson**2*sigma_42*tau + 12*lambda_**4*lambda_svensson*sigma_22 + 24*lambda_**4*lambda_svensson*sigma_42 - 12*lambda_**3*lambda_svensson**3*sigma_22*tau - 12*lambda_**3*lambda_svensson**3*sigma_32*tau - 12*lambda_**3*lambda_svensson**3*sigma_42*tau + 24*lambda_**3*lambda_svensson**2*sigma_22 + 11*lambda_**3*lambda_svensson**2*sigma_32 + 24*lambda_**3*lambda_svensson**2*sigma_42 - 4*lambda_**2*lambda_svensson**4*sigma_22*tau - 12*lambda_**2*lambda_svensson**4*sigma_32*tau - 4*lambda_**2*lambda_svensson**4*sigma_42*tau + 24*lambda_**2*lambda_svensson**3*sigma_22 + 33*lambda_**2*lambda_svensson**3*sigma_32 + 24*lambda_**2*lambda_svensson**3*sigma_42 - 4*lambda_*lambda_svensson**5*sigma_32*tau + 8*lambda_*lambda_svensson**4*sigma_22 + 33*lambda_*lambda_svensson**4*sigma_32 + 8*lambda_*lambda_svensson**4*sigma_42 + 11*lambda_svensson**5*sigma_32)*exp(tau*(2*lambda_ + lambda_svensson)) + sigma_32*(-2*lambda_**5*lambda_svensson**2*sigma_32*tau**2*exp(lambda_svensson*tau) - 4*lambda_**5*lambda_svensson**2*sigma_42*tau**2*exp(lambda_*tau) - 4*lambda_**5*lambda_svensson*sigma_22*tau*exp(lambda_*tau) + 4*lambda_**5*lambda_svensson*sigma_22*tau*exp(tau*(lambda_ + lambda_svensson)) + 4*lambda_**5*lambda_svensson*sigma_42*tau*exp(2*lambda_*tau) - 4*lambda_**5*lambda_svensson*sigma_42*tau*exp(lambda_*tau) + 4*lambda_**5*lambda_svensson*sigma_42*tau*exp(tau*(lambda_ + lambda_svensson)) + 4*lambda_**5*sigma_22*exp(2*lambda_*tau) + 8*lambda_**5*sigma_42*exp(2*lambda_*tau) - 6*lambda_**4*lambda_svensson**3*sigma_32*tau**2*exp(lambda_svensson*tau) - 8*lambda_**4*lambda_svensson**3*sigma_42*tau**2*exp(lambda_*tau) - 8*lambda_**4*lambda_svensson**2*sigma_22*tau*exp(lambda_*tau) + 12*lambda_**4*lambda_svensson**2*sigma_22*tau*exp(tau*(lambda_ + lambda_svensson)) - 6*lambda_**4*lambda_svensson**2*sigma_32*tau*exp(lambda_svensson*tau) + 8*lambda_**4*lambda_svensson**2*sigma_32*tau*exp(tau*(lambda_ + lambda_svensson)) + 12*lambda_**4*lambda_svensson**2*sigma_42*tau*exp(2*lambda_*tau) - 20*lambda_**4*lambda_svensson**2*sigma_42*tau*exp(lambda_*tau) + 12*lambda_**4*lambda_svensson**2*sigma_42*tau*exp(tau*(lambda_ + lambda_svensson)) + 12*lambda_**4*lambda_svensson*sigma_22*exp(2*lambda_*tau) - 8*lambda_**4*lambda_svensson*sigma_22*exp(lambda_*tau) + 8*lambda_**4*lambda_svensson*sigma_22*exp(tau*(lambda_ + lambda_svensson)) + 24*lambda_**4*lambda_svensson*sigma_42*exp(2*lambda_*tau) - 8*lambda_**4*lambda_svensson*sigma_42*exp(lambda_*tau) + 8*lambda_**4*lambda_svensson*sigma_42*exp(tau*(lambda_ + lambda_svensson)) - 6*lambda_**3*lambda_svensson**4*sigma_32*tau**2*exp(lambda_svensson*tau) - 4*lambda_**3*lambda_svensson**4*sigma_42*tau**2*exp(lambda_*tau) - 4*lambda_**3*lambda_svensson**3*sigma_22*tau*exp(lambda_*tau) + 12*lambda_**3*lambda_svensson**3*sigma_22*tau*exp(tau*(lambda_ + lambda_svensson)) - 18*lambda_**3*lambda_svensson**3*sigma_32*tau*exp(lambda_svensson*tau) + 24*lambda_**3*lambda_svensson**3*sigma_32*tau*exp(tau*(lambda_ + lambda_svensson)) + 12*lambda_**3*lambda_svensson**3*sigma_42*tau*exp(2*lambda_*tau) - 20*lambda_**3*lambda_svensson**3*sigma_42*tau*exp(lambda_*tau) + 12*lambda_**3*lambda_svensson**3*sigma_42*tau*exp(tau*(lambda_ + lambda_svensson)) + 12*lambda_**3*lambda_svensson**2*sigma_22*exp(2*lambda_*tau) - 12*lambda_**3*lambda_svensson**2*sigma_22*exp(lambda_*tau) + 24*lambda_**3*lambda_svensson**2*sigma_22*exp(tau*(lambda_ + lambda_svensson)) - 5*lambda_**3*lambda_svensson**2*sigma_32*exp(lambda_svensson*tau) + 16*lambda_**3*lambda_svensson**2*sigma_32*exp(tau*(lambda_ + lambda_svensson)) + 24*lambda_**3*lambda_svensson**2*sigma_42*exp(2*lambda_*tau) - 24*lambda_**3*lambda_svensson**2*sigma_42*exp(lambda_*tau) + 24*lambda_**3*lambda_svensson**2*sigma_42*exp(tau*(lambda_ + lambda_svensson)) - 2*lambda_**2*lambda_svensson**5*sigma_32*tau**2*exp(lambda_svensson*tau) + 4*lambda_**2*lambda_svensson**4*sigma_22*tau*exp(tau*(lambda_ + lambda_svensson)) - 18*lambda_**2*lambda_svensson**4*sigma_32*tau*exp(lambda_svensson*tau) + 24*lambda_**2*lambda_svensson**4*sigma_32*tau*exp(tau*(lambda_ + lambda_svensson)) + 4*lambda_**2*lambda_svensson**4*sigma_42*tau*exp(2*lambda_*tau) - 4*lambda_**2*lambda_svensson**4*sigma_42*tau*exp(lambda_*tau) + 4*lambda_**2*lambda_svensson**4*sigma_42*tau*exp(tau*(lambda_ + lambda_svensson)) + 4*lambda_**2*lambda_svensson**3*sigma_22*exp(2*lambda_*tau) - 4*lambda_**2*lambda_svensson**3*sigma_22*exp(lambda_*tau) + 24*lambda_**2*lambda_svensson**3*sigma_22*exp(tau*(lambda_ + lambda_svensson)) - 15*lambda_**2*lambda_svensson**3*sigma_32*exp(lambda_svensson*tau) + 48*lambda_**2*lambda_svensson**3*sigma_32*exp(tau*(lambda_ + lambda_svensson)) + 8*lambda_**2*lambda_svensson**3*sigma_42*exp(2*lambda_*tau) - 8*lambda_**2*lambda_svensson**3*sigma_42*exp(lambda_*tau) + 24*lambda_**2*lambda_svensson**3*sigma_42*exp(tau*(lambda_ + lambda_svensson)) - 6*lambda_*lambda_svensson**5*sigma_32*tau*exp(lambda_svensson*tau) + 8*lambda_*lambda_svensson**5*sigma_32*tau*exp(tau*(lambda_ + lambda_svensson)) + 8*lambda_*lambda_svensson**4*sigma_22*exp(tau*(lambda_ + lambda_svensson)) - 15*lambda_*lambda_svensson**4*sigma_32*exp(lambda_svensson*tau) + 48*lambda_*lambda_svensson**4*sigma_32*exp(tau*(lambda_ + lambda_svensson)) + 8*lambda_*lambda_svensson**4*sigma_42*exp(tau*(lambda_ + lambda_svensson)) - 5*lambda_svensson**5*sigma_32*exp(lambda_svensson*tau) + 16*lambda_svensson**5*sigma_32*exp(tau*(lambda_ + lambda_svensson))) - sigma_33*(-4*lambda_**5*lambda_svensson*sigma_43*tau + 8*lambda_**5*sigma_43 - 4*lambda_**4*lambda_svensson**2*sigma_33*tau - 12*lambda_**4*lambda_svensson**2*sigma_43*tau + 24*lambda_**4*lambda_svensson*sigma_43 - 12*lambda_**3*lambda_svensson**3*sigma_33*tau - 12*lambda_**3*lambda_svensson**3*sigma_43*tau + 11*lambda_**3*lambda_svensson**2*sigma_33 + 24*lambda_**3*lambda_svensson**2*sigma_43 - 12*lambda_**2*lambda_svensson**4*sigma_33*tau - 4*lambda_**2*lambda_svensson**4*sigma_43*tau + 33*lambda_**2*lambda_svensson**3*sigma_33 + 24*lambda_**2*lambda_svensson**3*sigma_43 - 4*lambda_*lambda_svensson**5*sigma_33*tau + 33*lambda_*lambda_svensson**4*sigma_33 + 8*lambda_*lambda_svensson**4*sigma_43 + 11*lambda_svensson**5*sigma_33)*exp(tau*(2*lambda_ + lambda_svensson)) + sigma_33*(-2*lambda_**5*lambda_svensson**2*sigma_33*tau**2*exp(lambda_svensson*tau) - 4*lambda_**5*lambda_svensson**2*sigma_43*tau**2*exp(lambda_*tau) + 4*lambda_**5*lambda_svensson*sigma_43*tau*exp(2*lambda_*tau) - 4*lambda_**5*lambda_svensson*sigma_43*tau*exp(lambda_*tau) + 4*lambda_**5*lambda_svensson*sigma_43*tau*exp(tau*(lambda_ + lambda_svensson)) + 8*lambda_**5*sigma_43*exp(2*lambda_*tau) - 6*lambda_**4*lambda_svensson**3*sigma_33*tau**2*exp(lambda_svensson*tau) - 8*lambda_**4*lambda_svensson**3*sigma_43*tau**2*exp(lambda_*tau) - 6*lambda_**4*lambda_svensson**2*sigma_33*tau*exp(lambda_svensson*tau) + 8*lambda_**4*lambda_svensson**2*sigma_33*tau*exp(tau*(lambda_ + lambda_svensson)) + 12*lambda_**4*lambda_svensson**2*sigma_43*tau*exp(2*lambda_*tau) - 20*lambda_**4*lambda_svensson**2*sigma_43*tau*exp(lambda_*tau) + 12*lambda_**4*lambda_svensson**2*sigma_43*tau*exp(tau*(lambda_ + lambda_svensson)) + 24*lambda_**4*lambda_svensson*sigma_43*exp(2*lambda_*tau) - 8*lambda_**4*lambda_svensson*sigma_43*exp(lambda_*tau) + 8*lambda_**4*lambda_svensson*sigma_43*exp(tau*(lambda_ + lambda_svensson)) - 6*lambda_**3*lambda_svensson**4*sigma_33*tau**2*exp(lambda_svensson*tau) - 4*lambda_**3*lambda_svensson**4*sigma_43*tau**2*exp(lambda_*tau) - 18*lambda_**3*lambda_svensson**3*sigma_33*tau*exp(lambda_svensson*tau) + 24*lambda_**3*lambda_svensson**3*sigma_33*tau*exp(tau*(lambda_ + lambda_svensson)) + 12*lambda_**3*lambda_svensson**3*sigma_43*tau*exp(2*lambda_*tau) - 20*lambda_**3*lambda_svensson**3*sigma_43*tau*exp(lambda_*tau) + 12*lambda_**3*lambda_svensson**3*sigma_43*tau*exp(tau*(lambda_ + lambda_svensson)) - 5*lambda_**3*lambda_svensson**2*sigma_33*exp(lambda_svensson*tau) + 16*lambda_**3*lambda_svensson**2*sigma_33*exp(tau*(lambda_ + lambda_svensson)) + 24*lambda_**3*lambda_svensson**2*sigma_43*exp(2*lambda_*tau) - 24*lambda_**3*lambda_svensson**2*sigma_43*exp(lambda_*tau) + 24*lambda_**3*lambda_svensson**2*sigma_43*exp(tau*(lambda_ + lambda_svensson)) - 2*lambda_**2*lambda_svensson**5*sigma_33*tau**2*exp(lambda_svensson*tau) - 18*lambda_**2*lambda_svensson**4*sigma_33*tau*exp(lambda_svensson*tau) + 24*lambda_**2*lambda_svensson**4*sigma_33*tau*exp(tau*(lambda_ + lambda_svensson)) + 4*lambda_**2*lambda_svensson**4*sigma_43*tau*exp(2*lambda_*tau) - 4*lambda_**2*lambda_svensson**4*sigma_43*tau*exp(lambda_*tau) + 4*lambda_**2*lambda_svensson**4*sigma_43*tau*exp(tau*(lambda_ + lambda_svensson)) - 15*lambda_**2*lambda_svensson**3*sigma_33*exp(lambda_svensson*tau) + 48*lambda_**2*lambda_svensson**3*sigma_33*exp(tau*(lambda_ + lambda_svensson)) + 8*lambda_**2*lambda_svensson**3*sigma_43*exp(2*lambda_*tau) - 8*lambda_**2*lambda_svensson**3*sigma_43*exp(lambda_*tau) + 24*lambda_**2*lambda_svensson**3*sigma_43*exp(tau*(lambda_ + lambda_svensson)) - 6*lambda_*lambda_svensson**5*sigma_33*tau*exp(lambda_svensson*tau) + 8*lambda_*lambda_svensson**5*sigma_33*tau*exp(tau*(lambda_ + lambda_svensson)) - 15*lambda_*lambda_svensson**4*sigma_33*exp(lambda_svensson*tau) + 48*lambda_*lambda_svensson**4*sigma_33*exp(tau*(lambda_ + lambda_svensson)) + 8*lambda_*lambda_svensson**4*sigma_43*exp(tau*(lambda_ + lambda_svensson)) - 5*lambda_svensson**5*sigma_33*exp(lambda_svensson*tau) + 16*lambda_svensson**5*sigma_33*exp(tau*(lambda_ + lambda_svensson))))*exp(3*tau*(lambda_ + 2*lambda_svensson)) + 0.125*lambda_svensson*(lambda_**3 + 3*lambda_**2*lambda_svensson + 3*lambda_*lambda_svensson**2 + lambda_svensson**3)*(-sigma_10*(-2*lambda_**4*lambda_svensson**2*sigma_00*tau**2 - 4*lambda_**4*lambda_svensson*sigma_20*tau - 4*lambda_**4*lambda_svensson*sigma_40*tau + 4*lambda_**4*sigma_20 + 8*lambda_**4*sigma_40 - 4*lambda_**3*lambda_svensson**3*sigma_00*tau**2 - 4*lambda_**3*lambda_svensson**2*sigma_10*tau - 8*lambda_**3*lambda_svensson**2*sigma_20*tau - 4*lambda_**3*lambda_svensson**2*sigma_30*tau - 8*lambda_**3*lambda_svensson**2*sigma_40*tau + 8*lambda_**3*lambda_svensson*sigma_20 + 16*lambda_**3*lambda_svensson*sigma_40 - 2*lambda_**2*lambda_svensson**4*sigma_00*tau**2 - 8*lambda_**2*lambda_svensson**3*sigma_10*tau - 4*lambda_**2*lambda_svensson**3*sigma_20*tau - 8*lambda_**2*lambda_svensson**3*sigma_30*tau - 4*lambda_**2*lambda_svensson**3*sigma_40*tau + 4*lambda_**2*lambda_svensson**2*sigma_00 + 6*lambda_**2*lambda_svensson**2*sigma_10 + 8*lambda_**2*lambda_svensson**2*sigma_20 + 9*lambda_**2*lambda_svensson**2*sigma_30 + 8*lambda_**2*lambda_svensson**2*sigma_40 - 4*lambda_*lambda_svensson**4*sigma_10*tau - 4*lambda_*lambda_svensson**4*sigma_30*tau + 8*lambda_*lambda_svensson**3*sigma_00 + 12*lambda_*lambda_svensson**3*sigma_10 + 4*lambda_*lambda_svensson**3*sigma_20 + 18*lambda_*lambda_svensson**3*sigma_30 + 4*lambda_*lambda_svensson**3*sigma_40 + 4*lambda_svensson**4*sigma_00 + 6*lambda_svensson**4*sigma_10 + 9*lambda_svensson**4*sigma_30)*exp(tau*(2*lambda_ + lambda_svensson)) + sigma_10*(4*lambda_**4*lambda_svensson*sigma_40*tau*exp(2*lambda_*tau) + 4*lambda_**4*sigma_20*exp(2*lambda_*tau) + 8*lambda_**4*sigma_40*exp(2*lambda_*tau) + 4*lambda_**3*lambda_svensson**2*sigma_00*tau*exp(tau*(lambda_ + lambda_svensson)) - 2*lambda_**3*lambda_svensson**2*sigma_30*tau*exp(lambda_svensson*tau) + 4*lambda_**3*lambda_svensson**2*sigma_30*tau*exp(tau*(lambda_ + lambda_svensson)) + 8*lambda_**3*lambda_svensson**2*sigma_40*tau*exp(2*lambda_*tau) - 4*lambda_**3*lambda_svensson**2*sigma_40*tau*exp(lambda_*tau) + 8*lambda_**3*lambda_svensson*sigma_20*exp(2*lambda_*tau) - 4*lambda_**3*lambda_svensson*sigma_20*exp(lambda_*tau) + 4*lambda_**3*lambda_svensson*sigma_20*exp(tau*(lambda_ + lambda_svensson)) + 16*lambda_**3*lambda_svensson*sigma_40*exp(2*lambda_*tau) - 4*lambda_**3*lambda_svensson*sigma_40*exp(lambda_*tau) + 4*lambda_**3*lambda_svensson*sigma_40*exp(tau*(lambda_ + lambda_svensson)) + 8*lambda_**2*lambda_svensson**3*sigma_00*tau*exp(tau*(lambda_ + lambda_svensson)) - 4*lambda_**2*lambda_svensson**3*sigma_30*tau*exp(lambda_svensson*tau) + 8*lambda_**2*lambda_svensson**3*sigma_30*tau*exp(tau*(lambda_ + lambda_svensson)) + 4*lambda_**2*lambda_svensson**3*sigma_40*tau*exp(2*lambda_*tau) - 4*lambda_**2*lambda_svensson**3*sigma_40*tau*exp(lambda_*tau) + 4*lambda_**2*lambda_svensson**2*sigma_00*exp(tau*(lambda_ + lambda_svensson)) - 2*lambda_**2*lambda_svensson**2*sigma_10*exp(lambda_svensson*tau) + 8*lambda_**2*lambda_svensson**2*sigma_10*exp(tau*(lambda_ + lambda_svensson)) + 4*lambda_**2*lambda_svensson**2*sigma_20*exp(2*lambda_*tau) - 4*lambda_**2*lambda_svensson**2*sigma_20*exp(lambda_*tau) + 8*lambda_**2*lambda_svensson**2*sigma_20*exp(tau*(lambda_ + lambda_svensson)) - 3*lambda_**2*lambda_svensson**2*sigma_30*exp(lambda_svensson*tau) + 12*lambda_**2*lambda_svensson**2*sigma_30*exp(tau*(lambda_ + lambda_svensson)) + 8*lambda_**2*lambda_svensson**2*sigma_40*exp(2*lambda_*tau) - 8*lambda_**2*lambda_svensson**2*sigma_40*exp(lambda_*tau) + 8*lambda_**2*lambda_svensson**2*sigma_40*exp(tau*(lambda_ + lambda_svensson)) + 4*lambda_*lambda_svensson**4*sigma_00*tau*exp(tau*(lambda_ + lambda_svensson)) - 2*lambda_*lambda_svensson**4*sigma_30*tau*exp(lambda_svensson*tau) + 4*lambda_*lambda_svensson**4*sigma_30*tau*exp(tau*(lambda_ + lambda_svensson)) + 8*lambda_*lambda_svensson**3*sigma_00*exp(tau*(lambda_ + lambda_svensson)) - 4*lambda_*lambda_svensson**3*sigma_10*exp(lambda_svensson*tau) + 16*lambda_*lambda_svensson**3*sigma_10*exp(tau*(lambda_ + lambda_svensson)) + 4*lambda_*lambda_svensson**3*sigma_20*exp(tau*(lambda_ + lambda_svensson)) - 6*lambda_*lambda_svensson**3*sigma_30*exp(lambda_svensson*tau) + 24*lambda_*lambda_svensson**3*sigma_30*exp(tau*(lambda_ + lambda_svensson)) + 4*lambda_*lambda_svensson**3*sigma_40*exp(tau*(lambda_ + lambda_svensson)) + 4*lambda_svensson**4*sigma_00*exp(tau*(lambda_ + lambda_svensson)) - 2*lambda_svensson**4*sigma_10*exp(lambda_svensson*tau) + 8*lambda_svensson**4*sigma_10*exp(tau*(lambda_ + lambda_svensson)) - 3*lambda_svensson**4*sigma_30*exp(lambda_svensson*tau) + 12*lambda_svensson**4*sigma_30*exp(tau*(lambda_ + lambda_svensson))) - sigma_11*(-4*lambda_**4*lambda_svensson*sigma_21*tau - 4*lambda_**4*lambda_svensson*sigma_41*tau + 4*lambda_**4*sigma_21 + 8*lambda_**4*sigma_41 - 4*lambda_**3*lambda_svensson**2*sigma_11*tau - 8*lambda_**3*lambda_svensson**2*sigma_21*tau - 4*lambda_**3*lambda_svensson**2*sigma_31*tau - 8*lambda_**3*lambda_svensson**2*sigma_41*tau + 8*lambda_**3*lambda_svensson*sigma_21 + 16*lambda_**3*lambda_svensson*sigma_41 - 8*lambda_**2*lambda_svensson**3*sigma_11*tau - 4*lambda_**2*lambda_svensson**3*sigma_21*tau - 8*lambda_**2*lambda_svensson**3*sigma_31*tau - 4*lambda_**2*lambda_svensson**3*sigma_41*tau + 6*lambda_**2*lambda_svensson**2*sigma_11 + 8*lambda_**2*lambda_svensson**2*sigma_21 + 9*lambda_**2*lambda_svensson**2*sigma_31 + 8*lambda_**2*lambda_svensson**2*sigma_41 - 4*lambda_*lambda_svensson**4*sigma_11*tau - 4*lambda_*lambda_svensson**4*sigma_31*tau + 12*lambda_*lambda_svensson**3*sigma_11 + 4*lambda_*lambda_svensson**3*sigma_21 + 18*lambda_*lambda_svensson**3*sigma_31 + 4*lambda_*lambda_svensson**3*sigma_41 + 6*lambda_svensson**4*sigma_11 + 9*lambda_svensson**4*sigma_31)*exp(tau*(2*lambda_ + lambda_svensson)) + sigma_11*(4*lambda_**4*lambda_svensson*sigma_41*tau*exp(2*lambda_*tau) + 4*lambda_**4*sigma_21*exp(2*lambda_*tau) + 8*lambda_**4*sigma_41*exp(2*lambda_*tau) - 2*lambda_**3*lambda_svensson**2*sigma_31*tau*exp(lambda_svensson*tau) + 4*lambda_**3*lambda_svensson**2*sigma_31*tau*exp(tau*(lambda_ + lambda_svensson)) + 8*lambda_**3*lambda_svensson**2*sigma_41*tau*exp(2*lambda_*tau) - 4*lambda_**3*lambda_svensson**2*sigma_41*tau*exp(lambda_*tau) + 8*lambda_**3*lambda_svensson*sigma_21*exp(2*lambda_*tau) - 4*lambda_**3*lambda_svensson*sigma_21*exp(lambda_*tau) + 4*lambda_**3*lambda_svensson*sigma_21*exp(tau*(lambda_ + lambda_svensson)) + 16*lambda_**3*lambda_svensson*sigma_41*exp(2*lambda_*tau) - 4*lambda_**3*lambda_svensson*sigma_41*exp(lambda_*tau) + 4*lambda_**3*lambda_svensson*sigma_41*exp(tau*(lambda_ + lambda_svensson)) - 4*lambda_**2*lambda_svensson**3*sigma_31*tau*exp(lambda_svensson*tau) + 8*lambda_**2*lambda_svensson**3*sigma_31*tau*exp(tau*(lambda_ + lambda_svensson)) + 4*lambda_**2*lambda_svensson**3*sigma_41*tau*exp(2*lambda_*tau) - 4*lambda_**2*lambda_svensson**3*sigma_41*tau*exp(lambda_*tau) - 2*lambda_**2*lambda_svensson**2*sigma_11*exp(lambda_svensson*tau) + 8*lambda_**2*lambda_svensson**2*sigma_11*exp(tau*(lambda_ + lambda_svensson)) + 4*lambda_**2*lambda_svensson**2*sigma_21*exp(2*lambda_*tau) - 4*lambda_**2*lambda_svensson**2*sigma_21*exp(lambda_*tau) + 8*lambda_**2*lambda_svensson**2*sigma_21*exp(tau*(lambda_ + lambda_svensson)) - 3*lambda_**2*lambda_svensson**2*sigma_31*exp(lambda_svensson*tau) + 12*lambda_**2*lambda_svensson**2*sigma_31*exp(tau*(lambda_ + lambda_svensson)) + 8*lambda_**2*lambda_svensson**2*sigma_41*exp(2*lambda_*tau) - 8*lambda_**2*lambda_svensson**2*sigma_41*exp(lambda_*tau) + 8*lambda_**2*lambda_svensson**2*sigma_41*exp(tau*(lambda_ + lambda_svensson)) - 2*lambda_*lambda_svensson**4*sigma_31*tau*exp(lambda_svensson*tau) + 4*lambda_*lambda_svensson**4*sigma_31*tau*exp(tau*(lambda_ + lambda_svensson)) - 4*lambda_*lambda_svensson**3*sigma_11*exp(lambda_svensson*tau) + 16*lambda_*lambda_svensson**3*sigma_11*exp(tau*(lambda_ + lambda_svensson)) + 4*lambda_*lambda_svensson**3*sigma_21*exp(tau*(lambda_ + lambda_svensson)) - 6*lambda_*lambda_svensson**3*sigma_31*exp(lambda_svensson*tau) + 24*lambda_*lambda_svensson**3*sigma_31*exp(tau*(lambda_ + lambda_svensson)) + 4*lambda_*lambda_svensson**3*sigma_41*exp(tau*(lambda_ + lambda_svensson)) - 2*lambda_svensson**4*sigma_11*exp(lambda_svensson*tau) + 8*lambda_svensson**4*sigma_11*exp(tau*(lambda_ + lambda_svensson)) - 3*lambda_svensson**4*sigma_31*exp(lambda_svensson*tau) + 12*lambda_svensson**4*sigma_31*exp(tau*(lambda_ + lambda_svensson))))*exp(3*tau*(lambda_ + 2*lambda_svensson)) + 0.0833333333333333*sigma_00*(lambda_**2 + 2*lambda_*lambda_svensson + lambda_svensson**2)*(lambda_**3 + 3*lambda_**2*lambda_svensson + 3*lambda_*lambda_svensson**2 + lambda_svensson**3)*(2*lambda_**3*lambda_svensson**3*sigma_00*tau**3 + 3*lambda_**3*lambda_svensson**2*tau**2*(sigma_20 + sigma_40) - 6*lambda_**3*(sigma_20 + 3*sigma_40) + 3*lambda_**2*lambda_svensson**3*tau**2*(sigma_10 + sigma_30) - 6*lambda_svensson**3*(sigma_10 + 3*sigma_30))*exp(tau*(5*lambda_ + 7*lambda_svensson)) + 0.5*sigma_00*(lambda_**2 + 2*lambda_*lambda_svensson + lambda_svensson**2)*(lambda_**3 + 3*lambda_**2*lambda_svensson + 3*lambda_*lambda_svensson**2 + lambda_svensson**3)*(lambda_**3*lambda_svensson**2*sigma_40*tau**2*exp(tau*(3*lambda_ + 2*lambda_svensson)) + lambda_**3*lambda_svensson*tau*(sigma_20 + 3*sigma_40)*exp(tau*(3*lambda_ + 2*lambda_svensson)) + lambda_**3*(sigma_20 + 3*sigma_40)*exp(tau*(3*lambda_ + 2*lambda_svensson)) + lambda_**2*lambda_svensson**3*sigma_30*tau**2*exp(tau*(2*lambda_ + 3*lambda_svensson)) + lambda_*lambda_svensson**3*tau*(sigma_10 + 3*sigma_30)*exp(tau*(2*lambda_ + 3*lambda_svensson)) + lambda_svensson**3*(sigma_10 + 3*sigma_30)*exp(tau*(2*lambda_ + 3*lambda_svensson)))*exp(2*tau*(lambda_ + 2*lambda_svensson)))*exp(-tau*(5*lambda_ + 7*lambda_svensson))/(lambda_**3*lambda_svensson**3*(lambda_**2 + 2*lambda_*lambda_svensson + lambda_svensson**2)*(lambda_**3 + 3*lambda_**2*lambda_svensson + 3*lambda_*lambda_svensson**2 + lambda_svensson**3))
    return 9.0 * (1.0 * lambda_ + 1.0 * lambda_svensson) ** 2 * (
                0.0185185185185185 * lambda_ ** 6 * lambda_svensson ** 3 * sigma_00 ** 2 * tau ** 3 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.0555555555555556 * lambda_ ** 6 * lambda_svensson ** 2 * sigma_00 * sigma_20 * tau ** 2 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.0555555555555556 * lambda_ ** 6 * lambda_svensson ** 2 * sigma_00 * sigma_40 * tau ** 2 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.111111111111111 * lambda_ ** 6 * lambda_svensson ** 2 * sigma_00 * sigma_40 * tau ** 2 * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.0277777777777778 * lambda_ ** 6 * lambda_svensson ** 2 * sigma_40 ** 2 * tau ** 2 * exp(
            2 * lambda_ * tau) - 0.0277777777777778 * lambda_ ** 6 * lambda_svensson ** 2 * sigma_41 ** 2 * tau ** 2 * exp(
            2 * lambda_ * tau) - 0.0277777777777778 * lambda_ ** 6 * lambda_svensson ** 2 * sigma_42 ** 2 * tau ** 2 * exp(
            2 * lambda_ * tau) - 0.0277777777777778 * lambda_ ** 6 * lambda_svensson ** 2 * sigma_43 ** 2 * tau ** 2 * exp(
            2 * lambda_ * tau) - 0.0277777777777778 * lambda_ ** 6 * lambda_svensson ** 2 * sigma_44 ** 2 * tau ** 2 * exp(
            2 * lambda_ * tau) + 0.111111111111111 * lambda_ ** 6 * lambda_svensson * sigma_00 * sigma_20 * tau * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) + 0.333333333333333 * lambda_ ** 6 * lambda_svensson * sigma_00 * sigma_40 * tau * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) + 0.0555555555555556 * lambda_ ** 6 * lambda_svensson * sigma_20 ** 2 * tau * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.111111111111111 * lambda_ ** 6 * lambda_svensson * sigma_20 * sigma_40 * tau * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.111111111111111 * lambda_ ** 6 * lambda_svensson * sigma_20 * sigma_40 * tau * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.0555555555555556 * lambda_ ** 6 * lambda_svensson * sigma_20 * sigma_40 * tau * exp(
            2 * lambda_ * tau) + 0.0555555555555556 * lambda_ ** 6 * lambda_svensson * sigma_21 ** 2 * tau * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.111111111111111 * lambda_ ** 6 * lambda_svensson * sigma_21 * sigma_41 * tau * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.111111111111111 * lambda_ ** 6 * lambda_svensson * sigma_21 * sigma_41 * tau * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.0555555555555556 * lambda_ ** 6 * lambda_svensson * sigma_21 * sigma_41 * tau * exp(
            2 * lambda_ * tau) + 0.0555555555555556 * lambda_ ** 6 * lambda_svensson * sigma_22 ** 2 * tau * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.111111111111111 * lambda_ ** 6 * lambda_svensson * sigma_22 * sigma_42 * tau * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.111111111111111 * lambda_ ** 6 * lambda_svensson * sigma_22 * sigma_42 * tau * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.0555555555555556 * lambda_ ** 6 * lambda_svensson * sigma_22 * sigma_42 * tau * exp(
            2 * lambda_ * tau) + 0.0555555555555556 * lambda_ ** 6 * lambda_svensson * sigma_40 ** 2 * tau * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.111111111111111 * lambda_ ** 6 * lambda_svensson * sigma_40 ** 2 * tau * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.0833333333333333 * lambda_ ** 6 * lambda_svensson * sigma_40 ** 2 * tau * exp(
            2 * lambda_ * tau) + 0.0555555555555556 * lambda_ ** 6 * lambda_svensson * sigma_41 ** 2 * tau * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.111111111111111 * lambda_ ** 6 * lambda_svensson * sigma_41 ** 2 * tau * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.0833333333333333 * lambda_ ** 6 * lambda_svensson * sigma_41 ** 2 * tau * exp(
            2 * lambda_ * tau) + 0.0555555555555556 * lambda_ ** 6 * lambda_svensson * sigma_42 ** 2 * tau * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.111111111111111 * lambda_ ** 6 * lambda_svensson * sigma_42 ** 2 * tau * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.0833333333333333 * lambda_ ** 6 * lambda_svensson * sigma_42 ** 2 * tau * exp(
            2 * lambda_ * tau) + 0.0555555555555556 * lambda_ ** 6 * lambda_svensson * sigma_43 ** 2 * tau * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.111111111111111 * lambda_ ** 6 * lambda_svensson * sigma_43 ** 2 * tau * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.0833333333333333 * lambda_ ** 6 * lambda_svensson * sigma_43 ** 2 * tau * exp(
            2 * lambda_ * tau) + 0.0555555555555556 * lambda_ ** 6 * lambda_svensson * sigma_44 ** 2 * tau * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.111111111111111 * lambda_ ** 6 * lambda_svensson * sigma_44 ** 2 * tau * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.0833333333333333 * lambda_ ** 6 * lambda_svensson * sigma_44 ** 2 * tau * exp(
            2 * lambda_ * tau) - 0.111111111111111 * lambda_ ** 6 * sigma_00 * sigma_20 * exp(2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.111111111111111 * lambda_ ** 6 * sigma_00 * sigma_20 * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.333333333333333 * lambda_ ** 6 * sigma_00 * sigma_40 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.333333333333333 * lambda_ ** 6 * sigma_00 * sigma_40 * exp(
            2 * lambda_ * tau) * exp(lambda_svensson * tau) - 0.0833333333333333 * lambda_ ** 6 * sigma_20 ** 2 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.111111111111111 * lambda_ ** 6 * sigma_20 ** 2 * exp(
            2 * lambda_ * tau) * exp(lambda_svensson * tau) - 0.0277777777777778 * lambda_ ** 6 * sigma_20 ** 2 * exp(
            2 * lambda_ * tau) - 0.25 * lambda_ ** 6 * sigma_20 * sigma_40 * exp(2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.333333333333333 * lambda_ ** 6 * sigma_20 * sigma_40 * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.0833333333333333 * lambda_ ** 6 * sigma_20 * sigma_40 * exp(
            2 * lambda_ * tau) - 0.0833333333333333 * lambda_ ** 6 * sigma_21 ** 2 * exp(2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.111111111111111 * lambda_ ** 6 * sigma_21 ** 2 * exp(
            2 * lambda_ * tau) * exp(lambda_svensson * tau) - 0.0277777777777778 * lambda_ ** 6 * sigma_21 ** 2 * exp(
            2 * lambda_ * tau) - 0.25 * lambda_ ** 6 * sigma_21 * sigma_41 * exp(2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.333333333333333 * lambda_ ** 6 * sigma_21 * sigma_41 * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.0833333333333333 * lambda_ ** 6 * sigma_21 * sigma_41 * exp(
            2 * lambda_ * tau) - 0.0833333333333333 * lambda_ ** 6 * sigma_22 ** 2 * exp(2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.111111111111111 * lambda_ ** 6 * sigma_22 ** 2 * exp(
            2 * lambda_ * tau) * exp(lambda_svensson * tau) - 0.0277777777777778 * lambda_ ** 6 * sigma_22 ** 2 * exp(
            2 * lambda_ * tau) - 0.25 * lambda_ ** 6 * sigma_22 * sigma_42 * exp(2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.333333333333333 * lambda_ ** 6 * sigma_22 * sigma_42 * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.0833333333333333 * lambda_ ** 6 * sigma_22 * sigma_42 * exp(
            2 * lambda_ * tau) - 0.152777777777778 * lambda_ ** 6 * sigma_40 ** 2 * exp(2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.222222222222222 * lambda_ ** 6 * sigma_40 ** 2 * exp(
            2 * lambda_ * tau) * exp(lambda_svensson * tau) - 0.0694444444444444 * lambda_ ** 6 * sigma_40 ** 2 * exp(
            2 * lambda_ * tau) - 0.152777777777778 * lambda_ ** 6 * sigma_41 ** 2 * exp(2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.222222222222222 * lambda_ ** 6 * sigma_41 ** 2 * exp(
            2 * lambda_ * tau) * exp(lambda_svensson * tau) - 0.0694444444444444 * lambda_ ** 6 * sigma_41 ** 2 * exp(
            2 * lambda_ * tau) - 0.152777777777778 * lambda_ ** 6 * sigma_42 ** 2 * exp(2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.222222222222222 * lambda_ ** 6 * sigma_42 ** 2 * exp(
            2 * lambda_ * tau) * exp(lambda_svensson * tau) - 0.0694444444444444 * lambda_ ** 6 * sigma_42 ** 2 * exp(
            2 * lambda_ * tau) - 0.152777777777778 * lambda_ ** 6 * sigma_43 ** 2 * exp(2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.222222222222222 * lambda_ ** 6 * sigma_43 ** 2 * exp(
            2 * lambda_ * tau) * exp(lambda_svensson * tau) - 0.0694444444444444 * lambda_ ** 6 * sigma_43 ** 2 * exp(
            2 * lambda_ * tau) - 0.152777777777778 * lambda_ ** 6 * sigma_44 ** 2 * exp(2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.222222222222222 * lambda_ ** 6 * sigma_44 ** 2 * exp(
            2 * lambda_ * tau) * exp(lambda_svensson * tau) - 0.0694444444444444 * lambda_ ** 6 * sigma_44 ** 2 * exp(
            2 * lambda_ * tau) + 0.0555555555555556 * lambda_ ** 5 * lambda_svensson ** 4 * sigma_00 ** 2 * tau ** 3 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.0555555555555556 * lambda_ ** 5 * lambda_svensson ** 3 * sigma_00 * sigma_10 * tau ** 2 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.166666666666667 * lambda_ ** 5 * lambda_svensson ** 3 * sigma_00 * sigma_20 * tau ** 2 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.0555555555555556 * lambda_ ** 5 * lambda_svensson ** 3 * sigma_00 * sigma_30 * tau ** 2 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.111111111111111 * lambda_ ** 5 * lambda_svensson ** 3 * sigma_00 * sigma_30 * tau ** 2 * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.166666666666667 * lambda_ ** 5 * lambda_svensson ** 3 * sigma_00 * sigma_40 * tau ** 2 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.333333333333333 * lambda_ ** 5 * lambda_svensson ** 3 * sigma_00 * sigma_40 * tau ** 2 * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.0277777777777778 * lambda_ ** 5 * lambda_svensson ** 3 * sigma_30 ** 2 * tau ** 2 * exp(
            2 * lambda_svensson * tau) - 0.111111111111111 * lambda_ ** 5 * lambda_svensson ** 3 * sigma_30 * sigma_40 * tau ** 2 * exp(
            lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.0277777777777778 * lambda_ ** 5 * lambda_svensson ** 3 * sigma_31 ** 2 * tau ** 2 * exp(
            2 * lambda_svensson * tau) - 0.111111111111111 * lambda_ ** 5 * lambda_svensson ** 3 * sigma_31 * sigma_41 * tau ** 2 * exp(
            lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.0277777777777778 * lambda_ ** 5 * lambda_svensson ** 3 * sigma_32 ** 2 * tau ** 2 * exp(
            2 * lambda_svensson * tau) - 0.111111111111111 * lambda_ ** 5 * lambda_svensson ** 3 * sigma_32 * sigma_42 * tau ** 2 * exp(
            lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.0277777777777778 * lambda_ ** 5 * lambda_svensson ** 3 * sigma_33 ** 2 * tau ** 2 * exp(
            2 * lambda_svensson * tau) - 0.111111111111111 * lambda_ ** 5 * lambda_svensson ** 3 * sigma_33 * sigma_43 * tau ** 2 * exp(
            lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.0833333333333333 * lambda_ ** 5 * lambda_svensson ** 3 * sigma_40 ** 2 * tau ** 2 * exp(
            2 * lambda_ * tau) - 0.0833333333333333 * lambda_ ** 5 * lambda_svensson ** 3 * sigma_41 ** 2 * tau ** 2 * exp(
            2 * lambda_ * tau) - 0.0833333333333333 * lambda_ ** 5 * lambda_svensson ** 3 * sigma_42 ** 2 * tau ** 2 * exp(
            2 * lambda_ * tau) - 0.0833333333333333 * lambda_ ** 5 * lambda_svensson ** 3 * sigma_43 ** 2 * tau ** 2 * exp(
            2 * lambda_ * tau) - 0.0833333333333333 * lambda_ ** 5 * lambda_svensson ** 3 * sigma_44 ** 2 * tau ** 2 * exp(
            2 * lambda_ * tau) + 0.333333333333333 * lambda_ ** 5 * lambda_svensson ** 2 * sigma_00 * sigma_20 * tau * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) + 1.0 * lambda_ ** 5 * lambda_svensson ** 2 * sigma_00 * sigma_40 * tau * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) + 0.111111111111111 * lambda_ ** 5 * lambda_svensson ** 2 * sigma_10 * sigma_20 * tau * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.111111111111111 * lambda_ ** 5 * lambda_svensson ** 2 * sigma_10 * sigma_40 * tau * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.111111111111111 * lambda_ ** 5 * lambda_svensson ** 2 * sigma_10 * sigma_40 * tau * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) + 0.111111111111111 * lambda_ ** 5 * lambda_svensson ** 2 * sigma_11 * sigma_21 * tau * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.111111111111111 * lambda_ ** 5 * lambda_svensson ** 2 * sigma_11 * sigma_41 * tau * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.111111111111111 * lambda_ ** 5 * lambda_svensson ** 2 * sigma_11 * sigma_41 * tau * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) + 0.166666666666667 * lambda_ ** 5 * lambda_svensson ** 2 * sigma_20 ** 2 * tau * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.111111111111111 * lambda_ ** 5 * lambda_svensson ** 2 * sigma_20 * sigma_30 * tau * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.111111111111111 * lambda_ ** 5 * lambda_svensson ** 2 * sigma_20 * sigma_30 * tau * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) - 0.111111111111111 * lambda_ ** 5 * lambda_svensson ** 2 * sigma_20 * sigma_30 * tau * exp(
            lambda_ * tau) * exp(
            lambda_svensson * tau) + 0.333333333333333 * lambda_ ** 5 * lambda_svensson ** 2 * sigma_20 * sigma_40 * tau * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.333333333333333 * lambda_ ** 5 * lambda_svensson ** 2 * sigma_20 * sigma_40 * tau * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.166666666666667 * lambda_ ** 5 * lambda_svensson ** 2 * sigma_20 * sigma_40 * tau * exp(
            2 * lambda_ * tau) + 0.166666666666667 * lambda_ ** 5 * lambda_svensson ** 2 * sigma_21 ** 2 * tau * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.111111111111111 * lambda_ ** 5 * lambda_svensson ** 2 * sigma_21 * sigma_31 * tau * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.111111111111111 * lambda_ ** 5 * lambda_svensson ** 2 * sigma_21 * sigma_31 * tau * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) - 0.111111111111111 * lambda_ ** 5 * lambda_svensson ** 2 * sigma_21 * sigma_31 * tau * exp(
            lambda_ * tau) * exp(
            lambda_svensson * tau) + 0.333333333333333 * lambda_ ** 5 * lambda_svensson ** 2 * sigma_21 * sigma_41 * tau * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.333333333333333 * lambda_ ** 5 * lambda_svensson ** 2 * sigma_21 * sigma_41 * tau * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.166666666666667 * lambda_ ** 5 * lambda_svensson ** 2 * sigma_21 * sigma_41 * tau * exp(
            2 * lambda_ * tau) + 0.166666666666667 * lambda_ ** 5 * lambda_svensson ** 2 * sigma_22 ** 2 * tau * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.111111111111111 * lambda_ ** 5 * lambda_svensson ** 2 * sigma_22 * sigma_32 * tau * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.111111111111111 * lambda_ ** 5 * lambda_svensson ** 2 * sigma_22 * sigma_32 * tau * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) - 0.111111111111111 * lambda_ ** 5 * lambda_svensson ** 2 * sigma_22 * sigma_32 * tau * exp(
            lambda_ * tau) * exp(
            lambda_svensson * tau) + 0.333333333333333 * lambda_ ** 5 * lambda_svensson ** 2 * sigma_22 * sigma_42 * tau * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.333333333333333 * lambda_ ** 5 * lambda_svensson ** 2 * sigma_22 * sigma_42 * tau * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.166666666666667 * lambda_ ** 5 * lambda_svensson ** 2 * sigma_22 * sigma_42 * tau * exp(
            2 * lambda_ * tau) + 0.111111111111111 * lambda_ ** 5 * lambda_svensson ** 2 * sigma_30 * sigma_40 * tau * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.111111111111111 * lambda_ ** 5 * lambda_svensson ** 2 * sigma_30 * sigma_40 * tau * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) + 0.111111111111111 * lambda_ ** 5 * lambda_svensson ** 2 * sigma_30 * sigma_40 * tau * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) - 0.111111111111111 * lambda_ ** 5 * lambda_svensson ** 2 * sigma_30 * sigma_40 * tau * exp(
            lambda_ * tau) * exp(
            lambda_svensson * tau) + 0.111111111111111 * lambda_ ** 5 * lambda_svensson ** 2 * sigma_31 * sigma_41 * tau * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.111111111111111 * lambda_ ** 5 * lambda_svensson ** 2 * sigma_31 * sigma_41 * tau * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) + 0.111111111111111 * lambda_ ** 5 * lambda_svensson ** 2 * sigma_31 * sigma_41 * tau * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) - 0.111111111111111 * lambda_ ** 5 * lambda_svensson ** 2 * sigma_31 * sigma_41 * tau * exp(
            lambda_ * tau) * exp(
            lambda_svensson * tau) + 0.111111111111111 * lambda_ ** 5 * lambda_svensson ** 2 * sigma_32 * sigma_42 * tau * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.111111111111111 * lambda_ ** 5 * lambda_svensson ** 2 * sigma_32 * sigma_42 * tau * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) + 0.111111111111111 * lambda_ ** 5 * lambda_svensson ** 2 * sigma_32 * sigma_42 * tau * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) - 0.111111111111111 * lambda_ ** 5 * lambda_svensson ** 2 * sigma_32 * sigma_42 * tau * exp(
            lambda_ * tau) * exp(
            lambda_svensson * tau) + 0.111111111111111 * lambda_ ** 5 * lambda_svensson ** 2 * sigma_33 * sigma_43 * tau * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.111111111111111 * lambda_ ** 5 * lambda_svensson ** 2 * sigma_33 * sigma_43 * tau * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) + 0.111111111111111 * lambda_ ** 5 * lambda_svensson ** 2 * sigma_33 * sigma_43 * tau * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) - 0.111111111111111 * lambda_ ** 5 * lambda_svensson ** 2 * sigma_33 * sigma_43 * tau * exp(
            lambda_ * tau) * exp(
            lambda_svensson * tau) + 0.166666666666667 * lambda_ ** 5 * lambda_svensson ** 2 * sigma_40 ** 2 * tau * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.333333333333333 * lambda_ ** 5 * lambda_svensson ** 2 * sigma_40 ** 2 * tau * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.25 * lambda_ ** 5 * lambda_svensson ** 2 * sigma_40 ** 2 * tau * exp(
            2 * lambda_ * tau) + 0.166666666666667 * lambda_ ** 5 * lambda_svensson ** 2 * sigma_41 ** 2 * tau * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.333333333333333 * lambda_ ** 5 * lambda_svensson ** 2 * sigma_41 ** 2 * tau * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.25 * lambda_ ** 5 * lambda_svensson ** 2 * sigma_41 ** 2 * tau * exp(
            2 * lambda_ * tau) + 0.166666666666667 * lambda_ ** 5 * lambda_svensson ** 2 * sigma_42 ** 2 * tau * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.333333333333333 * lambda_ ** 5 * lambda_svensson ** 2 * sigma_42 ** 2 * tau * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.25 * lambda_ ** 5 * lambda_svensson ** 2 * sigma_42 ** 2 * tau * exp(
            2 * lambda_ * tau) + 0.166666666666667 * lambda_ ** 5 * lambda_svensson ** 2 * sigma_43 ** 2 * tau * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.333333333333333 * lambda_ ** 5 * lambda_svensson ** 2 * sigma_43 ** 2 * tau * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.25 * lambda_ ** 5 * lambda_svensson ** 2 * sigma_43 ** 2 * tau * exp(
            2 * lambda_ * tau) + 0.166666666666667 * lambda_ ** 5 * lambda_svensson ** 2 * sigma_44 ** 2 * tau * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.333333333333333 * lambda_ ** 5 * lambda_svensson ** 2 * sigma_44 ** 2 * tau * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.25 * lambda_ ** 5 * lambda_svensson ** 2 * sigma_44 ** 2 * tau * exp(
            2 * lambda_ * tau) - 0.333333333333333 * lambda_ ** 5 * lambda_svensson * sigma_00 * sigma_20 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.333333333333333 * lambda_ ** 5 * lambda_svensson * sigma_00 * sigma_20 * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) - 1.0 * lambda_ ** 5 * lambda_svensson * sigma_00 * sigma_40 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 1.0 * lambda_ ** 5 * lambda_svensson * sigma_00 * sigma_40 * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.111111111111111 * lambda_ ** 5 * lambda_svensson * sigma_10 * sigma_20 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.111111111111111 * lambda_ ** 5 * lambda_svensson * sigma_10 * sigma_20 * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.222222222222222 * lambda_ ** 5 * lambda_svensson * sigma_10 * sigma_40 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.222222222222222 * lambda_ ** 5 * lambda_svensson * sigma_10 * sigma_40 * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.111111111111111 * lambda_ ** 5 * lambda_svensson * sigma_11 * sigma_21 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.111111111111111 * lambda_ ** 5 * lambda_svensson * sigma_11 * sigma_21 * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.222222222222222 * lambda_ ** 5 * lambda_svensson * sigma_11 * sigma_41 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.222222222222222 * lambda_ ** 5 * lambda_svensson * sigma_11 * sigma_41 * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.25 * lambda_ ** 5 * lambda_svensson * sigma_20 ** 2 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.333333333333333 * lambda_ ** 5 * lambda_svensson * sigma_20 ** 2 * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.0833333333333333 * lambda_ ** 5 * lambda_svensson * sigma_20 ** 2 * exp(
            2 * lambda_ * tau) - 0.111111111111111 * lambda_ ** 5 * lambda_svensson * sigma_20 * sigma_30 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.111111111111111 * lambda_ ** 5 * lambda_svensson * sigma_20 * sigma_30 * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.75 * lambda_ ** 5 * lambda_svensson * sigma_20 * sigma_40 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 1.0 * lambda_ ** 5 * lambda_svensson * sigma_20 * sigma_40 * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.25 * lambda_ ** 5 * lambda_svensson * sigma_20 * sigma_40 * exp(
            2 * lambda_ * tau) - 0.25 * lambda_ ** 5 * lambda_svensson * sigma_21 ** 2 * exp(2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.333333333333333 * lambda_ ** 5 * lambda_svensson * sigma_21 ** 2 * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.0833333333333333 * lambda_ ** 5 * lambda_svensson * sigma_21 ** 2 * exp(
            2 * lambda_ * tau) - 0.111111111111111 * lambda_ ** 5 * lambda_svensson * sigma_21 * sigma_31 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.111111111111111 * lambda_ ** 5 * lambda_svensson * sigma_21 * sigma_31 * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.75 * lambda_ ** 5 * lambda_svensson * sigma_21 * sigma_41 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 1.0 * lambda_ ** 5 * lambda_svensson * sigma_21 * sigma_41 * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.25 * lambda_ ** 5 * lambda_svensson * sigma_21 * sigma_41 * exp(
            2 * lambda_ * tau) - 0.25 * lambda_ ** 5 * lambda_svensson * sigma_22 ** 2 * exp(2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.333333333333333 * lambda_ ** 5 * lambda_svensson * sigma_22 ** 2 * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.0833333333333333 * lambda_ ** 5 * lambda_svensson * sigma_22 ** 2 * exp(
            2 * lambda_ * tau) - 0.111111111111111 * lambda_ ** 5 * lambda_svensson * sigma_22 * sigma_32 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.111111111111111 * lambda_ ** 5 * lambda_svensson * sigma_22 * sigma_32 * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.75 * lambda_ ** 5 * lambda_svensson * sigma_22 * sigma_42 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 1.0 * lambda_ ** 5 * lambda_svensson * sigma_22 * sigma_42 * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.25 * lambda_ ** 5 * lambda_svensson * sigma_22 * sigma_42 * exp(
            2 * lambda_ * tau) - 0.222222222222222 * lambda_ ** 5 * lambda_svensson * sigma_30 * sigma_40 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.222222222222222 * lambda_ ** 5 * lambda_svensson * sigma_30 * sigma_40 * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.222222222222222 * lambda_ ** 5 * lambda_svensson * sigma_31 * sigma_41 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.222222222222222 * lambda_ ** 5 * lambda_svensson * sigma_31 * sigma_41 * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.222222222222222 * lambda_ ** 5 * lambda_svensson * sigma_32 * sigma_42 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.222222222222222 * lambda_ ** 5 * lambda_svensson * sigma_32 * sigma_42 * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.222222222222222 * lambda_ ** 5 * lambda_svensson * sigma_33 * sigma_43 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.222222222222222 * lambda_ ** 5 * lambda_svensson * sigma_33 * sigma_43 * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.458333333333333 * lambda_ ** 5 * lambda_svensson * sigma_40 ** 2 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.666666666666667 * lambda_ ** 5 * lambda_svensson * sigma_40 ** 2 * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.208333333333333 * lambda_ ** 5 * lambda_svensson * sigma_40 ** 2 * exp(
            2 * lambda_ * tau) - 0.458333333333333 * lambda_ ** 5 * lambda_svensson * sigma_41 ** 2 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.666666666666667 * lambda_ ** 5 * lambda_svensson * sigma_41 ** 2 * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.208333333333333 * lambda_ ** 5 * lambda_svensson * sigma_41 ** 2 * exp(
            2 * lambda_ * tau) - 0.458333333333333 * lambda_ ** 5 * lambda_svensson * sigma_42 ** 2 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.666666666666667 * lambda_ ** 5 * lambda_svensson * sigma_42 ** 2 * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.208333333333333 * lambda_ ** 5 * lambda_svensson * sigma_42 ** 2 * exp(
            2 * lambda_ * tau) - 0.458333333333333 * lambda_ ** 5 * lambda_svensson * sigma_43 ** 2 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.666666666666667 * lambda_ ** 5 * lambda_svensson * sigma_43 ** 2 * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.208333333333333 * lambda_ ** 5 * lambda_svensson * sigma_43 ** 2 * exp(
            2 * lambda_ * tau) - 0.458333333333333 * lambda_ ** 5 * lambda_svensson * sigma_44 ** 2 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.666666666666667 * lambda_ ** 5 * lambda_svensson * sigma_44 ** 2 * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.208333333333333 * lambda_ ** 5 * lambda_svensson * sigma_44 ** 2 * exp(
            2 * lambda_ * tau) + 0.0555555555555556 * lambda_ ** 4 * lambda_svensson ** 5 * sigma_00 ** 2 * tau ** 3 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.166666666666667 * lambda_ ** 4 * lambda_svensson ** 4 * sigma_00 * sigma_10 * tau ** 2 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.166666666666667 * lambda_ ** 4 * lambda_svensson ** 4 * sigma_00 * sigma_20 * tau ** 2 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.166666666666667 * lambda_ ** 4 * lambda_svensson ** 4 * sigma_00 * sigma_30 * tau ** 2 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.333333333333333 * lambda_ ** 4 * lambda_svensson ** 4 * sigma_00 * sigma_30 * tau ** 2 * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.166666666666667 * lambda_ ** 4 * lambda_svensson ** 4 * sigma_00 * sigma_40 * tau ** 2 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.333333333333333 * lambda_ ** 4 * lambda_svensson ** 4 * sigma_00 * sigma_40 * tau ** 2 * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.0833333333333333 * lambda_ ** 4 * lambda_svensson ** 4 * sigma_30 ** 2 * tau ** 2 * exp(
            2 * lambda_svensson * tau) - 0.222222222222222 * lambda_ ** 4 * lambda_svensson ** 4 * sigma_30 * sigma_40 * tau ** 2 * exp(
            lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.0833333333333333 * lambda_ ** 4 * lambda_svensson ** 4 * sigma_31 ** 2 * tau ** 2 * exp(
            2 * lambda_svensson * tau) - 0.222222222222222 * lambda_ ** 4 * lambda_svensson ** 4 * sigma_31 * sigma_41 * tau ** 2 * exp(
            lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.0833333333333333 * lambda_ ** 4 * lambda_svensson ** 4 * sigma_32 ** 2 * tau ** 2 * exp(
            2 * lambda_svensson * tau) - 0.222222222222222 * lambda_ ** 4 * lambda_svensson ** 4 * sigma_32 * sigma_42 * tau ** 2 * exp(
            lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.0833333333333333 * lambda_ ** 4 * lambda_svensson ** 4 * sigma_33 ** 2 * tau ** 2 * exp(
            2 * lambda_svensson * tau) - 0.222222222222222 * lambda_ ** 4 * lambda_svensson ** 4 * sigma_33 * sigma_43 * tau ** 2 * exp(
            lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.0833333333333333 * lambda_ ** 4 * lambda_svensson ** 4 * sigma_40 ** 2 * tau ** 2 * exp(
            2 * lambda_ * tau) - 0.0833333333333333 * lambda_ ** 4 * lambda_svensson ** 4 * sigma_41 ** 2 * tau ** 2 * exp(
            2 * lambda_ * tau) - 0.0833333333333333 * lambda_ ** 4 * lambda_svensson ** 4 * sigma_42 ** 2 * tau ** 2 * exp(
            2 * lambda_ * tau) - 0.0833333333333333 * lambda_ ** 4 * lambda_svensson ** 4 * sigma_43 ** 2 * tau ** 2 * exp(
            2 * lambda_ * tau) - 0.0833333333333333 * lambda_ ** 4 * lambda_svensson ** 4 * sigma_44 ** 2 * tau ** 2 * exp(
            2 * lambda_ * tau) + 0.111111111111111 * lambda_ ** 4 * lambda_svensson ** 3 * sigma_00 * sigma_10 * tau * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.333333333333333 * lambda_ ** 4 * lambda_svensson ** 3 * sigma_00 * sigma_20 * tau * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) + 0.333333333333333 * lambda_ ** 4 * lambda_svensson ** 3 * sigma_00 * sigma_30 * tau * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 1.0 * lambda_ ** 4 * lambda_svensson ** 3 * sigma_00 * sigma_40 * tau * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) + 0.0555555555555556 * lambda_ ** 4 * lambda_svensson ** 3 * sigma_10 ** 2 * tau * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.333333333333333 * lambda_ ** 4 * lambda_svensson ** 3 * sigma_10 * sigma_20 * tau * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.111111111111111 * lambda_ ** 4 * lambda_svensson ** 3 * sigma_10 * sigma_30 * tau * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.111111111111111 * lambda_ ** 4 * lambda_svensson ** 3 * sigma_10 * sigma_30 * tau * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) - 0.0555555555555556 * lambda_ ** 4 * lambda_svensson ** 3 * sigma_10 * sigma_30 * tau * exp(
            2 * lambda_svensson * tau) + 0.333333333333333 * lambda_ ** 4 * lambda_svensson ** 3 * sigma_10 * sigma_40 * tau * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.333333333333333 * lambda_ ** 4 * lambda_svensson ** 3 * sigma_10 * sigma_40 * tau * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.111111111111111 * lambda_ ** 4 * lambda_svensson ** 3 * sigma_10 * sigma_40 * tau * exp(
            lambda_ * tau) * exp(
            lambda_svensson * tau) + 0.0555555555555556 * lambda_ ** 4 * lambda_svensson ** 3 * sigma_11 ** 2 * tau * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.333333333333333 * lambda_ ** 4 * lambda_svensson ** 3 * sigma_11 * sigma_21 * tau * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.111111111111111 * lambda_ ** 4 * lambda_svensson ** 3 * sigma_11 * sigma_31 * tau * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.111111111111111 * lambda_ ** 4 * lambda_svensson ** 3 * sigma_11 * sigma_31 * tau * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) - 0.0555555555555556 * lambda_ ** 4 * lambda_svensson ** 3 * sigma_11 * sigma_31 * tau * exp(
            2 * lambda_svensson * tau) + 0.333333333333333 * lambda_ ** 4 * lambda_svensson ** 3 * sigma_11 * sigma_41 * tau * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.333333333333333 * lambda_ ** 4 * lambda_svensson ** 3 * sigma_11 * sigma_41 * tau * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.111111111111111 * lambda_ ** 4 * lambda_svensson ** 3 * sigma_11 * sigma_41 * tau * exp(
            lambda_ * tau) * exp(
            lambda_svensson * tau) + 0.166666666666667 * lambda_ ** 4 * lambda_svensson ** 3 * sigma_20 ** 2 * tau * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.333333333333333 * lambda_ ** 4 * lambda_svensson ** 3 * sigma_20 * sigma_30 * tau * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.333333333333333 * lambda_ ** 4 * lambda_svensson ** 3 * sigma_20 * sigma_30 * tau * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) - 0.222222222222222 * lambda_ ** 4 * lambda_svensson ** 3 * sigma_20 * sigma_30 * tau * exp(
            lambda_ * tau) * exp(
            lambda_svensson * tau) + 0.333333333333333 * lambda_ ** 4 * lambda_svensson ** 3 * sigma_20 * sigma_40 * tau * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.333333333333333 * lambda_ ** 4 * lambda_svensson ** 3 * sigma_20 * sigma_40 * tau * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.166666666666667 * lambda_ ** 4 * lambda_svensson ** 3 * sigma_20 * sigma_40 * tau * exp(
            2 * lambda_ * tau) + 0.166666666666667 * lambda_ ** 4 * lambda_svensson ** 3 * sigma_21 ** 2 * tau * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.333333333333333 * lambda_ ** 4 * lambda_svensson ** 3 * sigma_21 * sigma_31 * tau * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.333333333333333 * lambda_ ** 4 * lambda_svensson ** 3 * sigma_21 * sigma_31 * tau * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) - 0.222222222222222 * lambda_ ** 4 * lambda_svensson ** 3 * sigma_21 * sigma_31 * tau * exp(
            lambda_ * tau) * exp(
            lambda_svensson * tau) + 0.333333333333333 * lambda_ ** 4 * lambda_svensson ** 3 * sigma_21 * sigma_41 * tau * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.333333333333333 * lambda_ ** 4 * lambda_svensson ** 3 * sigma_21 * sigma_41 * tau * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.166666666666667 * lambda_ ** 4 * lambda_svensson ** 3 * sigma_21 * sigma_41 * tau * exp(
            2 * lambda_ * tau) + 0.166666666666667 * lambda_ ** 4 * lambda_svensson ** 3 * sigma_22 ** 2 * tau * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.333333333333333 * lambda_ ** 4 * lambda_svensson ** 3 * sigma_22 * sigma_32 * tau * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.333333333333333 * lambda_ ** 4 * lambda_svensson ** 3 * sigma_22 * sigma_32 * tau * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) - 0.222222222222222 * lambda_ ** 4 * lambda_svensson ** 3 * sigma_22 * sigma_32 * tau * exp(
            lambda_ * tau) * exp(
            lambda_svensson * tau) + 0.333333333333333 * lambda_ ** 4 * lambda_svensson ** 3 * sigma_22 * sigma_42 * tau * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.333333333333333 * lambda_ ** 4 * lambda_svensson ** 3 * sigma_22 * sigma_42 * tau * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.166666666666667 * lambda_ ** 4 * lambda_svensson ** 3 * sigma_22 * sigma_42 * tau * exp(
            2 * lambda_ * tau) + 0.0555555555555556 * lambda_ ** 4 * lambda_svensson ** 3 * sigma_30 ** 2 * tau * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.111111111111111 * lambda_ ** 4 * lambda_svensson ** 3 * sigma_30 ** 2 * tau * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) - 0.0833333333333333 * lambda_ ** 4 * lambda_svensson ** 3 * sigma_30 ** 2 * tau * exp(
            2 * lambda_svensson * tau) + 0.333333333333333 * lambda_ ** 4 * lambda_svensson ** 3 * sigma_30 * sigma_40 * tau * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.333333333333333 * lambda_ ** 4 * lambda_svensson ** 3 * sigma_30 * sigma_40 * tau * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) + 0.333333333333333 * lambda_ ** 4 * lambda_svensson ** 3 * sigma_30 * sigma_40 * tau * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) - 0.555555555555556 * lambda_ ** 4 * lambda_svensson ** 3 * sigma_30 * sigma_40 * tau * exp(
            lambda_ * tau) * exp(
            lambda_svensson * tau) + 0.0555555555555556 * lambda_ ** 4 * lambda_svensson ** 3 * sigma_31 ** 2 * tau * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.111111111111111 * lambda_ ** 4 * lambda_svensson ** 3 * sigma_31 ** 2 * tau * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) - 0.0833333333333333 * lambda_ ** 4 * lambda_svensson ** 3 * sigma_31 ** 2 * tau * exp(
            2 * lambda_svensson * tau) + 0.333333333333333 * lambda_ ** 4 * lambda_svensson ** 3 * sigma_31 * sigma_41 * tau * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.333333333333333 * lambda_ ** 4 * lambda_svensson ** 3 * sigma_31 * sigma_41 * tau * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) + 0.333333333333333 * lambda_ ** 4 * lambda_svensson ** 3 * sigma_31 * sigma_41 * tau * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) - 0.555555555555556 * lambda_ ** 4 * lambda_svensson ** 3 * sigma_31 * sigma_41 * tau * exp(
            lambda_ * tau) * exp(
            lambda_svensson * tau) + 0.0555555555555556 * lambda_ ** 4 * lambda_svensson ** 3 * sigma_32 ** 2 * tau * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.111111111111111 * lambda_ ** 4 * lambda_svensson ** 3 * sigma_32 ** 2 * tau * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) - 0.0833333333333333 * lambda_ ** 4 * lambda_svensson ** 3 * sigma_32 ** 2 * tau * exp(
            2 * lambda_svensson * tau) + 0.333333333333333 * lambda_ ** 4 * lambda_svensson ** 3 * sigma_32 * sigma_42 * tau * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.333333333333333 * lambda_ ** 4 * lambda_svensson ** 3 * sigma_32 * sigma_42 * tau * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) + 0.333333333333333 * lambda_ ** 4 * lambda_svensson ** 3 * sigma_32 * sigma_42 * tau * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) - 0.555555555555556 * lambda_ ** 4 * lambda_svensson ** 3 * sigma_32 * sigma_42 * tau * exp(
            lambda_ * tau) * exp(
            lambda_svensson * tau) + 0.0555555555555556 * lambda_ ** 4 * lambda_svensson ** 3 * sigma_33 ** 2 * tau * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.111111111111111 * lambda_ ** 4 * lambda_svensson ** 3 * sigma_33 ** 2 * tau * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) - 0.0833333333333333 * lambda_ ** 4 * lambda_svensson ** 3 * sigma_33 ** 2 * tau * exp(
            2 * lambda_svensson * tau) + 0.333333333333333 * lambda_ ** 4 * lambda_svensson ** 3 * sigma_33 * sigma_43 * tau * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.333333333333333 * lambda_ ** 4 * lambda_svensson ** 3 * sigma_33 * sigma_43 * tau * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) + 0.333333333333333 * lambda_ ** 4 * lambda_svensson ** 3 * sigma_33 * sigma_43 * tau * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) - 0.555555555555556 * lambda_ ** 4 * lambda_svensson ** 3 * sigma_33 * sigma_43 * tau * exp(
            lambda_ * tau) * exp(
            lambda_svensson * tau) + 0.166666666666667 * lambda_ ** 4 * lambda_svensson ** 3 * sigma_40 ** 2 * tau * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.333333333333333 * lambda_ ** 4 * lambda_svensson ** 3 * sigma_40 ** 2 * tau * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.25 * lambda_ ** 4 * lambda_svensson ** 3 * sigma_40 ** 2 * tau * exp(
            2 * lambda_ * tau) + 0.166666666666667 * lambda_ ** 4 * lambda_svensson ** 3 * sigma_41 ** 2 * tau * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.333333333333333 * lambda_ ** 4 * lambda_svensson ** 3 * sigma_41 ** 2 * tau * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.25 * lambda_ ** 4 * lambda_svensson ** 3 * sigma_41 ** 2 * tau * exp(
            2 * lambda_ * tau) + 0.166666666666667 * lambda_ ** 4 * lambda_svensson ** 3 * sigma_42 ** 2 * tau * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.333333333333333 * lambda_ ** 4 * lambda_svensson ** 3 * sigma_42 ** 2 * tau * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.25 * lambda_ ** 4 * lambda_svensson ** 3 * sigma_42 ** 2 * tau * exp(
            2 * lambda_ * tau) + 0.166666666666667 * lambda_ ** 4 * lambda_svensson ** 3 * sigma_43 ** 2 * tau * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.333333333333333 * lambda_ ** 4 * lambda_svensson ** 3 * sigma_43 ** 2 * tau * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.25 * lambda_ ** 4 * lambda_svensson ** 3 * sigma_43 ** 2 * tau * exp(
            2 * lambda_ * tau) + 0.166666666666667 * lambda_ ** 4 * lambda_svensson ** 3 * sigma_44 ** 2 * tau * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.333333333333333 * lambda_ ** 4 * lambda_svensson ** 3 * sigma_44 ** 2 * tau * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.25 * lambda_ ** 4 * lambda_svensson ** 3 * sigma_44 ** 2 * tau * exp(
            2 * lambda_ * tau) - 0.333333333333333 * lambda_ ** 4 * lambda_svensson ** 2 * sigma_00 * sigma_20 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.333333333333333 * lambda_ ** 4 * lambda_svensson ** 2 * sigma_00 * sigma_20 * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) - 1.0 * lambda_ ** 4 * lambda_svensson ** 2 * sigma_00 * sigma_40 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 1.0 * lambda_ ** 4 * lambda_svensson ** 2 * sigma_00 * sigma_40 * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.333333333333333 * lambda_ ** 4 * lambda_svensson ** 2 * sigma_10 * sigma_20 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.333333333333333 * lambda_ ** 4 * lambda_svensson ** 2 * sigma_10 * sigma_20 * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) + 0.111111111111111 * lambda_ ** 4 * lambda_svensson ** 2 * sigma_10 * sigma_20 * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) - 0.111111111111111 * lambda_ ** 4 * lambda_svensson ** 2 * sigma_10 * sigma_20 * exp(
            lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.666666666666667 * lambda_ ** 4 * lambda_svensson ** 2 * sigma_10 * sigma_40 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.666666666666667 * lambda_ ** 4 * lambda_svensson ** 2 * sigma_10 * sigma_40 * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) + 0.111111111111111 * lambda_ ** 4 * lambda_svensson ** 2 * sigma_10 * sigma_40 * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) - 0.111111111111111 * lambda_ ** 4 * lambda_svensson ** 2 * sigma_10 * sigma_40 * exp(
            lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.333333333333333 * lambda_ ** 4 * lambda_svensson ** 2 * sigma_11 * sigma_21 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.333333333333333 * lambda_ ** 4 * lambda_svensson ** 2 * sigma_11 * sigma_21 * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) + 0.111111111111111 * lambda_ ** 4 * lambda_svensson ** 2 * sigma_11 * sigma_21 * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) - 0.111111111111111 * lambda_ ** 4 * lambda_svensson ** 2 * sigma_11 * sigma_21 * exp(
            lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.666666666666667 * lambda_ ** 4 * lambda_svensson ** 2 * sigma_11 * sigma_41 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.666666666666667 * lambda_ ** 4 * lambda_svensson ** 2 * sigma_11 * sigma_41 * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) + 0.111111111111111 * lambda_ ** 4 * lambda_svensson ** 2 * sigma_11 * sigma_41 * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) - 0.111111111111111 * lambda_ ** 4 * lambda_svensson ** 2 * sigma_11 * sigma_41 * exp(
            lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.25 * lambda_ ** 4 * lambda_svensson ** 2 * sigma_20 ** 2 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.333333333333333 * lambda_ ** 4 * lambda_svensson ** 2 * sigma_20 ** 2 * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.0833333333333333 * lambda_ ** 4 * lambda_svensson ** 2 * sigma_20 ** 2 * exp(
            2 * lambda_ * tau) - 0.333333333333333 * lambda_ ** 4 * lambda_svensson ** 2 * sigma_20 * sigma_30 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.333333333333333 * lambda_ ** 4 * lambda_svensson ** 2 * sigma_20 * sigma_30 * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) + 0.222222222222222 * lambda_ ** 4 * lambda_svensson ** 2 * sigma_20 * sigma_30 * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) - 0.222222222222222 * lambda_ ** 4 * lambda_svensson ** 2 * sigma_20 * sigma_30 * exp(
            lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.75 * lambda_ ** 4 * lambda_svensson ** 2 * sigma_20 * sigma_40 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 1.0 * lambda_ ** 4 * lambda_svensson ** 2 * sigma_20 * sigma_40 * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.25 * lambda_ ** 4 * lambda_svensson ** 2 * sigma_20 * sigma_40 * exp(
            2 * lambda_ * tau) - 0.25 * lambda_ ** 4 * lambda_svensson ** 2 * sigma_21 ** 2 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.333333333333333 * lambda_ ** 4 * lambda_svensson ** 2 * sigma_21 ** 2 * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.0833333333333333 * lambda_ ** 4 * lambda_svensson ** 2 * sigma_21 ** 2 * exp(
            2 * lambda_ * tau) - 0.333333333333333 * lambda_ ** 4 * lambda_svensson ** 2 * sigma_21 * sigma_31 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.333333333333333 * lambda_ ** 4 * lambda_svensson ** 2 * sigma_21 * sigma_31 * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) + 0.222222222222222 * lambda_ ** 4 * lambda_svensson ** 2 * sigma_21 * sigma_31 * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) - 0.222222222222222 * lambda_ ** 4 * lambda_svensson ** 2 * sigma_21 * sigma_31 * exp(
            lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.75 * lambda_ ** 4 * lambda_svensson ** 2 * sigma_21 * sigma_41 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 1.0 * lambda_ ** 4 * lambda_svensson ** 2 * sigma_21 * sigma_41 * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.25 * lambda_ ** 4 * lambda_svensson ** 2 * sigma_21 * sigma_41 * exp(
            2 * lambda_ * tau) - 0.25 * lambda_ ** 4 * lambda_svensson ** 2 * sigma_22 ** 2 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.333333333333333 * lambda_ ** 4 * lambda_svensson ** 2 * sigma_22 ** 2 * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.0833333333333333 * lambda_ ** 4 * lambda_svensson ** 2 * sigma_22 ** 2 * exp(
            2 * lambda_ * tau) - 0.333333333333333 * lambda_ ** 4 * lambda_svensson ** 2 * sigma_22 * sigma_32 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.333333333333333 * lambda_ ** 4 * lambda_svensson ** 2 * sigma_22 * sigma_32 * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) + 0.222222222222222 * lambda_ ** 4 * lambda_svensson ** 2 * sigma_22 * sigma_32 * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) - 0.222222222222222 * lambda_ ** 4 * lambda_svensson ** 2 * sigma_22 * sigma_32 * exp(
            lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.75 * lambda_ ** 4 * lambda_svensson ** 2 * sigma_22 * sigma_42 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 1.0 * lambda_ ** 4 * lambda_svensson ** 2 * sigma_22 * sigma_42 * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.25 * lambda_ ** 4 * lambda_svensson ** 2 * sigma_22 * sigma_42 * exp(
            2 * lambda_ * tau) - 0.666666666666667 * lambda_ ** 4 * lambda_svensson ** 2 * sigma_30 * sigma_40 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.666666666666667 * lambda_ ** 4 * lambda_svensson ** 2 * sigma_30 * sigma_40 * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) + 0.222222222222222 * lambda_ ** 4 * lambda_svensson ** 2 * sigma_30 * sigma_40 * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) - 0.222222222222222 * lambda_ ** 4 * lambda_svensson ** 2 * sigma_30 * sigma_40 * exp(
            lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.666666666666667 * lambda_ ** 4 * lambda_svensson ** 2 * sigma_31 * sigma_41 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.666666666666667 * lambda_ ** 4 * lambda_svensson ** 2 * sigma_31 * sigma_41 * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) + 0.222222222222222 * lambda_ ** 4 * lambda_svensson ** 2 * sigma_31 * sigma_41 * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) - 0.222222222222222 * lambda_ ** 4 * lambda_svensson ** 2 * sigma_31 * sigma_41 * exp(
            lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.666666666666667 * lambda_ ** 4 * lambda_svensson ** 2 * sigma_32 * sigma_42 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.666666666666667 * lambda_ ** 4 * lambda_svensson ** 2 * sigma_32 * sigma_42 * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) + 0.222222222222222 * lambda_ ** 4 * lambda_svensson ** 2 * sigma_32 * sigma_42 * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) - 0.222222222222222 * lambda_ ** 4 * lambda_svensson ** 2 * sigma_32 * sigma_42 * exp(
            lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.666666666666667 * lambda_ ** 4 * lambda_svensson ** 2 * sigma_33 * sigma_43 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.666666666666667 * lambda_ ** 4 * lambda_svensson ** 2 * sigma_33 * sigma_43 * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) + 0.222222222222222 * lambda_ ** 4 * lambda_svensson ** 2 * sigma_33 * sigma_43 * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) - 0.222222222222222 * lambda_ ** 4 * lambda_svensson ** 2 * sigma_33 * sigma_43 * exp(
            lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.458333333333333 * lambda_ ** 4 * lambda_svensson ** 2 * sigma_40 ** 2 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.666666666666667 * lambda_ ** 4 * lambda_svensson ** 2 * sigma_40 ** 2 * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.208333333333333 * lambda_ ** 4 * lambda_svensson ** 2 * sigma_40 ** 2 * exp(
            2 * lambda_ * tau) - 0.458333333333333 * lambda_ ** 4 * lambda_svensson ** 2 * sigma_41 ** 2 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.666666666666667 * lambda_ ** 4 * lambda_svensson ** 2 * sigma_41 ** 2 * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.208333333333333 * lambda_ ** 4 * lambda_svensson ** 2 * sigma_41 ** 2 * exp(
            2 * lambda_ * tau) - 0.458333333333333 * lambda_ ** 4 * lambda_svensson ** 2 * sigma_42 ** 2 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.666666666666667 * lambda_ ** 4 * lambda_svensson ** 2 * sigma_42 ** 2 * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.208333333333333 * lambda_ ** 4 * lambda_svensson ** 2 * sigma_42 ** 2 * exp(
            2 * lambda_ * tau) - 0.458333333333333 * lambda_ ** 4 * lambda_svensson ** 2 * sigma_43 ** 2 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.666666666666667 * lambda_ ** 4 * lambda_svensson ** 2 * sigma_43 ** 2 * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.208333333333333 * lambda_ ** 4 * lambda_svensson ** 2 * sigma_43 ** 2 * exp(
            2 * lambda_ * tau) - 0.458333333333333 * lambda_ ** 4 * lambda_svensson ** 2 * sigma_44 ** 2 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.666666666666667 * lambda_ ** 4 * lambda_svensson ** 2 * sigma_44 ** 2 * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.208333333333333 * lambda_ ** 4 * lambda_svensson ** 2 * sigma_44 ** 2 * exp(
            2 * lambda_ * tau) + 0.0185185185185185 * lambda_ ** 3 * lambda_svensson ** 6 * sigma_00 ** 2 * tau ** 3 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.166666666666667 * lambda_ ** 3 * lambda_svensson ** 5 * sigma_00 * sigma_10 * tau ** 2 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.0555555555555556 * lambda_ ** 3 * lambda_svensson ** 5 * sigma_00 * sigma_20 * tau ** 2 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.166666666666667 * lambda_ ** 3 * lambda_svensson ** 5 * sigma_00 * sigma_30 * tau ** 2 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.333333333333333 * lambda_ ** 3 * lambda_svensson ** 5 * sigma_00 * sigma_30 * tau ** 2 * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.0555555555555556 * lambda_ ** 3 * lambda_svensson ** 5 * sigma_00 * sigma_40 * tau ** 2 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.111111111111111 * lambda_ ** 3 * lambda_svensson ** 5 * sigma_00 * sigma_40 * tau ** 2 * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.0833333333333333 * lambda_ ** 3 * lambda_svensson ** 5 * sigma_30 ** 2 * tau ** 2 * exp(
            2 * lambda_svensson * tau) - 0.111111111111111 * lambda_ ** 3 * lambda_svensson ** 5 * sigma_30 * sigma_40 * tau ** 2 * exp(
            lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.0833333333333333 * lambda_ ** 3 * lambda_svensson ** 5 * sigma_31 ** 2 * tau ** 2 * exp(
            2 * lambda_svensson * tau) - 0.111111111111111 * lambda_ ** 3 * lambda_svensson ** 5 * sigma_31 * sigma_41 * tau ** 2 * exp(
            lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.0833333333333333 * lambda_ ** 3 * lambda_svensson ** 5 * sigma_32 ** 2 * tau ** 2 * exp(
            2 * lambda_svensson * tau) - 0.111111111111111 * lambda_ ** 3 * lambda_svensson ** 5 * sigma_32 * sigma_42 * tau ** 2 * exp(
            lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.0833333333333333 * lambda_ ** 3 * lambda_svensson ** 5 * sigma_33 ** 2 * tau ** 2 * exp(
            2 * lambda_svensson * tau) - 0.111111111111111 * lambda_ ** 3 * lambda_svensson ** 5 * sigma_33 * sigma_43 * tau ** 2 * exp(
            lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.0277777777777778 * lambda_ ** 3 * lambda_svensson ** 5 * sigma_40 ** 2 * tau ** 2 * exp(
            2 * lambda_ * tau) - 0.0277777777777778 * lambda_ ** 3 * lambda_svensson ** 5 * sigma_41 ** 2 * tau ** 2 * exp(
            2 * lambda_ * tau) - 0.0277777777777778 * lambda_ ** 3 * lambda_svensson ** 5 * sigma_42 ** 2 * tau ** 2 * exp(
            2 * lambda_ * tau) - 0.0277777777777778 * lambda_ ** 3 * lambda_svensson ** 5 * sigma_43 ** 2 * tau ** 2 * exp(
            2 * lambda_ * tau) - 0.0277777777777778 * lambda_ ** 3 * lambda_svensson ** 5 * sigma_44 ** 2 * tau ** 2 * exp(
            2 * lambda_ * tau) + 0.333333333333333 * lambda_ ** 3 * lambda_svensson ** 4 * sigma_00 * sigma_10 * tau * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.111111111111111 * lambda_ ** 3 * lambda_svensson ** 4 * sigma_00 * sigma_20 * tau * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) + 1.0 * lambda_ ** 3 * lambda_svensson ** 4 * sigma_00 * sigma_30 * tau * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.333333333333333 * lambda_ ** 3 * lambda_svensson ** 4 * sigma_00 * sigma_40 * tau * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) + 0.166666666666667 * lambda_ ** 3 * lambda_svensson ** 4 * sigma_10 ** 2 * tau * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.333333333333333 * lambda_ ** 3 * lambda_svensson ** 4 * sigma_10 * sigma_20 * tau * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.333333333333333 * lambda_ ** 3 * lambda_svensson ** 4 * sigma_10 * sigma_30 * tau * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.333333333333333 * lambda_ ** 3 * lambda_svensson ** 4 * sigma_10 * sigma_30 * tau * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) - 0.166666666666667 * lambda_ ** 3 * lambda_svensson ** 4 * sigma_10 * sigma_30 * tau * exp(
            2 * lambda_svensson * tau) + 0.333333333333333 * lambda_ ** 3 * lambda_svensson ** 4 * sigma_10 * sigma_40 * tau * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.333333333333333 * lambda_ ** 3 * lambda_svensson ** 4 * sigma_10 * sigma_40 * tau * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.222222222222222 * lambda_ ** 3 * lambda_svensson ** 4 * sigma_10 * sigma_40 * tau * exp(
            lambda_ * tau) * exp(
            lambda_svensson * tau) + 0.166666666666667 * lambda_ ** 3 * lambda_svensson ** 4 * sigma_11 ** 2 * tau * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.333333333333333 * lambda_ ** 3 * lambda_svensson ** 4 * sigma_11 * sigma_21 * tau * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.333333333333333 * lambda_ ** 3 * lambda_svensson ** 4 * sigma_11 * sigma_31 * tau * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.333333333333333 * lambda_ ** 3 * lambda_svensson ** 4 * sigma_11 * sigma_31 * tau * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) - 0.166666666666667 * lambda_ ** 3 * lambda_svensson ** 4 * sigma_11 * sigma_31 * tau * exp(
            2 * lambda_svensson * tau) + 0.333333333333333 * lambda_ ** 3 * lambda_svensson ** 4 * sigma_11 * sigma_41 * tau * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.333333333333333 * lambda_ ** 3 * lambda_svensson ** 4 * sigma_11 * sigma_41 * tau * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.222222222222222 * lambda_ ** 3 * lambda_svensson ** 4 * sigma_11 * sigma_41 * tau * exp(
            lambda_ * tau) * exp(
            lambda_svensson * tau) + 0.0555555555555556 * lambda_ ** 3 * lambda_svensson ** 4 * sigma_20 ** 2 * tau * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.333333333333333 * lambda_ ** 3 * lambda_svensson ** 4 * sigma_20 * sigma_30 * tau * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.333333333333333 * lambda_ ** 3 * lambda_svensson ** 4 * sigma_20 * sigma_30 * tau * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) - 0.111111111111111 * lambda_ ** 3 * lambda_svensson ** 4 * sigma_20 * sigma_30 * tau * exp(
            lambda_ * tau) * exp(
            lambda_svensson * tau) + 0.111111111111111 * lambda_ ** 3 * lambda_svensson ** 4 * sigma_20 * sigma_40 * tau * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.111111111111111 * lambda_ ** 3 * lambda_svensson ** 4 * sigma_20 * sigma_40 * tau * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.0555555555555556 * lambda_ ** 3 * lambda_svensson ** 4 * sigma_20 * sigma_40 * tau * exp(
            2 * lambda_ * tau) + 0.0555555555555556 * lambda_ ** 3 * lambda_svensson ** 4 * sigma_21 ** 2 * tau * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.333333333333333 * lambda_ ** 3 * lambda_svensson ** 4 * sigma_21 * sigma_31 * tau * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.333333333333333 * lambda_ ** 3 * lambda_svensson ** 4 * sigma_21 * sigma_31 * tau * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) - 0.111111111111111 * lambda_ ** 3 * lambda_svensson ** 4 * sigma_21 * sigma_31 * tau * exp(
            lambda_ * tau) * exp(
            lambda_svensson * tau) + 0.111111111111111 * lambda_ ** 3 * lambda_svensson ** 4 * sigma_21 * sigma_41 * tau * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.111111111111111 * lambda_ ** 3 * lambda_svensson ** 4 * sigma_21 * sigma_41 * tau * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.0555555555555556 * lambda_ ** 3 * lambda_svensson ** 4 * sigma_21 * sigma_41 * tau * exp(
            2 * lambda_ * tau) + 0.0555555555555556 * lambda_ ** 3 * lambda_svensson ** 4 * sigma_22 ** 2 * tau * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.333333333333333 * lambda_ ** 3 * lambda_svensson ** 4 * sigma_22 * sigma_32 * tau * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.333333333333333 * lambda_ ** 3 * lambda_svensson ** 4 * sigma_22 * sigma_32 * tau * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) - 0.111111111111111 * lambda_ ** 3 * lambda_svensson ** 4 * sigma_22 * sigma_32 * tau * exp(
            lambda_ * tau) * exp(
            lambda_svensson * tau) + 0.111111111111111 * lambda_ ** 3 * lambda_svensson ** 4 * sigma_22 * sigma_42 * tau * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.111111111111111 * lambda_ ** 3 * lambda_svensson ** 4 * sigma_22 * sigma_42 * tau * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.0555555555555556 * lambda_ ** 3 * lambda_svensson ** 4 * sigma_22 * sigma_42 * tau * exp(
            2 * lambda_ * tau) + 0.166666666666667 * lambda_ ** 3 * lambda_svensson ** 4 * sigma_30 ** 2 * tau * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.333333333333333 * lambda_ ** 3 * lambda_svensson ** 4 * sigma_30 ** 2 * tau * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) - 0.25 * lambda_ ** 3 * lambda_svensson ** 4 * sigma_30 ** 2 * tau * exp(
            2 * lambda_svensson * tau) + 0.333333333333333 * lambda_ ** 3 * lambda_svensson ** 4 * sigma_30 * sigma_40 * tau * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.333333333333333 * lambda_ ** 3 * lambda_svensson ** 4 * sigma_30 * sigma_40 * tau * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) + 0.333333333333333 * lambda_ ** 3 * lambda_svensson ** 4 * sigma_30 * sigma_40 * tau * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) - 0.555555555555556 * lambda_ ** 3 * lambda_svensson ** 4 * sigma_30 * sigma_40 * tau * exp(
            lambda_ * tau) * exp(
            lambda_svensson * tau) + 0.166666666666667 * lambda_ ** 3 * lambda_svensson ** 4 * sigma_31 ** 2 * tau * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.333333333333333 * lambda_ ** 3 * lambda_svensson ** 4 * sigma_31 ** 2 * tau * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) - 0.25 * lambda_ ** 3 * lambda_svensson ** 4 * sigma_31 ** 2 * tau * exp(
            2 * lambda_svensson * tau) + 0.333333333333333 * lambda_ ** 3 * lambda_svensson ** 4 * sigma_31 * sigma_41 * tau * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.333333333333333 * lambda_ ** 3 * lambda_svensson ** 4 * sigma_31 * sigma_41 * tau * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) + 0.333333333333333 * lambda_ ** 3 * lambda_svensson ** 4 * sigma_31 * sigma_41 * tau * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) - 0.555555555555556 * lambda_ ** 3 * lambda_svensson ** 4 * sigma_31 * sigma_41 * tau * exp(
            lambda_ * tau) * exp(
            lambda_svensson * tau) + 0.166666666666667 * lambda_ ** 3 * lambda_svensson ** 4 * sigma_32 ** 2 * tau * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.333333333333333 * lambda_ ** 3 * lambda_svensson ** 4 * sigma_32 ** 2 * tau * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) - 0.25 * lambda_ ** 3 * lambda_svensson ** 4 * sigma_32 ** 2 * tau * exp(
            2 * lambda_svensson * tau) + 0.333333333333333 * lambda_ ** 3 * lambda_svensson ** 4 * sigma_32 * sigma_42 * tau * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.333333333333333 * lambda_ ** 3 * lambda_svensson ** 4 * sigma_32 * sigma_42 * tau * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) + 0.333333333333333 * lambda_ ** 3 * lambda_svensson ** 4 * sigma_32 * sigma_42 * tau * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) - 0.555555555555556 * lambda_ ** 3 * lambda_svensson ** 4 * sigma_32 * sigma_42 * tau * exp(
            lambda_ * tau) * exp(
            lambda_svensson * tau) + 0.166666666666667 * lambda_ ** 3 * lambda_svensson ** 4 * sigma_33 ** 2 * tau * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.333333333333333 * lambda_ ** 3 * lambda_svensson ** 4 * sigma_33 ** 2 * tau * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) - 0.25 * lambda_ ** 3 * lambda_svensson ** 4 * sigma_33 ** 2 * tau * exp(
            2 * lambda_svensson * tau) + 0.333333333333333 * lambda_ ** 3 * lambda_svensson ** 4 * sigma_33 * sigma_43 * tau * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.333333333333333 * lambda_ ** 3 * lambda_svensson ** 4 * sigma_33 * sigma_43 * tau * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) + 0.333333333333333 * lambda_ ** 3 * lambda_svensson ** 4 * sigma_33 * sigma_43 * tau * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) - 0.555555555555556 * lambda_ ** 3 * lambda_svensson ** 4 * sigma_33 * sigma_43 * tau * exp(
            lambda_ * tau) * exp(
            lambda_svensson * tau) + 0.0555555555555556 * lambda_ ** 3 * lambda_svensson ** 4 * sigma_40 ** 2 * tau * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.111111111111111 * lambda_ ** 3 * lambda_svensson ** 4 * sigma_40 ** 2 * tau * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.0833333333333333 * lambda_ ** 3 * lambda_svensson ** 4 * sigma_40 ** 2 * tau * exp(
            2 * lambda_ * tau) + 0.0555555555555556 * lambda_ ** 3 * lambda_svensson ** 4 * sigma_41 ** 2 * tau * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.111111111111111 * lambda_ ** 3 * lambda_svensson ** 4 * sigma_41 ** 2 * tau * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.0833333333333333 * lambda_ ** 3 * lambda_svensson ** 4 * sigma_41 ** 2 * tau * exp(
            2 * lambda_ * tau) + 0.0555555555555556 * lambda_ ** 3 * lambda_svensson ** 4 * sigma_42 ** 2 * tau * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.111111111111111 * lambda_ ** 3 * lambda_svensson ** 4 * sigma_42 ** 2 * tau * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.0833333333333333 * lambda_ ** 3 * lambda_svensson ** 4 * sigma_42 ** 2 * tau * exp(
            2 * lambda_ * tau) + 0.0555555555555556 * lambda_ ** 3 * lambda_svensson ** 4 * sigma_43 ** 2 * tau * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.111111111111111 * lambda_ ** 3 * lambda_svensson ** 4 * sigma_43 ** 2 * tau * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.0833333333333333 * lambda_ ** 3 * lambda_svensson ** 4 * sigma_43 ** 2 * tau * exp(
            2 * lambda_ * tau) + 0.0555555555555556 * lambda_ ** 3 * lambda_svensson ** 4 * sigma_44 ** 2 * tau * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.111111111111111 * lambda_ ** 3 * lambda_svensson ** 4 * sigma_44 ** 2 * tau * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.0833333333333333 * lambda_ ** 3 * lambda_svensson ** 4 * sigma_44 ** 2 * tau * exp(
            2 * lambda_ * tau) - 0.111111111111111 * lambda_ ** 3 * lambda_svensson ** 3 * sigma_00 * sigma_10 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.111111111111111 * lambda_ ** 3 * lambda_svensson ** 3 * sigma_00 * sigma_10 * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) - 0.111111111111111 * lambda_ ** 3 * lambda_svensson ** 3 * sigma_00 * sigma_20 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.111111111111111 * lambda_ ** 3 * lambda_svensson ** 3 * sigma_00 * sigma_20 * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.333333333333333 * lambda_ ** 3 * lambda_svensson ** 3 * sigma_00 * sigma_30 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.333333333333333 * lambda_ ** 3 * lambda_svensson ** 3 * sigma_00 * sigma_30 * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) - 0.333333333333333 * lambda_ ** 3 * lambda_svensson ** 3 * sigma_00 * sigma_40 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.333333333333333 * lambda_ ** 3 * lambda_svensson ** 3 * sigma_00 * sigma_40 * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.0833333333333333 * lambda_ ** 3 * lambda_svensson ** 3 * sigma_10 ** 2 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.111111111111111 * lambda_ ** 3 * lambda_svensson ** 3 * sigma_10 ** 2 * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) - 0.0277777777777778 * lambda_ ** 3 * lambda_svensson ** 3 * sigma_10 ** 2 * exp(
            2 * lambda_svensson * tau) - 0.444444444444444 * lambda_ ** 3 * lambda_svensson ** 3 * sigma_10 * sigma_20 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.333333333333333 * lambda_ ** 3 * lambda_svensson ** 3 * sigma_10 * sigma_20 * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) + 0.333333333333333 * lambda_ ** 3 * lambda_svensson ** 3 * sigma_10 * sigma_20 * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) - 0.222222222222222 * lambda_ ** 3 * lambda_svensson ** 3 * sigma_10 * sigma_20 * exp(
            lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.25 * lambda_ ** 3 * lambda_svensson ** 3 * sigma_10 * sigma_30 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.333333333333333 * lambda_ ** 3 * lambda_svensson ** 3 * sigma_10 * sigma_30 * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) - 0.0833333333333333 * lambda_ ** 3 * lambda_svensson ** 3 * sigma_10 * sigma_30 * exp(
            2 * lambda_svensson * tau) - 0.666666666666667 * lambda_ ** 3 * lambda_svensson ** 3 * sigma_10 * sigma_40 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.666666666666667 * lambda_ ** 3 * lambda_svensson ** 3 * sigma_10 * sigma_40 * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) + 0.333333333333333 * lambda_ ** 3 * lambda_svensson ** 3 * sigma_10 * sigma_40 * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) - 0.333333333333333 * lambda_ ** 3 * lambda_svensson ** 3 * sigma_10 * sigma_40 * exp(
            lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.0833333333333333 * lambda_ ** 3 * lambda_svensson ** 3 * sigma_11 ** 2 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.111111111111111 * lambda_ ** 3 * lambda_svensson ** 3 * sigma_11 ** 2 * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) - 0.0277777777777778 * lambda_ ** 3 * lambda_svensson ** 3 * sigma_11 ** 2 * exp(
            2 * lambda_svensson * tau) - 0.444444444444444 * lambda_ ** 3 * lambda_svensson ** 3 * sigma_11 * sigma_21 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.333333333333333 * lambda_ ** 3 * lambda_svensson ** 3 * sigma_11 * sigma_21 * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) + 0.333333333333333 * lambda_ ** 3 * lambda_svensson ** 3 * sigma_11 * sigma_21 * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) - 0.222222222222222 * lambda_ ** 3 * lambda_svensson ** 3 * sigma_11 * sigma_21 * exp(
            lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.25 * lambda_ ** 3 * lambda_svensson ** 3 * sigma_11 * sigma_31 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.333333333333333 * lambda_ ** 3 * lambda_svensson ** 3 * sigma_11 * sigma_31 * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) - 0.0833333333333333 * lambda_ ** 3 * lambda_svensson ** 3 * sigma_11 * sigma_31 * exp(
            2 * lambda_svensson * tau) - 0.666666666666667 * lambda_ ** 3 * lambda_svensson ** 3 * sigma_11 * sigma_41 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.666666666666667 * lambda_ ** 3 * lambda_svensson ** 3 * sigma_11 * sigma_41 * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) + 0.333333333333333 * lambda_ ** 3 * lambda_svensson ** 3 * sigma_11 * sigma_41 * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) - 0.333333333333333 * lambda_ ** 3 * lambda_svensson ** 3 * sigma_11 * sigma_41 * exp(
            lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.0833333333333333 * lambda_ ** 3 * lambda_svensson ** 3 * sigma_20 ** 2 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.111111111111111 * lambda_ ** 3 * lambda_svensson ** 3 * sigma_20 ** 2 * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.0277777777777778 * lambda_ ** 3 * lambda_svensson ** 3 * sigma_20 ** 2 * exp(
            2 * lambda_ * tau) - 0.666666666666667 * lambda_ ** 3 * lambda_svensson ** 3 * sigma_20 * sigma_30 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.333333333333333 * lambda_ ** 3 * lambda_svensson ** 3 * sigma_20 * sigma_30 * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) + 0.666666666666667 * lambda_ ** 3 * lambda_svensson ** 3 * sigma_20 * sigma_30 * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) - 0.333333333333333 * lambda_ ** 3 * lambda_svensson ** 3 * sigma_20 * sigma_30 * exp(
            lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.25 * lambda_ ** 3 * lambda_svensson ** 3 * sigma_20 * sigma_40 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.333333333333333 * lambda_ ** 3 * lambda_svensson ** 3 * sigma_20 * sigma_40 * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.0833333333333333 * lambda_ ** 3 * lambda_svensson ** 3 * sigma_20 * sigma_40 * exp(
            2 * lambda_ * tau) - 0.0833333333333333 * lambda_ ** 3 * lambda_svensson ** 3 * sigma_21 ** 2 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.111111111111111 * lambda_ ** 3 * lambda_svensson ** 3 * sigma_21 ** 2 * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.0277777777777778 * lambda_ ** 3 * lambda_svensson ** 3 * sigma_21 ** 2 * exp(
            2 * lambda_ * tau) - 0.666666666666667 * lambda_ ** 3 * lambda_svensson ** 3 * sigma_21 * sigma_31 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.333333333333333 * lambda_ ** 3 * lambda_svensson ** 3 * sigma_21 * sigma_31 * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) + 0.666666666666667 * lambda_ ** 3 * lambda_svensson ** 3 * sigma_21 * sigma_31 * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) - 0.333333333333333 * lambda_ ** 3 * lambda_svensson ** 3 * sigma_21 * sigma_31 * exp(
            lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.25 * lambda_ ** 3 * lambda_svensson ** 3 * sigma_21 * sigma_41 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.333333333333333 * lambda_ ** 3 * lambda_svensson ** 3 * sigma_21 * sigma_41 * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.0833333333333333 * lambda_ ** 3 * lambda_svensson ** 3 * sigma_21 * sigma_41 * exp(
            2 * lambda_ * tau) - 0.0833333333333333 * lambda_ ** 3 * lambda_svensson ** 3 * sigma_22 ** 2 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.111111111111111 * lambda_ ** 3 * lambda_svensson ** 3 * sigma_22 ** 2 * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.0277777777777778 * lambda_ ** 3 * lambda_svensson ** 3 * sigma_22 ** 2 * exp(
            2 * lambda_ * tau) - 0.666666666666667 * lambda_ ** 3 * lambda_svensson ** 3 * sigma_22 * sigma_32 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.333333333333333 * lambda_ ** 3 * lambda_svensson ** 3 * sigma_22 * sigma_32 * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) + 0.666666666666667 * lambda_ ** 3 * lambda_svensson ** 3 * sigma_22 * sigma_32 * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) - 0.333333333333333 * lambda_ ** 3 * lambda_svensson ** 3 * sigma_22 * sigma_32 * exp(
            lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.25 * lambda_ ** 3 * lambda_svensson ** 3 * sigma_22 * sigma_42 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.333333333333333 * lambda_ ** 3 * lambda_svensson ** 3 * sigma_22 * sigma_42 * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.0833333333333333 * lambda_ ** 3 * lambda_svensson ** 3 * sigma_22 * sigma_42 * exp(
            2 * lambda_ * tau) - 0.152777777777778 * lambda_ ** 3 * lambda_svensson ** 3 * sigma_30 ** 2 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.222222222222222 * lambda_ ** 3 * lambda_svensson ** 3 * sigma_30 ** 2 * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) - 0.0694444444444444 * lambda_ ** 3 * lambda_svensson ** 3 * sigma_30 ** 2 * exp(
            2 * lambda_svensson * tau) - 0.666666666666667 * lambda_ ** 3 * lambda_svensson ** 3 * sigma_30 * sigma_40 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.666666666666667 * lambda_ ** 3 * lambda_svensson ** 3 * sigma_30 * sigma_40 * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) + 0.666666666666667 * lambda_ ** 3 * lambda_svensson ** 3 * sigma_30 * sigma_40 * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) - 0.666666666666667 * lambda_ ** 3 * lambda_svensson ** 3 * sigma_30 * sigma_40 * exp(
            lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.152777777777778 * lambda_ ** 3 * lambda_svensson ** 3 * sigma_31 ** 2 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.222222222222222 * lambda_ ** 3 * lambda_svensson ** 3 * sigma_31 ** 2 * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) - 0.0694444444444444 * lambda_ ** 3 * lambda_svensson ** 3 * sigma_31 ** 2 * exp(
            2 * lambda_svensson * tau) - 0.666666666666667 * lambda_ ** 3 * lambda_svensson ** 3 * sigma_31 * sigma_41 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.666666666666667 * lambda_ ** 3 * lambda_svensson ** 3 * sigma_31 * sigma_41 * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) + 0.666666666666667 * lambda_ ** 3 * lambda_svensson ** 3 * sigma_31 * sigma_41 * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) - 0.666666666666667 * lambda_ ** 3 * lambda_svensson ** 3 * sigma_31 * sigma_41 * exp(
            lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.152777777777778 * lambda_ ** 3 * lambda_svensson ** 3 * sigma_32 ** 2 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.222222222222222 * lambda_ ** 3 * lambda_svensson ** 3 * sigma_32 ** 2 * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) - 0.0694444444444444 * lambda_ ** 3 * lambda_svensson ** 3 * sigma_32 ** 2 * exp(
            2 * lambda_svensson * tau) - 0.666666666666667 * lambda_ ** 3 * lambda_svensson ** 3 * sigma_32 * sigma_42 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.666666666666667 * lambda_ ** 3 * lambda_svensson ** 3 * sigma_32 * sigma_42 * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) + 0.666666666666667 * lambda_ ** 3 * lambda_svensson ** 3 * sigma_32 * sigma_42 * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) - 0.666666666666667 * lambda_ ** 3 * lambda_svensson ** 3 * sigma_32 * sigma_42 * exp(
            lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.152777777777778 * lambda_ ** 3 * lambda_svensson ** 3 * sigma_33 ** 2 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.222222222222222 * lambda_ ** 3 * lambda_svensson ** 3 * sigma_33 ** 2 * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) - 0.0694444444444444 * lambda_ ** 3 * lambda_svensson ** 3 * sigma_33 ** 2 * exp(
            2 * lambda_svensson * tau) - 0.666666666666667 * lambda_ ** 3 * lambda_svensson ** 3 * sigma_33 * sigma_43 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.666666666666667 * lambda_ ** 3 * lambda_svensson ** 3 * sigma_33 * sigma_43 * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) + 0.666666666666667 * lambda_ ** 3 * lambda_svensson ** 3 * sigma_33 * sigma_43 * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) - 0.666666666666667 * lambda_ ** 3 * lambda_svensson ** 3 * sigma_33 * sigma_43 * exp(
            lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.152777777777778 * lambda_ ** 3 * lambda_svensson ** 3 * sigma_40 ** 2 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.222222222222222 * lambda_ ** 3 * lambda_svensson ** 3 * sigma_40 ** 2 * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.0694444444444444 * lambda_ ** 3 * lambda_svensson ** 3 * sigma_40 ** 2 * exp(
            2 * lambda_ * tau) - 0.152777777777778 * lambda_ ** 3 * lambda_svensson ** 3 * sigma_41 ** 2 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.222222222222222 * lambda_ ** 3 * lambda_svensson ** 3 * sigma_41 ** 2 * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.0694444444444444 * lambda_ ** 3 * lambda_svensson ** 3 * sigma_41 ** 2 * exp(
            2 * lambda_ * tau) - 0.152777777777778 * lambda_ ** 3 * lambda_svensson ** 3 * sigma_42 ** 2 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.222222222222222 * lambda_ ** 3 * lambda_svensson ** 3 * sigma_42 ** 2 * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.0694444444444444 * lambda_ ** 3 * lambda_svensson ** 3 * sigma_42 ** 2 * exp(
            2 * lambda_ * tau) - 0.152777777777778 * lambda_ ** 3 * lambda_svensson ** 3 * sigma_43 ** 2 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.222222222222222 * lambda_ ** 3 * lambda_svensson ** 3 * sigma_43 ** 2 * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.0694444444444444 * lambda_ ** 3 * lambda_svensson ** 3 * sigma_43 ** 2 * exp(
            2 * lambda_ * tau) - 0.152777777777778 * lambda_ ** 3 * lambda_svensson ** 3 * sigma_44 ** 2 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.222222222222222 * lambda_ ** 3 * lambda_svensson ** 3 * sigma_44 ** 2 * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.0694444444444444 * lambda_ ** 3 * lambda_svensson ** 3 * sigma_44 ** 2 * exp(
            2 * lambda_ * tau) + 0.0555555555555556 * lambda_ ** 2 * lambda_svensson ** 6 * sigma_00 * sigma_10 * tau ** 2 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.0555555555555556 * lambda_ ** 2 * lambda_svensson ** 6 * sigma_00 * sigma_30 * tau ** 2 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.111111111111111 * lambda_ ** 2 * lambda_svensson ** 6 * sigma_00 * sigma_30 * tau ** 2 * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) - 0.0277777777777778 * lambda_ ** 2 * lambda_svensson ** 6 * sigma_30 ** 2 * tau ** 2 * exp(
            2 * lambda_svensson * tau) - 0.0277777777777778 * lambda_ ** 2 * lambda_svensson ** 6 * sigma_31 ** 2 * tau ** 2 * exp(
            2 * lambda_svensson * tau) - 0.0277777777777778 * lambda_ ** 2 * lambda_svensson ** 6 * sigma_32 ** 2 * tau ** 2 * exp(
            2 * lambda_svensson * tau) - 0.0277777777777778 * lambda_ ** 2 * lambda_svensson ** 6 * sigma_33 ** 2 * tau ** 2 * exp(
            2 * lambda_svensson * tau) + 0.333333333333333 * lambda_ ** 2 * lambda_svensson ** 5 * sigma_00 * sigma_10 * tau * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 1.0 * lambda_ ** 2 * lambda_svensson ** 5 * sigma_00 * sigma_30 * tau * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.166666666666667 * lambda_ ** 2 * lambda_svensson ** 5 * sigma_10 ** 2 * tau * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.111111111111111 * lambda_ ** 2 * lambda_svensson ** 5 * sigma_10 * sigma_20 * tau * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.333333333333333 * lambda_ ** 2 * lambda_svensson ** 5 * sigma_10 * sigma_30 * tau * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.333333333333333 * lambda_ ** 2 * lambda_svensson ** 5 * sigma_10 * sigma_30 * tau * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) - 0.166666666666667 * lambda_ ** 2 * lambda_svensson ** 5 * sigma_10 * sigma_30 * tau * exp(
            2 * lambda_svensson * tau) + 0.111111111111111 * lambda_ ** 2 * lambda_svensson ** 5 * sigma_10 * sigma_40 * tau * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.111111111111111 * lambda_ ** 2 * lambda_svensson ** 5 * sigma_10 * sigma_40 * tau * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.111111111111111 * lambda_ ** 2 * lambda_svensson ** 5 * sigma_10 * sigma_40 * tau * exp(
            lambda_ * tau) * exp(
            lambda_svensson * tau) + 0.166666666666667 * lambda_ ** 2 * lambda_svensson ** 5 * sigma_11 ** 2 * tau * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.111111111111111 * lambda_ ** 2 * lambda_svensson ** 5 * sigma_11 * sigma_21 * tau * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.333333333333333 * lambda_ ** 2 * lambda_svensson ** 5 * sigma_11 * sigma_31 * tau * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.333333333333333 * lambda_ ** 2 * lambda_svensson ** 5 * sigma_11 * sigma_31 * tau * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) - 0.166666666666667 * lambda_ ** 2 * lambda_svensson ** 5 * sigma_11 * sigma_31 * tau * exp(
            2 * lambda_svensson * tau) + 0.111111111111111 * lambda_ ** 2 * lambda_svensson ** 5 * sigma_11 * sigma_41 * tau * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.111111111111111 * lambda_ ** 2 * lambda_svensson ** 5 * sigma_11 * sigma_41 * tau * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.111111111111111 * lambda_ ** 2 * lambda_svensson ** 5 * sigma_11 * sigma_41 * tau * exp(
            lambda_ * tau) * exp(
            lambda_svensson * tau) + 0.111111111111111 * lambda_ ** 2 * lambda_svensson ** 5 * sigma_20 * sigma_30 * tau * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.111111111111111 * lambda_ ** 2 * lambda_svensson ** 5 * sigma_20 * sigma_30 * tau * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.111111111111111 * lambda_ ** 2 * lambda_svensson ** 5 * sigma_21 * sigma_31 * tau * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.111111111111111 * lambda_ ** 2 * lambda_svensson ** 5 * sigma_21 * sigma_31 * tau * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.111111111111111 * lambda_ ** 2 * lambda_svensson ** 5 * sigma_22 * sigma_32 * tau * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.111111111111111 * lambda_ ** 2 * lambda_svensson ** 5 * sigma_22 * sigma_32 * tau * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.166666666666667 * lambda_ ** 2 * lambda_svensson ** 5 * sigma_30 ** 2 * tau * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.333333333333333 * lambda_ ** 2 * lambda_svensson ** 5 * sigma_30 ** 2 * tau * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) - 0.25 * lambda_ ** 2 * lambda_svensson ** 5 * sigma_30 ** 2 * tau * exp(
            2 * lambda_svensson * tau) + 0.111111111111111 * lambda_ ** 2 * lambda_svensson ** 5 * sigma_30 * sigma_40 * tau * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.111111111111111 * lambda_ ** 2 * lambda_svensson ** 5 * sigma_30 * sigma_40 * tau * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) + 0.111111111111111 * lambda_ ** 2 * lambda_svensson ** 5 * sigma_30 * sigma_40 * tau * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) - 0.111111111111111 * lambda_ ** 2 * lambda_svensson ** 5 * sigma_30 * sigma_40 * tau * exp(
            lambda_ * tau) * exp(
            lambda_svensson * tau) + 0.166666666666667 * lambda_ ** 2 * lambda_svensson ** 5 * sigma_31 ** 2 * tau * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.333333333333333 * lambda_ ** 2 * lambda_svensson ** 5 * sigma_31 ** 2 * tau * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) - 0.25 * lambda_ ** 2 * lambda_svensson ** 5 * sigma_31 ** 2 * tau * exp(
            2 * lambda_svensson * tau) + 0.111111111111111 * lambda_ ** 2 * lambda_svensson ** 5 * sigma_31 * sigma_41 * tau * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.111111111111111 * lambda_ ** 2 * lambda_svensson ** 5 * sigma_31 * sigma_41 * tau * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) + 0.111111111111111 * lambda_ ** 2 * lambda_svensson ** 5 * sigma_31 * sigma_41 * tau * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) - 0.111111111111111 * lambda_ ** 2 * lambda_svensson ** 5 * sigma_31 * sigma_41 * tau * exp(
            lambda_ * tau) * exp(
            lambda_svensson * tau) + 0.166666666666667 * lambda_ ** 2 * lambda_svensson ** 5 * sigma_32 ** 2 * tau * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.333333333333333 * lambda_ ** 2 * lambda_svensson ** 5 * sigma_32 ** 2 * tau * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) - 0.25 * lambda_ ** 2 * lambda_svensson ** 5 * sigma_32 ** 2 * tau * exp(
            2 * lambda_svensson * tau) + 0.111111111111111 * lambda_ ** 2 * lambda_svensson ** 5 * sigma_32 * sigma_42 * tau * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.111111111111111 * lambda_ ** 2 * lambda_svensson ** 5 * sigma_32 * sigma_42 * tau * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) + 0.111111111111111 * lambda_ ** 2 * lambda_svensson ** 5 * sigma_32 * sigma_42 * tau * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) - 0.111111111111111 * lambda_ ** 2 * lambda_svensson ** 5 * sigma_32 * sigma_42 * tau * exp(
            lambda_ * tau) * exp(
            lambda_svensson * tau) + 0.166666666666667 * lambda_ ** 2 * lambda_svensson ** 5 * sigma_33 ** 2 * tau * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.333333333333333 * lambda_ ** 2 * lambda_svensson ** 5 * sigma_33 ** 2 * tau * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) - 0.25 * lambda_ ** 2 * lambda_svensson ** 5 * sigma_33 ** 2 * tau * exp(
            2 * lambda_svensson * tau) + 0.111111111111111 * lambda_ ** 2 * lambda_svensson ** 5 * sigma_33 * sigma_43 * tau * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.111111111111111 * lambda_ ** 2 * lambda_svensson ** 5 * sigma_33 * sigma_43 * tau * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) + 0.111111111111111 * lambda_ ** 2 * lambda_svensson ** 5 * sigma_33 * sigma_43 * tau * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) - 0.111111111111111 * lambda_ ** 2 * lambda_svensson ** 5 * sigma_33 * sigma_43 * tau * exp(
            lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.333333333333333 * lambda_ ** 2 * lambda_svensson ** 4 * sigma_00 * sigma_10 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.333333333333333 * lambda_ ** 2 * lambda_svensson ** 4 * sigma_00 * sigma_10 * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) - 1.0 * lambda_ ** 2 * lambda_svensson ** 4 * sigma_00 * sigma_30 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 1.0 * lambda_ ** 2 * lambda_svensson ** 4 * sigma_00 * sigma_30 * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) - 0.25 * lambda_ ** 2 * lambda_svensson ** 4 * sigma_10 ** 2 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.333333333333333 * lambda_ ** 2 * lambda_svensson ** 4 * sigma_10 ** 2 * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) - 0.0833333333333333 * lambda_ ** 2 * lambda_svensson ** 4 * sigma_10 ** 2 * exp(
            2 * lambda_svensson * tau) - 0.333333333333333 * lambda_ ** 2 * lambda_svensson ** 4 * sigma_10 * sigma_20 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.111111111111111 * lambda_ ** 2 * lambda_svensson ** 4 * sigma_10 * sigma_20 * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) + 0.333333333333333 * lambda_ ** 2 * lambda_svensson ** 4 * sigma_10 * sigma_20 * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) - 0.111111111111111 * lambda_ ** 2 * lambda_svensson ** 4 * sigma_10 * sigma_20 * exp(
            lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.75 * lambda_ ** 2 * lambda_svensson ** 4 * sigma_10 * sigma_30 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 1.0 * lambda_ ** 2 * lambda_svensson ** 4 * sigma_10 * sigma_30 * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) - 0.25 * lambda_ ** 2 * lambda_svensson ** 4 * sigma_10 * sigma_30 * exp(
            2 * lambda_svensson * tau) - 0.333333333333333 * lambda_ ** 2 * lambda_svensson ** 4 * sigma_10 * sigma_40 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.222222222222222 * lambda_ ** 2 * lambda_svensson ** 4 * sigma_10 * sigma_40 * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) + 0.333333333333333 * lambda_ ** 2 * lambda_svensson ** 4 * sigma_10 * sigma_40 * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) - 0.222222222222222 * lambda_ ** 2 * lambda_svensson ** 4 * sigma_10 * sigma_40 * exp(
            lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.25 * lambda_ ** 2 * lambda_svensson ** 4 * sigma_11 ** 2 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.333333333333333 * lambda_ ** 2 * lambda_svensson ** 4 * sigma_11 ** 2 * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) - 0.0833333333333333 * lambda_ ** 2 * lambda_svensson ** 4 * sigma_11 ** 2 * exp(
            2 * lambda_svensson * tau) - 0.333333333333333 * lambda_ ** 2 * lambda_svensson ** 4 * sigma_11 * sigma_21 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.111111111111111 * lambda_ ** 2 * lambda_svensson ** 4 * sigma_11 * sigma_21 * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) + 0.333333333333333 * lambda_ ** 2 * lambda_svensson ** 4 * sigma_11 * sigma_21 * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) - 0.111111111111111 * lambda_ ** 2 * lambda_svensson ** 4 * sigma_11 * sigma_21 * exp(
            lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.75 * lambda_ ** 2 * lambda_svensson ** 4 * sigma_11 * sigma_31 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 1.0 * lambda_ ** 2 * lambda_svensson ** 4 * sigma_11 * sigma_31 * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) - 0.25 * lambda_ ** 2 * lambda_svensson ** 4 * sigma_11 * sigma_31 * exp(
            2 * lambda_svensson * tau) - 0.333333333333333 * lambda_ ** 2 * lambda_svensson ** 4 * sigma_11 * sigma_41 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.222222222222222 * lambda_ ** 2 * lambda_svensson ** 4 * sigma_11 * sigma_41 * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) + 0.333333333333333 * lambda_ ** 2 * lambda_svensson ** 4 * sigma_11 * sigma_41 * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) - 0.222222222222222 * lambda_ ** 2 * lambda_svensson ** 4 * sigma_11 * sigma_41 * exp(
            lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.666666666666667 * lambda_ ** 2 * lambda_svensson ** 4 * sigma_20 * sigma_30 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.111111111111111 * lambda_ ** 2 * lambda_svensson ** 4 * sigma_20 * sigma_30 * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) + 0.666666666666667 * lambda_ ** 2 * lambda_svensson ** 4 * sigma_20 * sigma_30 * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) - 0.111111111111111 * lambda_ ** 2 * lambda_svensson ** 4 * sigma_20 * sigma_30 * exp(
            lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.666666666666667 * lambda_ ** 2 * lambda_svensson ** 4 * sigma_21 * sigma_31 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.111111111111111 * lambda_ ** 2 * lambda_svensson ** 4 * sigma_21 * sigma_31 * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) + 0.666666666666667 * lambda_ ** 2 * lambda_svensson ** 4 * sigma_21 * sigma_31 * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) - 0.111111111111111 * lambda_ ** 2 * lambda_svensson ** 4 * sigma_21 * sigma_31 * exp(
            lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.666666666666667 * lambda_ ** 2 * lambda_svensson ** 4 * sigma_22 * sigma_32 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.111111111111111 * lambda_ ** 2 * lambda_svensson ** 4 * sigma_22 * sigma_32 * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) + 0.666666666666667 * lambda_ ** 2 * lambda_svensson ** 4 * sigma_22 * sigma_32 * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) - 0.111111111111111 * lambda_ ** 2 * lambda_svensson ** 4 * sigma_22 * sigma_32 * exp(
            lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.458333333333333 * lambda_ ** 2 * lambda_svensson ** 4 * sigma_30 ** 2 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.666666666666667 * lambda_ ** 2 * lambda_svensson ** 4 * sigma_30 ** 2 * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) - 0.208333333333333 * lambda_ ** 2 * lambda_svensson ** 4 * sigma_30 ** 2 * exp(
            2 * lambda_svensson * tau) - 0.666666666666667 * lambda_ ** 2 * lambda_svensson ** 4 * sigma_30 * sigma_40 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.222222222222222 * lambda_ ** 2 * lambda_svensson ** 4 * sigma_30 * sigma_40 * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) + 0.666666666666667 * lambda_ ** 2 * lambda_svensson ** 4 * sigma_30 * sigma_40 * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) - 0.222222222222222 * lambda_ ** 2 * lambda_svensson ** 4 * sigma_30 * sigma_40 * exp(
            lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.458333333333333 * lambda_ ** 2 * lambda_svensson ** 4 * sigma_31 ** 2 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.666666666666667 * lambda_ ** 2 * lambda_svensson ** 4 * sigma_31 ** 2 * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) - 0.208333333333333 * lambda_ ** 2 * lambda_svensson ** 4 * sigma_31 ** 2 * exp(
            2 * lambda_svensson * tau) - 0.666666666666667 * lambda_ ** 2 * lambda_svensson ** 4 * sigma_31 * sigma_41 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.222222222222222 * lambda_ ** 2 * lambda_svensson ** 4 * sigma_31 * sigma_41 * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) + 0.666666666666667 * lambda_ ** 2 * lambda_svensson ** 4 * sigma_31 * sigma_41 * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) - 0.222222222222222 * lambda_ ** 2 * lambda_svensson ** 4 * sigma_31 * sigma_41 * exp(
            lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.458333333333333 * lambda_ ** 2 * lambda_svensson ** 4 * sigma_32 ** 2 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.666666666666667 * lambda_ ** 2 * lambda_svensson ** 4 * sigma_32 ** 2 * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) - 0.208333333333333 * lambda_ ** 2 * lambda_svensson ** 4 * sigma_32 ** 2 * exp(
            2 * lambda_svensson * tau) - 0.666666666666667 * lambda_ ** 2 * lambda_svensson ** 4 * sigma_32 * sigma_42 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.222222222222222 * lambda_ ** 2 * lambda_svensson ** 4 * sigma_32 * sigma_42 * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) + 0.666666666666667 * lambda_ ** 2 * lambda_svensson ** 4 * sigma_32 * sigma_42 * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) - 0.222222222222222 * lambda_ ** 2 * lambda_svensson ** 4 * sigma_32 * sigma_42 * exp(
            lambda_ * tau) * exp(
            lambda_svensson * tau) - 0.458333333333333 * lambda_ ** 2 * lambda_svensson ** 4 * sigma_33 ** 2 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.666666666666667 * lambda_ ** 2 * lambda_svensson ** 4 * sigma_33 ** 2 * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) - 0.208333333333333 * lambda_ ** 2 * lambda_svensson ** 4 * sigma_33 ** 2 * exp(
            2 * lambda_svensson * tau) - 0.666666666666667 * lambda_ ** 2 * lambda_svensson ** 4 * sigma_33 * sigma_43 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.222222222222222 * lambda_ ** 2 * lambda_svensson ** 4 * sigma_33 * sigma_43 * exp(
            2 * lambda_ * tau) * exp(
            lambda_svensson * tau) + 0.666666666666667 * lambda_ ** 2 * lambda_svensson ** 4 * sigma_33 * sigma_43 * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) - 0.222222222222222 * lambda_ ** 2 * lambda_svensson ** 4 * sigma_33 * sigma_43 * exp(
            lambda_ * tau) * exp(
            lambda_svensson * tau) + 0.111111111111111 * lambda_ * lambda_svensson ** 6 * sigma_00 * sigma_10 * tau * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.333333333333333 * lambda_ * lambda_svensson ** 6 * sigma_00 * sigma_30 * tau * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.0555555555555556 * lambda_ * lambda_svensson ** 6 * sigma_10 ** 2 * tau * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.111111111111111 * lambda_ * lambda_svensson ** 6 * sigma_10 * sigma_30 * tau * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.111111111111111 * lambda_ * lambda_svensson ** 6 * sigma_10 * sigma_30 * tau * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) - 0.0555555555555556 * lambda_ * lambda_svensson ** 6 * sigma_10 * sigma_30 * tau * exp(
            2 * lambda_svensson * tau) + 0.0555555555555556 * lambda_ * lambda_svensson ** 6 * sigma_11 ** 2 * tau * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.111111111111111 * lambda_ * lambda_svensson ** 6 * sigma_11 * sigma_31 * tau * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.111111111111111 * lambda_ * lambda_svensson ** 6 * sigma_11 * sigma_31 * tau * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) - 0.0555555555555556 * lambda_ * lambda_svensson ** 6 * sigma_11 * sigma_31 * tau * exp(
            2 * lambda_svensson * tau) + 0.0555555555555556 * lambda_ * lambda_svensson ** 6 * sigma_30 ** 2 * tau * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.111111111111111 * lambda_ * lambda_svensson ** 6 * sigma_30 ** 2 * tau * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) - 0.0833333333333333 * lambda_ * lambda_svensson ** 6 * sigma_30 ** 2 * tau * exp(
            2 * lambda_svensson * tau) + 0.0555555555555556 * lambda_ * lambda_svensson ** 6 * sigma_31 ** 2 * tau * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.111111111111111 * lambda_ * lambda_svensson ** 6 * sigma_31 ** 2 * tau * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) - 0.0833333333333333 * lambda_ * lambda_svensson ** 6 * sigma_31 ** 2 * tau * exp(
            2 * lambda_svensson * tau) + 0.0555555555555556 * lambda_ * lambda_svensson ** 6 * sigma_32 ** 2 * tau * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.111111111111111 * lambda_ * lambda_svensson ** 6 * sigma_32 ** 2 * tau * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) - 0.0833333333333333 * lambda_ * lambda_svensson ** 6 * sigma_32 ** 2 * tau * exp(
            2 * lambda_svensson * tau) + 0.0555555555555556 * lambda_ * lambda_svensson ** 6 * sigma_33 ** 2 * tau * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.111111111111111 * lambda_ * lambda_svensson ** 6 * sigma_33 ** 2 * tau * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) - 0.0833333333333333 * lambda_ * lambda_svensson ** 6 * sigma_33 ** 2 * tau * exp(
            2 * lambda_svensson * tau) - 0.333333333333333 * lambda_ * lambda_svensson ** 5 * sigma_00 * sigma_10 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.333333333333333 * lambda_ * lambda_svensson ** 5 * sigma_00 * sigma_10 * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) - 1.0 * lambda_ * lambda_svensson ** 5 * sigma_00 * sigma_30 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 1.0 * lambda_ * lambda_svensson ** 5 * sigma_00 * sigma_30 * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) - 0.25 * lambda_ * lambda_svensson ** 5 * sigma_10 ** 2 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.333333333333333 * lambda_ * lambda_svensson ** 5 * sigma_10 ** 2 * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) - 0.0833333333333333 * lambda_ * lambda_svensson ** 5 * sigma_10 ** 2 * exp(
            2 * lambda_svensson * tau) - 0.111111111111111 * lambda_ * lambda_svensson ** 5 * sigma_10 * sigma_20 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.111111111111111 * lambda_ * lambda_svensson ** 5 * sigma_10 * sigma_20 * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) - 0.75 * lambda_ * lambda_svensson ** 5 * sigma_10 * sigma_30 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 1.0 * lambda_ * lambda_svensson ** 5 * sigma_10 * sigma_30 * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) - 0.25 * lambda_ * lambda_svensson ** 5 * sigma_10 * sigma_30 * exp(
            2 * lambda_svensson * tau) - 0.111111111111111 * lambda_ * lambda_svensson ** 5 * sigma_10 * sigma_40 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.111111111111111 * lambda_ * lambda_svensson ** 5 * sigma_10 * sigma_40 * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) - 0.25 * lambda_ * lambda_svensson ** 5 * sigma_11 ** 2 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.333333333333333 * lambda_ * lambda_svensson ** 5 * sigma_11 ** 2 * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) - 0.0833333333333333 * lambda_ * lambda_svensson ** 5 * sigma_11 ** 2 * exp(
            2 * lambda_svensson * tau) - 0.111111111111111 * lambda_ * lambda_svensson ** 5 * sigma_11 * sigma_21 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.111111111111111 * lambda_ * lambda_svensson ** 5 * sigma_11 * sigma_21 * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) - 0.75 * lambda_ * lambda_svensson ** 5 * sigma_11 * sigma_31 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 1.0 * lambda_ * lambda_svensson ** 5 * sigma_11 * sigma_31 * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) - 0.25 * lambda_ * lambda_svensson ** 5 * sigma_11 * sigma_31 * exp(
            2 * lambda_svensson * tau) - 0.111111111111111 * lambda_ * lambda_svensson ** 5 * sigma_11 * sigma_41 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.111111111111111 * lambda_ * lambda_svensson ** 5 * sigma_11 * sigma_41 * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) - 0.222222222222222 * lambda_ * lambda_svensson ** 5 * sigma_20 * sigma_30 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.222222222222222 * lambda_ * lambda_svensson ** 5 * sigma_20 * sigma_30 * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) - 0.222222222222222 * lambda_ * lambda_svensson ** 5 * sigma_21 * sigma_31 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.222222222222222 * lambda_ * lambda_svensson ** 5 * sigma_21 * sigma_31 * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) - 0.222222222222222 * lambda_ * lambda_svensson ** 5 * sigma_22 * sigma_32 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.222222222222222 * lambda_ * lambda_svensson ** 5 * sigma_22 * sigma_32 * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) - 0.458333333333333 * lambda_ * lambda_svensson ** 5 * sigma_30 ** 2 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.666666666666667 * lambda_ * lambda_svensson ** 5 * sigma_30 ** 2 * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) - 0.208333333333333 * lambda_ * lambda_svensson ** 5 * sigma_30 ** 2 * exp(
            2 * lambda_svensson * tau) - 0.222222222222222 * lambda_ * lambda_svensson ** 5 * sigma_30 * sigma_40 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.222222222222222 * lambda_ * lambda_svensson ** 5 * sigma_30 * sigma_40 * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) - 0.458333333333333 * lambda_ * lambda_svensson ** 5 * sigma_31 ** 2 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.666666666666667 * lambda_ * lambda_svensson ** 5 * sigma_31 ** 2 * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) - 0.208333333333333 * lambda_ * lambda_svensson ** 5 * sigma_31 ** 2 * exp(
            2 * lambda_svensson * tau) - 0.222222222222222 * lambda_ * lambda_svensson ** 5 * sigma_31 * sigma_41 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.222222222222222 * lambda_ * lambda_svensson ** 5 * sigma_31 * sigma_41 * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) - 0.458333333333333 * lambda_ * lambda_svensson ** 5 * sigma_32 ** 2 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.666666666666667 * lambda_ * lambda_svensson ** 5 * sigma_32 ** 2 * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) - 0.208333333333333 * lambda_ * lambda_svensson ** 5 * sigma_32 ** 2 * exp(
            2 * lambda_svensson * tau) - 0.222222222222222 * lambda_ * lambda_svensson ** 5 * sigma_32 * sigma_42 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.222222222222222 * lambda_ * lambda_svensson ** 5 * sigma_32 * sigma_42 * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) - 0.458333333333333 * lambda_ * lambda_svensson ** 5 * sigma_33 ** 2 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.666666666666667 * lambda_ * lambda_svensson ** 5 * sigma_33 ** 2 * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) - 0.208333333333333 * lambda_ * lambda_svensson ** 5 * sigma_33 ** 2 * exp(
            2 * lambda_svensson * tau) - 0.222222222222222 * lambda_ * lambda_svensson ** 5 * sigma_33 * sigma_43 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.222222222222222 * lambda_ * lambda_svensson ** 5 * sigma_33 * sigma_43 * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) - 0.111111111111111 * lambda_svensson ** 6 * sigma_00 * sigma_10 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.111111111111111 * lambda_svensson ** 6 * sigma_00 * sigma_10 * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) - 0.333333333333333 * lambda_svensson ** 6 * sigma_00 * sigma_30 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.333333333333333 * lambda_svensson ** 6 * sigma_00 * sigma_30 * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) - 0.0833333333333333 * lambda_svensson ** 6 * sigma_10 ** 2 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.111111111111111 * lambda_svensson ** 6 * sigma_10 ** 2 * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) - 0.0277777777777778 * lambda_svensson ** 6 * sigma_10 ** 2 * exp(
            2 * lambda_svensson * tau) - 0.25 * lambda_svensson ** 6 * sigma_10 * sigma_30 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.333333333333333 * lambda_svensson ** 6 * sigma_10 * sigma_30 * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) - 0.0833333333333333 * lambda_svensson ** 6 * sigma_10 * sigma_30 * exp(
            2 * lambda_svensson * tau) - 0.0833333333333333 * lambda_svensson ** 6 * sigma_11 ** 2 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.111111111111111 * lambda_svensson ** 6 * sigma_11 ** 2 * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) - 0.0277777777777778 * lambda_svensson ** 6 * sigma_11 ** 2 * exp(
            2 * lambda_svensson * tau) - 0.25 * lambda_svensson ** 6 * sigma_11 * sigma_31 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.333333333333333 * lambda_svensson ** 6 * sigma_11 * sigma_31 * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) - 0.0833333333333333 * lambda_svensson ** 6 * sigma_11 * sigma_31 * exp(
            2 * lambda_svensson * tau) - 0.152777777777778 * lambda_svensson ** 6 * sigma_30 ** 2 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.222222222222222 * lambda_svensson ** 6 * sigma_30 ** 2 * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) - 0.0694444444444444 * lambda_svensson ** 6 * sigma_30 ** 2 * exp(
            2 * lambda_svensson * tau) - 0.152777777777778 * lambda_svensson ** 6 * sigma_31 ** 2 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.222222222222222 * lambda_svensson ** 6 * sigma_31 ** 2 * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) - 0.0694444444444444 * lambda_svensson ** 6 * sigma_31 ** 2 * exp(
            2 * lambda_svensson * tau) - 0.152777777777778 * lambda_svensson ** 6 * sigma_32 ** 2 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.222222222222222 * lambda_svensson ** 6 * sigma_32 ** 2 * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) - 0.0694444444444444 * lambda_svensson ** 6 * sigma_32 ** 2 * exp(
            2 * lambda_svensson * tau) - 0.152777777777778 * lambda_svensson ** 6 * sigma_33 ** 2 * exp(
            2 * lambda_ * tau) * exp(
            2 * lambda_svensson * tau) + 0.222222222222222 * lambda_svensson ** 6 * sigma_33 ** 2 * exp(
            lambda_ * tau) * exp(
            2 * lambda_svensson * tau) - 0.0694444444444444 * lambda_svensson ** 6 * sigma_33 ** 2 * exp(
            2 * lambda_svensson * tau)) * exp(-2 * lambda_ * tau) * exp(-2 * lambda_svensson * tau) / (
                lambda_ ** 3 * lambda_svensson ** 3 * (lambda_ + lambda_svensson) ** 5)
