import numpy as np
import pandas as pd
import scipy
import sympy
from IPython.core.display_functions import clear_output, display

from .packing import unpack, pack_tightly, SPECTRAL_RADIUS_CONSTANT
from .utils import numpyify, spectral_radius_numba, is_diagonal
from .nss import B_matrix, C_vector, T
from .kalman import kalman_filter


def compute_Q(K, Sigma, time_step_size, slow=False):
    if slow:
        exp_minusTK = sympy.exp(-T * sympy.Matrix(K))
        Q = sympy.integrate(exp_minusTK @ Sigma @ Sigma.T @ exp_minusTK.T,
                            (T, 0, time_step_size))
        Q = np.array(Q).astype(float)
        return Q
    else:
        return _compute_Q_fast(*numpyify(K, Sigma), time_step_size)


# @jit(nopython=True)
def _compute_Q_fast(K, Sigma, time_step_size):
    if is_diagonal(Sigma) and is_diagonal(K):
        return _compute_Q_fast_all_diagonal(K, Sigma, time_step_size)
    Sigma_squared = Sigma @ Sigma.T

    def integrand(t):
        exp_minustK = scipy.linalg.expm(-t * K)
        return exp_minustK @ Sigma_squared @ exp_minustK.T

    return scipy.integrate.quad_vec(integrand, 0, time_step_size)[0]


def _compute_Q_fast_all_diagonal(K, Sigma, time_step_size):
    return np.diag(
        [-Sigma[i][i] ** 2 / (2 * K[i][i]) * (np.exp(-2 * time_step_size * K[i][i]) - 1) for i in range(K.shape[0])])


def ensure_unit_spectrum(K_x, time_step_size):
    A_x_unscaled = scipy.linalg.expm(-time_step_size * K_x)
    scaling_factor = 1 + spectral_radius_numba(A_x_unscaled) + SPECTRAL_RADIUS_CONSTANT
    A_x = A_x_unscaled / scaling_factor
    K_x += np.log(scaling_factor) * np.eye(K_x.shape[0]) / time_step_size
    return A_x, K_x


def real_K_initial(K_initial, time_step_size):
    """unscale so that scaled is K_initial"""

    def objective_helper(x):
        return K_initial + x * np.eye(K_initial.shape[0]) / time_step_size

    def objective(x):
        A, K = ensure_unit_spectrum(objective_helper(x), time_step_size)
        return np.linalg.norm(K - K_initial)

    x = scipy.optimize.minimize(objective, x0=0).x
    return objective_helper(x)


def kalman_optimize(synth_yields_for_optimization, theta_initial, K_initial, H_initial, Sigma_initial,
                    lambda_initial=1,
                    lambda_svensson_initial=.25,
                    diag_restrictions=(2,), spd_restrictions=(2,),
                    lower_triangle_restrictions=(3,),
                    unit_spectrum_restrictions=tuple(), svensson_active=None, yield_adjustment_term_active=True,
                    out=None, time_step_size=1):
    grid = synth_yields_for_optimization.columns
    print(f"grid = {grid}")

    K_initial = real_K_initial(K_initial, time_step_size)
    objs = numpyify(theta_initial, K_initial, H_initial, Sigma_initial)
    n = K_initial.shape[0]

    shapes = [obj.shape for obj in objs]
    diag = diag_restrictions
    spd = spd_restrictions
    lower_triangle = lower_triangle_restrictions
    unit_spectrum = unit_spectrum_restrictions
    scale = {}  # {1: two_step_K.to_numpy()}

    if svensson_active is None:
        svensson_active = lambda_svensson_initial is not None

    theta_xs, K_xs, H_xs, Sigma_xs, loglikelihood_xs, lambda_xs, lambda_svensson_xs, n_ev = [], [], [], [], [], [], [], 0

    def objective(x, verbose=True):  # , no_tqdm=True):
        nonlocal theta_xs, K_xs, H_xs, Sigma_xs, loglikelihood_xs, lambda_xs, lambda_svensson_xs, n_ev
        n_ev += 1

        H_x, K_x, Sigma_x, lambda_, lambda_svensson, theta_x, A_x = unpack_all(x)

        if verbose and n_ev % 500 == 1:
            print()
        if verbose and n_ev % 1000 == 1:
            print(f"""***
        theta={theta_x}
        eigvals(K)={np.linalg.eigvals(K_x)}
        eigvals(H)={np.linalg.eigvalsh(H_x)}
        eigvals(Sigma)={np.linalg.eigvals(Sigma_x)}
        lambda={lambda_}
        lambda_svensson={lambda_svensson}
    """)
            if out is not None:
                with out[1]:
                    clear_output()
                    display(pd.Series(theta_x))
                with out[2]:
                    clear_output()
                    display(pd.DataFrame(K_x))
                with out[3]:
                    clear_output()
                    display(pd.DataFrame(H_x,
                                         index=synth_yields_for_optimization.columns,
                                         columns=synth_yields_for_optimization.columns))
                with out[4]:
                    clear_output()
                    display(pd.DataFrame(Sigma_x))
                with out[5]:
                    clear_output()
                    display(lambda_)
                with out[6]:
                    clear_output()
                    display(lambda_svensson)

            theta_xs.append(theta_x)
            K_xs.append(K_x)
            H_xs.append(H_x)
            Sigma_xs.append(Sigma_x)
            lambda_xs.append(lambda_)
            if svensson_active:
                lambda_svensson_xs.append(lambda_svensson)

        B = B_matrix(lambda_, grid, lambda_svensson_value=lambda_svensson)
        B_for_optimization = B.to_numpy()
        if yield_adjustment_term_active:
            c_vector = C_vector(lambda_, grid, K_times_theta=None, Sigma=Sigma_x, lambda_svensson_value=lambda_svensson)
            c_vector = c_vector.to_numpy()
        else:
            c_vector = None

        # print(B_for_optimization)
        # print(c_vector)

        # time_before = time.perf_counter()
        Q_x = compute_Q(K_x, Sigma_x, time_step_size=time_step_size, slow=False)
        # time_after = time.perf_counter()
        # print("Execution took", time_after - time_before, "seconds")

        # print(theta_x, A_x, H_x, Q_x, B_for_optimization, c_vector)
        res = kalman_filter(synth_yields_for_optimization, theta_x, A_x, H_x, Q_x, B=B_for_optimization,
                            exclude_first_observations_for_loglikelihood=min(10,
                                                                             .05 * len(synth_yields_for_optimization)),
                            c=c_vector)[1]  # , no_tqdm=no_tqdm)[1]

        loglikelihood_xs.append(res)

        if verbose:
            if n_ev % 500 == 1:
                print("loglikelihood =", res)
                if out is not None:
                    with out[0]:
                        clear_output()
                        display(res)
            # else:
            # print(".", end="")

        return -res

    def unpack_all(x):
        if svensson_active:
            x, lambda_, lambda_svensson = x[:-2], np.exp(x[-2]), np.exp(x[-1] + x[-2])
        else:
            x, lambda_, lambda_svensson = x[:-1], np.exp(x[-1]), None
        theta_x, K_x, H_x, Sigma_x = unpack(x, shapes, diag=diag, spd=spd, unit_spectrum=unit_spectrum, scale=scale,
                                            lower_triangle=lower_triangle)

        A_x, K_x = ensure_unit_spectrum(K_x, time_step_size)

        assert np.allclose(scipy.linalg.expm(-time_step_size * K_x), A_x)

        return H_x, K_x, Sigma_x, lambda_, lambda_svensson, theta_x, A_x

    if scale:
        x0_ = np.concatenate([objs[0], np.array([1]),
                              pack_tightly(objs[2:],
                                           diag=[-2 + x for x in diag],
                                           spd=[-2 + x for x in spd],
                                           lower_triangle=[-2 + x for x in lower_triangle],
                                           unit_spectrum=[-2 + x for x in unit_spectrum])])
    else:
        x0_ = pack_tightly(objs,
                           diag=diag,
                           spd=spd,
                           lower_triangle=lower_triangle,
                           unit_spectrum=unit_spectrum)

    if svensson_active:
        assert lambda_initial > lambda_svensson_initial
        x0_ = np.append(x0_,
                        np.array([np.log(lambda_initial), np.log(lambda_svensson_initial) - np.log(lambda_initial)]))
    else:
        x0_ = np.append(x0_, np.log(lambda_initial))

    n_theta = n
    n_K = n if 1 in diag else n ** 2
    n_H = len(grid) if 2 in diag else len(grid) * (len(grid) + 1) // 2
    n_Sigma = n if 3 in diag else n * (n + 1) // 2 if 3 in lower_triangle or 3 in spd else n ** 2
    inf = np.inf
    bounds = [(-inf, inf) for i in range(n_theta)] + \
             [(-inf, inf) for i in range(n_K)] + \
             [(np.log(1e-10), 0) if 2 in diag and 2 in spd else (-inf, inf) for i in range(n_H)] + \
             [(-inf, inf)
              # (np.log(1e-15), np.log(.1)) if 3 in diag and 3 in spd else (1e-15, .1) if 3 in diag else (-.2, .2)
              for i in range(n_Sigma)] + \
             [(np.log(.02), np.log(2))] + \
             ([(np.log(.005), np.log(1))] if svensson_active else [])

    assert len(bounds) == len(x0_)

    # ------------------------------------ START TEST INTERCHANGING lambda1 / lambda2
    if False and n == 5:
        x0_lambda1lambda2interchanged = x0_[[0, 2, 1, 4, 3,  # theta
                                             0 + 5, 2 + 5, 1 + 5, 4 + 5, 3 + 5, ] +  # K
                                            list(range(10, n_H + 10)) +  # H
                                            [0 + n_H + 10, 2 + n_H + 10, 1 + n_H + 10, 4 + n_H + 10, 3 + n_H + 10,
                                             # Sigma
                                             -1, -2  # lambda_2/1
                                             ]].copy()

        lambda1 = np.exp(x0_[-2])
        lambda2 = np.exp(x0_[-2] + x0_[-1])

        x0_lambda1lambda2interchanged[-2] = np.log(lambda2)
        x0_lambda1lambda2interchanged[-1] = np.log(lambda1) - np.log(lambda2)

        print("x0 =", x0_)
        print("x0_interchanged =", x0_lambda1lambda2interchanged)
        obj0 = objective(x0_)
        n_ev = 0
        obj0i = objective(x0_lambda1lambda2interchanged)
        print("obj =", obj0, obj0i)
        assert len(x0_) == len(x0_lambda1lambda2interchanged)
        assert obj0 == obj0i
    # ------------------------------------------ END TEST INTERCHANGING lambda1 / lambda2

    optres = scipy.optimize.minimize(objective,
                                     x0=x0_,
                                     # options={"gtol": 1e-7 * len(synth_yields_for_optimization)},
                                     options={"maxfun": 99999999, "maxiter": 9999999,
                                              "ftol": 1e4 * np.finfo(float).eps},
                                     method="L-BFGS-B", bounds=bounds)
    if out is not None:
        with out[0]:
            clear_output()
            display(optres)

    H_opt, K_opt, Sigma_opt, lambda_opt, lambda_svensson_opt, theta_opt, A_opt = unpack_all(optres.x)

    return optres, H_opt, K_opt, Sigma_opt, lambda_opt, lambda_svensson_opt, theta_opt, \
        theta_xs, K_xs, H_xs, Sigma_xs, loglikelihood_xs, lambda_xs, lambda_svensson_xs, n_ev
