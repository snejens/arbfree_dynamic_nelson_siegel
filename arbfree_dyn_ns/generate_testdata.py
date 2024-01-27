import pickle

import numpy as np
import scipy

from .vector_autoregression_process_simulation import simulate
from .nss import B_matrix, C_vector
from .optimization import compute_Q, kalman_filter
from .config import PKL_BASE_PATH


def afgns_data():
    A, B, H, Q, c, theta, sigma, K, lambda_, lambda_svensson = afgns_parameters()
    n_obs = 30 * 12
    return simulate(None, Q, H, B, theta, A, n_obs, c=c)


def afgns_parameters():
    grid = np.array([3, 6, 9, 12, 18, 24, 36, 48, 60, 84, 96, 108, 120, 180, 240, 360]) / 12
    N = len(grid)
    A = np.diag([.9191, .9779, .9687, .8892, .9282])
    Sigma = np.diag([0.01057, .01975, .01773, .05049, .04304109])
    K = np.diag([1.012, .2685, .3812, 1.409, .8940])
    Q = np.diag([8.52e-6, .0000317, .00002533, .000188, .000143])
    theta = np.array([.1165, -.04551, -0.02912, -0.02398, -0.09662])
    time_step_size = 1 / 12
    assert np.allclose(np.diagonal(compute_Q(K, Sigma, time_step_size)) / np.diagonal(Q), 1, atol=.01)
    H = np.eye(N) * 1e-8
    lambda_, lambda_svensson = 1.005, .2343
    B = B_matrix(lambda_, grid, lambda_svensson)
    c = C_vector(lambda_, grid, None, Sigma, lambda_svensson)
    assert np.allclose(np.diagonal(scipy.linalg.expm(-time_step_size * K)) / np.diagonal(A), 1, atol=.01)
    np.allclose(scipy.linalg.expm(-time_step_size * K) - A, 0)

    return A, B, H, Q, c, theta, Sigma, K, lambda_, lambda_svensson


if __name__ == '__main__':
    res = afgns_data()
    xs, ys = res
    with open(PKL_BASE_PATH / "afgns_data.pkl", "wb") as f:
        pickle.dump(res, f)

    A, B, H, Q, c, theta, Sigma, K, lambda_, lambda_svensson = afgns_parameters()
    kalman_res = kalman_filter(ys, theta, A, H, Q, B, c=c)
    x, loglikelihood, loglikelihood_contribution, P, F, v, x_next = kalman_res
    print(loglikelihood)
