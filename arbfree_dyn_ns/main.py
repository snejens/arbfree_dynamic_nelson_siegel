import argparse
import itertools
import logging
import pickle

import numpy as np
import pandas as pd
import scipy

from .generate_testdata import afgns_data, afgns_parameters
from .lw_data import lw_yields
from .optimization import kalman_optimize
from .config import PKL_BASE_PATH, MAIN_GRID
from .nss import B_matrix, C_vector
from .buba_data import buba_yields

logger = logging.getLogger(__name__)

DEFAULT_CUTOFFS = {"buba": (pd.Timestamp("2019-08-21"), pd.Timestamp("2038-01-01")),
                   # 2019-08-21 first time with long-term (30y) ZCB in Buba data
                   "lw": (pd.Timestamp("1970-01-01"), pd.Timestamp("2038-01-01")),
                   }


def main(n, yield_adjustment_term_active, use_afgns_test_data=False,
         cutoff=None,
         frequency_weekly=True, data_source="buba", alternative_initial_solution=0):  # else monthly
    assert n in (3, 5)
    if cutoff is None:
        cutoff = DEFAULT_CUTOFFS[data_source]
    if not cutoff[0] or pd.isna(cutoff[0]):
        cutoff = list(cutoff)
        cutoff[0] = DEFAULT_CUTOFFS[data_source][0]
    if not cutoff[1] or pd.isna(cutoff[1]):
        cutoff = list(cutoff)
        cutoff[1] = DEFAULT_CUTOFFS[data_source][1]
    if data_source == "lw":
        assert not frequency_weekly, "lw => frequency_weekly must be False"

    if use_afgns_test_data:
        desc = 'TEST'
        xs, ys = afgns_data()
        synth_yields = ys
        time_step_size = 1 / 12
    else:
        print(cutoff)
        desc = f'DataSource{data_source}Cutoff{cutoff[0].strftime("%y%m%d")}To{cutoff[1].strftime("%y%m%d")}Frequency{"Wkly" if frequency_weekly else "Mthly"}'
        synth_yields = synth_yields_for_main(frequency_weekly, cutoff, data_source)
        time_step_size = 7 / 365.2425 if frequency_weekly else 1 / 12
        # synth_yields = synth_yields[(synth_yields.index < cutoff[0]) & (synth_yields.index < cutoff[1])]

    N = synth_yields.shape[1]
    theta_initial, K_initial, H_initial, Sigma_initial, lambda_initial, lambda_svensson_initial = \
        initial_solution(N, n, synth_yields.columns, yield_adjustment_term_active,
                         alternative=alternative_initial_solution)

    print("------------------------------------------------------------------------------")
    print(desc, n, yield_adjustment_term_active)
    print("------------------------------------------------------------------------------")
    kalman_optimize_output = kalman_optimize(synth_yields, theta_initial, K_initial, H_initial, Sigma_initial,
                                             lambda_initial=lambda_initial, diag_restrictions=(1, 2, 3),
                                             spd_restrictions=(2, 3),
                                             # 3 to force Sigma entries positive (only for case 3 in diag_restrictions!)
                                             lambda_svensson_initial=lambda_svensson_initial,
                                             svensson_active=n == 5,
                                             yield_adjustment_term_active=yield_adjustment_term_active,
                                             out=None, time_step_size=time_step_size)

    optres, H_opt, K_opt, Sigma_opt, lambda_opt, lambda_svensson_opt, theta_opt, \
        theta_xs, K_xs, H_xs, Sigma_xs, loglikelihood_xs, lambda_xs, lambda_svensson_xs, n_ev = kalman_optimize_output

    ts = pd.Timestamp('now').strftime('%Y-%m-%d-%H%M%S')
    with open(PKL_BASE_PATH / f"main{desc}-{n}-{yield_adjustment_term_active}-{ts}.pkl", "wb") as f:
        pickle.dump(kalman_optimize_output, f)


def synth_yields_for_main(frequency_weekly, cutoff, data_source):
    grid = MAIN_GRID
    if data_source == "buba":
        synth_yields = buba_yields(grid, frequency_weekly, cutoff)
        return synth_yields

    else:
        assert data_source == "lw" and not frequency_weekly
        yields = lw_yields(grid)
        return yields[(yields.index > cutoff[0]) & (yields.index < cutoff[1])]


def compute_theta_initial(Sigma, grid, lambda_, lambda_svensson):
    B = B_matrix(lambda_, grid, lambda_svensson)
    c = C_vector(lambda_, grid, None, Sigma, lambda_svensson)
    return scipy.optimize.minimize(lambda x: np.linalg.norm(B @ x + c), x0=np.zeros(B.shape[1])).x


def initial_solution(N, n, grid, yield_adjustment_term_active, alternative=0):
    if alternative == 42:
        A_initial, B_initial, H_initial, Q_initial, c_initial, theta_initial, Sigma_initial, K_initial, lambda_initial, lambda_svensson_initial = afgns_parameters()
    elif alternative == 0:
        lambda_initial = 1.
        lambda_svensson_initial = .25 if n == 5 else None
        K_initial = np.eye(n)
        H_initial = np.eye(N) / 100
        Sigma_initial = np.eye(n) * .015

        theta_initial = compute_theta_initial(Sigma_initial, grid, lambda_initial,
                                              lambda_svensson_initial) if yield_adjustment_term_active else np.zeros(n)
    elif alternative == 1:
        lambda_initial = .3
        lambda_svensson_initial = .03 if n == 5 else None
        K_initial = np.eye(n)
        H_initial = np.eye(N) / 100
        Sigma_initial = np.eye(n) * .015

        theta_initial = compute_theta_initial(Sigma_initial, grid, lambda_initial,
                                              lambda_svensson_initial) if yield_adjustment_term_active else np.zeros(n)
    elif alternative == 2:
        lambda_initial = .03
        lambda_svensson_initial = .01 if n == 5 else None
        K_initial = np.eye(n)
        H_initial = np.eye(N) / 100
        Sigma_initial = np.eye(n) * .015

        theta_initial = compute_theta_initial(Sigma_initial, grid, lambda_initial,
                                              lambda_svensson_initial) if yield_adjustment_term_active else np.zeros(n)
    else:
        raise ValueError("value for 'alternative' unknown")

    return theta_initial, K_initial, H_initial, Sigma_initial, lambda_initial, lambda_svensson_initial


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arbfree_dyn_ns Main')
    parser.add_argument('--test', action="store_true", default=False,
                        help='whether to run test case instead of actual run')
    parser.add_argument('--frequency_weekly', action="store_true", default=False,
                        help='whether to use weekly observations instead of monthly')
    parser.add_argument('--cutoff_min', type=str, default="",
                        help='cutoff date (min, exclusive)')
    parser.add_argument('--cutoff_max', type=str, default="",
                        help='cutoff date (max, exclusive)')
    parser.add_argument('--data_source', type=str, default="lw",
                        help='data_source')
    parser.add_argument('--n', nargs='+', type=int, default=(3, 5),
                        help='n\'s to run for (out of 3, 5)')
    parser.add_argument('--yield_adjustment_term', nargs='+', type=bool, default=(True, False),
                        help='include/exclude yield adjustment term')

    # Parse the command-line arguments
    args = parser.parse_args()

    if args.test:
        main(5, True, True)
    else:
        for n, yield_adjustment_term_active in itertools.product(args.n, args.yield_adjustment_term):
            # if n == 3 and yield_adjustment_term_active:
            #    continue
            for alternative_initial_solution in range(3):
                try:
                    main(n, yield_adjustment_term_active,
                         cutoff=tuple(pd.Timestamp(cd) for cd in (args.cutoff_min, args.cutoff_max)),
                         data_source=args.data_source,
                         frequency_weekly=args.frequency_weekly,
                         alternative_initial_solution=alternative_initial_solution)
                    break
                except:
                    logger.exception("with alternative_initial_solution=%s", alternative_initial_solution)
