import datetime
import logging
import re

import numpy as np
import pandas as pd
from numba import jit

from .buba_data import buba_yields
from .config import MAIN_GRID
from .lw_data import lw_yields

logger = logging.getLogger(__name__)


def price(yield_, ttm_years):
    ttm_full = int(ttm_years)
    ttm_left = ttm_years - ttm_full
    res = 1
    if ttm_full > 0:
        res *= (1 + yield_) ** (-ttm_full)
    res *= np.exp(-yield_ * ttm_left)
    return res


def plot_fit_on_day(synth_yields, fitted_lm, ts, ylim=None):
    actual = synth_yields.loc[ts]
    prediction = pd.Series(fitted_lm.loc[ts].predict(), index=actual.index)
    return actual.rename(f"actual_{ts.to_pydatetime().date()}").to_frame().join(
        prediction.rename(f"prediction_{ts.to_pydatetime().date()}")).plot(ylim=ylim)


def file_name_to_datetime(file_name):
    if hasattr(file_name, "name"):
        file_name = file_name.name
    return datetime.datetime.strptime("".join(file_name.rsplit("-")[-4:]).split(".")[0], "%Y%m%d%H%M%S")


FILE_NAME_PATTERN = re.compile(
    r".*?(?:DataSource(?P<data_source>.*))?Cutoff(?P<cutoff_0>[0-9]*)(?:To(?P<cutoff_1>[0-9]*))?Frequency(?P<frequency>[^-]*).*")


def dissect_file_name(file_name):
    m = re.match(FILE_NAME_PATTERN, file_name)
    if m is None:
        return
    gd = m.groupdict()
    if gd["cutoff_0"]:
        gd["cutoff_0"] = pd.to_datetime(("20" if gd["cutoff_0"][:2] != "70" else "19") + gd["cutoff_0"],
                                        format="%Y%m%d")
    else:
        gd["cutoff_0"] = pd.Timestamp("1970-01-01")
    if gd["cutoff_1"]:
        gd["cutoff_1"] = pd.to_datetime("20" + gd["cutoff_1"], format="%Y%m%d")
    else:
        gd["cutoff_1"] = pd.Timestamp("2038-01-01")

    if gd["data_source"] == "lw":
        gd["yields"] = lw_yields(MAIN_GRID)
        gd["yields"] = gd["yields"][(gd["yields"].index > gd["cutoff_0"])
                                    & (gd["yields"].index < gd["cutoff_1"])]
    else:
        assert gd["data_source"] is None or gd["data_source"] == "buba"
        gd["yields"] = buba_yields(MAIN_GRID, gd["frequency"] == "Wkly", (gd["cutoff_0"], gd["cutoff_1"]))
    return gd


@jit(nopython=True)
def spectral_radius_numba(A):
    return np.max(np.abs(np.linalg.eigvals(A.astype(np.complex128))))


def spectral_radius(A):
    return spectral_radius_numba(numpyify_single(A, dtype=np.complex128))


def numpyify(*args):
    return list(map(numpyify_single, args))


def numpyify_single(arg, dtype=float):
    if hasattr(arg, "to_numpy"):
        return arg.to_numpy().astype(dtype)
    return arg


def is_diagonal(a):
    for i in range(a.shape[0]):
        row = a[i]
        for j in range(a.shape[1]):
            if i != j and row[j] != 0:
                return False
    return True


def is_lower_triangular(a):
    try:
        return np.all(a[np.triu_indices(a.shape[0], 1)] == 0)
    except:
        logger.exception("in is_lower_triangular. maybe not a np.ndarray?")
        return False
