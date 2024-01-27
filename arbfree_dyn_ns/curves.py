import pandas as pd
import scipy


def fwd_curve(zero_yield_observations: pd.Series, cmp_cont=False) -> pd.Series:
    if cmp_cont:
        def calc(s):
            return (s.iloc[1] * s.index[1] - s.iloc[0] * s.index[0]) / (s.index[1] - s.index[0])
    else:  # yearly compounding
        def calc(s):
            return ((1 + s.iloc[1]) ** s.index[1] / (1 + s.iloc[0]) ** s.index[0]) ** (
                    1 / (s.index[1] - s.index[0])) - 1
    return zero_yield_observations.rolling(2, min_periods=1).apply(
        lambda s: s.iloc[0] if len(s) == 1 else calc(s)
    )


def zero_yield_curve(forward_rate_observations: pd.Series, cmp_cont=False) -> pd.Series:
    if cmp_cont:
        def calc(t, last_t, f, last_yield):
            return 1 / t * (last_yield * last_t + f * (t - last_t))
    else:
        def calc(t, last_t, f, last_yield):
            return ((1 + last_yield) ** last_t * (1 + f) ** (t - last_t)) ** (1 / t) - 1

    zero_yields, last_t = ([forward_rate_observations.iloc[0]],
                           forward_rate_observations.index[0])
    for t, f in forward_rate_observations.iloc[1:].items():
        zero_yields.append(calc(t, last_t, f, zero_yields[-1]))
        last_t = t
    return pd.Series(zero_yields, index=forward_rate_observations.index)


def piecewise_constant_interp1d(series: pd.Series):
    return scipy.interpolate.interp1d(series.index, series.values,
                                      kind="previous",
                                      fill_value="extrapolate")


def linear_interp1d(series: pd.Series):
    return scipy.interpolate.interp1d(series.index, series.values,
                                      kind="linear",
                                      fill_value="extrapolate")


def fama_bliss_unsmoothed_yields(raw_yield_curve, target_grid):
    ip = piecewise_constant_interp1d(fwd_curve(raw_yield_curve))
    return zero_yield_curve(pd.Series(ip(target_grid), index=target_grid))


def simple_unsmoothed_yields(raw_yield_curve, target_grid):
    ip = linear_interp1d(raw_yield_curve)
    return pd.Series(ip(target_grid), index=target_grid)


def add_rolling_means(raw_curve, n=3):
    rolling_mean = raw_curve.rolling(n).mean()

    rolling_mean_3before = rolling_mean.shift(1).bfill().rename("mean_3before")
    rolling_mean_3after = rolling_mean.shift(-n).ffill().rename("mean_3after")

    tmp = raw_curve.rename("raw").to_frame().join(rolling_mean_3before).join(rolling_mean_3after)
    tmp = tmp.join(
        tmp[["mean_3before", "mean_3after"]].apply(lambda r: pd.Series(sorted(r), index=["lower", "upper"]), axis=1))
    tmp["include"] = (tmp["raw"] > tmp["lower"]) & (tmp["raw"] < tmp["upper"])
    return tmp


def grid_yields(ts_or_df, buba_data, grid):
    if not isinstance(ts_or_df, (pd.DataFrame, pd.Series)):
        return grid_yields(
            buba_data.loc[ts_or_df][["TTM_yf3652425", "yield"]].set_index("TTM_yf3652425", drop=True).squeeze(1),
            buba_data=buba_data, grid=grid)
    raw_curve = ts_or_df.copy()
    # rm = add_rolling_means(raw_curve)
    # raw_curve = raw_curve[rm["include"]]
    # rm = add_rolling_means(fwd_curve(raw_curve), n=5)
    # display(rm)
    # raw_curve = raw_curve[rm["include"]]
    ts_or_df = ts_or_df[~ts_or_df.index.duplicated()]  # .drop_duplicates()
    rm = add_rolling_means(ts_or_df, n=3)  # see it in action on 2020-08-25, 2021-11-18
    ts_or_df = ts_or_df[rm["include"] & (rm.index > .5)]
    # raw_curve = raw_curve[raw_curve.index > .5]

    return simple_unsmoothed_yields(ts_or_df, grid), raw_curve


def grid_yields_for_apply(ts_or_df, buba_data, grid):
    try:
        return grid_yields(ts_or_df, buba_data, grid)[0]
    except:
        pass
