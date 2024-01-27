import pandas as pd
from dateutil import relativedelta
from tqdm.auto import tqdm

from .config import memory, DEFAULT_BUBA_PATH, PKL_BASE_PATH
from .curves import grid_yields_for_apply


def clean_buba_excel_df(date, df):
    df = df.loc[df.iloc[:, 0].str.startswith("DE").fillna(False)].copy()
    columns_to_drop = df.columns[df.isna().sum(0) > 20]  # drop columns with more than 20 N/As
    df = df.drop(columns_to_drop, axis=1)
    columns = ["ISIN 1/3", "ISIN 2/3", "ISIN 3/3", "Description 1/2", "Description 2/2", "Maturity", "TTM",
               "Notional (bn EUR)", "price_clean", "yield", "price_dirty"]
    if len(df.columns) == 12:
        columns.insert(-1, "dirty_yield")
    df.columns = columns
    df["ISIN 3/3"] = df["ISIN 3/3"].astype(int)
    df["Maturity"] = pd.to_datetime(df["Maturity"], dayfirst=True)
    try:
        df = df[~df["yield"].str.startswith("#N/A").fillna(False)]  # see e.g. 2021-07-15
    except AttributeError:
        pass
    df = df[(df["price_clean"] != '0') & (df["yield"] != '-')]  # another data quality issue
    df = df[df["TTM"].str.contains("/").fillna(False)]
    df.loc[~df["TTM"].str.contains(" / "), "TTM"] = df.loc[~df["TTM"].str.contains(" / "), "TTM"].str.replace("/",
                                                                                                              " / ")
    ttm_split = df["TTM"].str.split(" / ", expand=True).astype(float)
    df["TTM"] = ttm_split.iloc[:, 0] + ttm_split.iloc[:, 1] / 12
    df["TTM_precise"] = df["Maturity"] - date
    df["TTM_relativedelta"] = df["Maturity"].apply(lambda m: relativedelta.relativedelta(m, date))
    df["TTM_yf3652425"] = (df["TTM_precise"].dt.days / 365.2425).astype(float)
    df["TTM_yf360"] = (df["TTM_precise"].dt.days / 360).astype(float)
    df["TTM_yf365"] = (df["TTM_precise"].dt.days / 365).astype(float)
    df["inflation_indexed"] = df["Description 2/2"].str.lower().str.contains("index")
    df["green"] = df["Description 2/2"].str.lower().str.contains("green")
    df["ISIN"] = df[[f"ISIN {i}/3" for i in range(1, 3 + 1)]].astype(str).agg("".join, axis=1)
    df.set_index("ISIN", inplace=True, drop=True)
    df.sort_values(["TTM_precise"], inplace=True)
    df = pd.concat({date: df})
    df.index.set_names("date", level=0, inplace=True)
    df[["price_clean", "yield", "price_dirty"]] /= 100
    return df.astype({c: float for c in ("Notional (bn EUR)", "price_clean", "yield", "price_dirty")})


def sheet_name_to_date(sheet_name):
    if sheet_name == "31.05.211":
        sheet_name = "31.05.2011"
    elif sheet_name == "04.062012":
        sheet_name = "04.06.2012"
    elif sheet_name == "08.02.20018":
        sheet_name = "08.02.2018"
    elif sheet_name == "17.05.2103":
        sheet_name = "17.05.2013"
    elif sheet_name == "24.05.2103":
        sheet_name = "24.05.2013"
    elif sheet_name == "13.09.2103":
        sheet_name = "13.09.2013"
    elif sheet_name == "06.02.2104":
        sheet_name = "06.02.2014"
    elif sheet_name == "13.10.2105":
        sheet_name = "13.10.2015"
    elif sheet_name == "30.12.2106":
        sheet_name = "30.12.2016"
    result = pd.to_datetime(sheet_name, dayfirst=True)
    assert pd.Timestamp("2005-01-01") < result < pd.Timestamp("2024-01-01")
    return result


@memory.cache
def get_buba_data_single(excel_path):
    d = pd.read_excel(excel_path, sheet_name=None)
    return {(new_k := sheet_name_to_date(k)): clean_buba_excel_df(new_k, v)
            for k, v in d.items() if k != "Active8_Reference_Sheet"}


def get_buba_data(buba_path=DEFAULT_BUBA_PATH):
    buba_data_dict = {}
    for p in tqdm(sorted(buba_path.glob("*.xlsx"))):
        buba_data_dict.update(get_buba_data_single(p))
    buba_data = pd.concat(buba_data_dict.values())
    buba_data = buba_data[~(buba_data["inflation_indexed"].fillna(False)) & ~(buba_data["green"].fillna(False))]
    return buba_data


def buba_yields(grid, frequency_weekly, cutoff):
    cache_file_name = "cached_synth_yields"
    if frequency_weekly:
        cache_file_name += "Wkly"
    cache_file_name += cutoff[0].strftime("%y%m%d")
    cache_file_name += "To"
    cache_file_name += cutoff[1].strftime("%y%m%d")
    cache_file_name += ".pkl"
    cache_path = PKL_BASE_PATH / cache_file_name
    if not cache_path.exists():
        print("caching to", cache_path)

        buba_data = get_buba_data()
        buba_dates = buba_data.index.get_level_values("date").unique().to_series()
        # print(buba_dates, cutoff)
        buba_dates = buba_dates[(buba_dates > cutoff[0]) & (buba_dates < cutoff[1])]
        buba_dates.min(), buba_dates.max()
        monthly_dates = pd.date_range(buba_dates.min(), buba_dates.max(),
                                      freq=pd.offsets.Week() if frequency_weekly else pd.offsets.MonthBegin())
        # get closest matching available dates
        dates = monthly_dates.to_series().apply(lambda d: buba_dates[(buba_dates >= d)].idxmin()).values

        # if not isinstance(dates, pd.Series):
        dates = pd.Index(dates).to_series()
        result = dates.progress_apply(grid_yields_for_apply, buba_data=buba_data, grid=grid)
        result.to_pickle(cache_path)
        return result
    else:
        print("using cached synth_yields")
        result = pd.read_pickle(cache_path)
        # assert all(result.index == dates)
        return result


if __name__ == '__main__':
    buba_data = get_buba_data()
    print(buba_data)
