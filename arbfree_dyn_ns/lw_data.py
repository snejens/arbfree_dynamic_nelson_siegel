import numpy as np
import pandas as pd

from .config import memory, BASE_PATH


@memory.cache
def lw_yields(grid):
    data = pd.read_excel(BASE_PATH / "data" / "LW_monthly.xlsx")
    columns = data.iloc[7]
    columns.iloc[0] = "date"
    columns.iloc[1:] = columns.iloc[1:].str.extract(" *([0-9]*) m").astype(float).squeeze(1) / 12.
    data.columns = columns
    data = data.iloc[8:].set_index("date")
    data.index = pd.to_datetime(data.index, format="%Y%m")
    return data[grid].dropna() / 100.


if __name__ == '__main__':
    grid = np.array([1, 5, 10, 30])
    yields = lw_yields(grid)
    print(yields)
