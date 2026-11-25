import numpy as np
import pandas as pd

from utility_stats import column_utility, group_mean_error, ks_distance, numeric_view


def test_numeric_view_converts_bands():
    s = pd.Series(["30-34", 41, "20-24", None])
    out = numeric_view(s)
    assert out.iloc[0] == 32.5 and out.iloc[1] == 41.0 and out.iloc[2] == 22.5 and np.isnan(out.iloc[3])


def test_ks_distance():
    a = pd.Series(np.arange(100.0))
    assert ks_distance(a, a) == 0.0
    assert ks_distance(a, a + 50) > 0.4


def test_column_utility_identical_is_zero():
    s = pd.Series(np.arange(1, 101, dtype=float))
    u = column_utility(s, s.copy())
    assert u["ks"] == 0.0 and u["mean_rel_err"] == 0.0 and u["record_mae_norm"] == 0.0


def test_group_mean_error():
    df = pd.DataFrame({"v": [10.0, 10.0, 20.0, 20.0], "g": ["a", "a", "b", "b"]})
    noisy = df.assign(v=[11.0, 9.0, 22.0, 18.0])
    assert group_mean_error(df, noisy, "v", "g") == 0.0
    shifted = df.assign(v=[11.0, 11.0, 20.0, 20.0])
    assert group_mean_error(df, shifted, "v", "g") == 0.05