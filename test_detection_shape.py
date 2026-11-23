import numpy as np
import pandas as pd

from detection import detect


def _df(n=300):
    rng = np.random.default_rng(0)
    first = ["An", "Binh", "Chi", "Dung", "Hoa", "Khoa", "Lan", "Minh", "Nam", "Phuc"]
    last = ["Nguyen", "Tran", "Le", "Pham", "Hoang", "Vo", "Dang", "Bui"]
    return pd.DataFrame({
        "v1": rng.choice(np.arange(15_000_000, 16_000_000), n, replace=False),
        "v2": np.arange(1, n + 1),
        "v3": [f"{rng.choice(last)} {rng.choice(first)}{i}".replace(str(i), "") + "x" * (i % 7) for i in range(n)],
        "v4": rng.integers(18, 85, n),
        "v5": pd.to_datetime("1960-01-01") + pd.to_timedelta(rng.integers(0, 365 * 40, n), unit="D"),
        "v6": pd.to_datetime("2020-01-01") + pd.to_timedelta(rng.integers(0, 365 * 3, n), unit="D"),
        "v7": [f"{rng.integers(1, 300)} Le Loi, Quan {rng.integers(1, 12)}, Ho Chi Minh" for _ in range(n)],
        "v8": [f"{rng.choice(last)} {rng.choice(['Group', 'Inc', 'LLC', 'and Sons'])}" for _ in range(n)],
        "v9": rng.choice(["Male", "Female"], n),
        "v10": rng.normal(1000, 200, n).round(2),
    })


def test_shape_detection():
    got = {f["column"]: f["type"] for f in detect(_df())}
    assert got["v1"] == "ID" and got["v4"] == "AGE" and got["v5"] == "DOB" and got["v7"] == "ADDRESS"


def test_shape_avoids_false_positives():
    got = {f["column"] for f in detect(_df())}
    assert not {"v2", "v6", "v8", "v9", "v10"} & got


def test_small_tables_are_left_alone():
    assert detect(_df(20)) == []