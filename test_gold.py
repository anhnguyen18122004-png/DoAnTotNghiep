import pandas as pd

from gold import dataset_profile, k_anonymity, l_diversity, record_risk


def _df():
    return pd.DataFrame({"a": [1, 1, 2, 2, 2, 3], "s": ["x", "y", "x", "x", "x", "z"]})


def test_k_anonymity():
    k = k_anonymity(_df(), ["a"])
    assert k["k_min"] == 1 and k["n_classes"] == 3
    assert k["pct_unique"] == round(1 / 6, 4)
    assert k["avg_risk"] == 0.5 and k["max_risk"] == 1.0


def test_record_risk():
    r = record_risk(_df(), ["a"])
    assert r["class_size"].tolist() == [2, 2, 3, 3, 3, 1]
    assert r["risk"].iloc[5] == 1.0


def test_l_diversity():
    l = l_diversity(_df(), ["a"], "s")
    assert l["l_min"] == 1 and l["pct_homogeneous"] == round(4 / 6, 4)


def test_dataset_profile():
    df = pd.DataFrame({"a": [1, None], "b": ["x", "y"]})
    f = [{"column": "b", "type": "PERSON"}]
    p = dataset_profile(df, f, {"rows_in": 4, "duplicates_removed": 2}, masked={"b"})
    assert p["n_rows"] == 2 and p["duplicate_rate"] == 0.5
    assert p["n_pii_cols"] == 1 and p["n_pii_unmasked"] == 0