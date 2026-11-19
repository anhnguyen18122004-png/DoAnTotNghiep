import numpy as np
import pandas as pd

from gold import _group_ids
from risk import score_records
from risk_classifier import build_features, run_experiment, split_indices

QI, SA = ["age", "gender", "blood"], ["s"]


def _scored():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "age": rng.integers(20, 50, 1500),
        "gender": rng.choice(["M", "F"], 1500),
        "blood": rng.choice(list("ABCDEFGH"), 1500),
        "s": rng.choice(["x", "y", "z"], 1500),
    })
    return score_records(df, QI, SA)


def test_group_split_has_disjoint_classes():
    sc = _scored()
    tr, te = split_indices(sc, QI, "group", seed=1)
    g = _group_ids(sc, QI).to_numpy()
    assert not set(g[tr]) & set(g[te])


def test_features_exclude_label_columns():
    X = build_features(_scored(), QI, SA)
    assert not {"class_size", "risk", "sa_conf", "homogeneous", "level"} & set(X.columns)
    assert "age__freq" in X.columns


def test_run_experiment():
    res = run_experiment(_scored(), QI, SA, models=["lr", "rf", "baseline"])
    assert {r["model"] for r in res["results"]} == {"lr", "rf", "baseline"}
    for r in res["results"]:
        assert 0.0 <= r["macro_f1"] <= 1.0
    assert res["n_train"] + res["n_test"] == 1500