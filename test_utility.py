import numpy as np
import pandas as pd

from utility import fit_eval, prepare_features, privacy_metrics, shared_split


def _data(n=1500):
    rng = np.random.default_rng(0)
    x = rng.normal(size=n)
    df = pd.DataFrame({
        "x": x,
        "noise": rng.normal(size=n),
        "grp": rng.choice(["a", "b", "c"], n),
        "id_like": [f"id{i}" for i in range(n)],
        "const": 1,
        "y": (x + rng.normal(scale=0.5, size=n) > 0).astype(int),
    })
    return df


def test_prepare_features_drops_constant_and_high_cardinality():
    X, dropped = prepare_features(_data(), "y", max_card=50)
    assert "y" not in X.columns
    assert set(dropped) == {"id_like", "const"}


def test_shared_split_is_deterministic_and_disjoint():
    y = _data()["y"].to_numpy()
    tr1, te1 = shared_split(y, seed=3)
    tr2, te2 = shared_split(y, seed=3)
    assert (tr1 == tr2).all() and not set(tr1) & set(te1)


def test_fit_eval_signal_beats_baseline():
    df = _data()
    X, _ = prepare_features(df, "y")
    y = df["y"].to_numpy()
    tr, te = shared_split(y)
    res = fit_eval(X, y, tr, te, ["lr", "baseline"])
    assert res["lr"]["auc"] > 0.8 and res["baseline"]["auc"] == 0.5


def test_privacy_metrics_keys():
    df = _data()
    p = privacy_metrics(df, ["grp"], ["y"], 0)
    assert {"avg_risk", "pct_unique", "pct_k_ge_5", "level"} == set(p)