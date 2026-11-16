import pandas as pd

from risk import dataset_risk, score_records, simulate_attack, validate


def _df():
    return pd.DataFrame({"a": [1, 1, 2, 2, 2, 3], "s": ["x", "y", "x", "x", "x", "z"]})


def test_levels_and_homogeneous_escalation():
    r = score_records(_df(), ["a"], ["s"], k_high=2, k_med=3)
    assert r["class_size"].tolist() == [2, 2, 3, 3, 3, 1]
    assert r["homogeneous"].tolist() == [False, False, True, True, True, False]
    assert r["level"].tolist() == [1, 1, 1, 1, 1, 2]


def test_dataset_risk():
    r = score_records(_df(), ["a"], ["s"], k_high=2, k_med=3)
    assert dataset_risk(r)["level"] == "Trung bình"
    low = pd.DataFrame({"risk": [0.01] * 10, "level": [0] * 10})
    assert dataset_risk(low)["level"] == "Thấp"
    assert dataset_risk(low, n_pii_unmasked=1)["level"] == "Cao"


def test_attack_is_monotonic():
    df = pd.DataFrame({"risk": [1.0] * 200 + [0.1] * 200 + [0.01] * 200, "level": [2] * 200 + [1] * 200 + [0] * 200})
    rate = simulate_attack(df, seed=1)
    assert rate["Thấp"] < rate["Trung bình"] < rate["Cao"]


def test_validate_keys():
    df = pd.DataFrame({"a": list(range(20)) * 3, "s": ["x", "y", "z"] * 20})
    s = score_records(df, ["a"], ["s"])
    v = validate(df, ["a"], ["s"], s)
    assert set(v) == {"attack_success", "monotonic", "stability"}