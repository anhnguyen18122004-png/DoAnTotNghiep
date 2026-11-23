import pandas as pd

from detection_eval import evaluate, leakage, prf, run


def test_prf():
    r = prf(8, 2, 2)
    assert r["precision"] == 0.8 and r["recall"] == 0.8 and r["f1"] == 0.8
    assert prf(0, 0, 0)["f1"] == 0.0


def test_evaluate_counts_errors():
    found = [{"column": "a", "type": "PHONE"}, {"column": "b", "type": "EMAIL"}]
    res = evaluate(found, [("a", "PHONE"), ("c", "PERSON")])
    assert res["micro"]["tp"] == 1 and res["micro"]["fp"] == 1 and res["micro"]["fn"] == 1
    assert res["false_positives"] == [("b", "EMAIL")] and res["false_negatives"] == [("c", "PERSON")]
    assert res["per_type"]["PHONE"]["f1"] == 1.0 and res["per_type"]["EMAIL"]["precision"] == 0.0


def test_leakage_rates():
    silver = pd.DataFrame({"phone": ["0912345678"] * 10, "name": ["An", "Binh"] * 5})
    anon = silver.copy()
    anon["phone"] = "*******678"
    out = leakage(silver, anon, [("phone", "PHONE"), ("name", "PERSON")])
    assert out["phone:PHONE"] == 0.0 and out["name:PERSON"] == 1.0


def test_run_blind_hides_names_but_keeps_regex_types():
    silver = pd.DataFrame({
        "sdt": ["0912345678", "0987654321"] * 10,
        "email": ["a@x.com", "b@y.org"] * 10,
        "other": ["x", "y"] * 10,
    })
    truth = [("sdt", "PHONE"), ("email", "EMAIL")]
    res = run(silver, truth, ner=None, blind=True)
    assert res["micro"]["f1"] == 1.0