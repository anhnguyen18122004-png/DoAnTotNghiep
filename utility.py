import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from anonymization import DEFAULT_POLICY, anonymize, generalize_numeric, laplace_perturb
from cleaning import clean_and_standardize, save_silver
from detection import detect, load_ner
from gold import GOLD, k_anonymity
from ingestion import ingest_file
from risk import dataset_risk, score_records
from risk_classifier import make_models

DIRECT = {"PERSON", "ID", "CCCD", "CMND", "PHONE", "EMAIL", "ADDRESS"}


def prepare_features(df, target, drop=(), max_card=50):
    X = df.drop(columns=[target] + [c for c in drop if c in df.columns]).copy()
    dropped = []
    for c in list(X.columns):
        s = X[c]
        if pd.api.types.is_datetime64_any_dtype(s):
            X[c] = (s - s.min()).dt.days.astype(float)
        elif pd.api.types.is_numeric_dtype(s):
            X[c] = pd.to_numeric(s, errors="coerce").astype(float)
        else:
            X[c] = s.astype(object).where(s.notna(), "NA").astype(str)
        if X[c].nunique() <= 1 or (not pd.api.types.is_numeric_dtype(X[c]) and X[c].nunique() > max_card):
            dropped.append(c)
    return X.drop(columns=dropped), dropped


def _prep(X):
    num = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat = [c for c in X.columns if c not in num]
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
    ])


def shared_split(y, test_size=0.25, seed=0):
    return train_test_split(np.arange(len(y)), test_size=test_size, random_state=seed, stratify=y)


def fit_eval(X, y, tr, te, models, seed=0):
    out = {}
    weights = compute_sample_weight("balanced", y[tr])
    for name, model in make_models(seed).items():
        if name not in models:
            continue
        pipe = Pipeline([("prep", _prep(X)), ("clf", model)])
        fit_args = {"clf__sample_weight": weights} if name in ("xgb", "baseline") else {}
        pipe.fit(X.iloc[tr], y[tr], **fit_args)
        proba = pipe.predict_proba(X.iloc[te])
        pred = proba.argmax(axis=1)
        k = proba.shape[1]
        auc = roc_auc_score(y[te], proba[:, 1] if k == 2 else proba, multi_class="ovr") if len(np.unique(y[te])) > 1 else np.nan
        out[name] = {"macro_f1": round(float(f1_score(y[te], pred, average="macro", zero_division=0)), 4),
                     "auc": round(float(auc), 4)}
    return out


def privacy_metrics(df, qi, sa, n_unmasked):
    k = k_anonymity(df, qi)
    d = dataset_risk(score_records(df, qi, sa), n_unmasked)
    return {"avg_risk": k["avg_risk"], "pct_unique": k["pct_unique"], "pct_k_ge_5": k["pct_k_ge_5"], "level": d["level"]}


def build_configs(age_bands, epsilons, bins, dp_cols):
    cfgs = [("original", None, {}), ("default", {}, {})]
    for w in age_bands:
        cfgs.append((f"age_band_{w}", {"AGE": ("generalize_numeric", {"width": w})}, {}))
    if bins:
        cfgs.append(("bins", {}, {"bins": bins}))
    for e in epsilons:
        cfgs.append((f"dp_{e}", {"SALARY": ("dp", {"epsilon": e})}, {"dp": {c: e for c in dp_cols}}))
    return cfgs


def apply_config(silver, findings, policy_over, extra, seed):
    if policy_over is None:
        return silver.copy(), set()
    policy = {**DEFAULT_POLICY, **policy_over}
    anon, applied = anonymize(silver, findings, b"utility", policy=policy, seed=seed)
    for col, w in extra.get("bins", {}).items():
        anon[col] = generalize_numeric(pd.to_numeric(anon[col]), w)
    for col, e in extra.get("dp", {}).items():
        anon[col] = laplace_perturb(pd.to_numeric(anon[col]), epsilon=e, seed=seed)
    return anon, {a["column"] for a in applied}


def run(silver, findings, target, qi, sa, drop=(), age_bands=(5, 10, 20), epsilons=(100.0, 10.0, 1.0),
        bins=None, dp_cols=(), models=("lr", "rf", "lgbm", "baseline"), seed=0, max_card=50):
    if target in {f["column"] for f in findings}:
        raise ValueError("Cột mục tiêu không được là cột PII")
    pii = {f["column"] for f in findings}
    direct = {f["column"] for f in findings if f["type"] in DIRECT}
    classes = sorted(silver[target].dropna().unique())
    y = silver[target].map({c: i for i, c in enumerate(classes)}).fillna(-1).astype(int).to_numpy()
    tr, te = shared_split(y, seed=seed)

    rows = []
    for name, over, extra in build_configs(age_bands, epsilons, bins or {}, dp_cols):
        df, masked = apply_config(silver, findings, over, extra, seed)
        X, dropped = prepare_features(df, target, list(drop) + sorted(direct), max_card)
        util = fit_eval(X, y, tr, te, models, seed)
        rows.append({"config": name, "privacy": privacy_metrics(df, qi, sa, len(pii - masked)),
                     "utility": util, "n_features": X.shape[1], "dropped": dropped})
    base = rows[0]["utility"]
    for r in rows:
        r["delta_auc"] = {m: round(base[m]["auc"] - r["utility"][m]["auc"], 4) for m in r["utility"]}
        r["delta_f1"] = {m: round(base[m]["macro_f1"] - r["utility"][m]["macro_f1"], 4) for m in r["utility"]}
    return {"target": target, "classes": [str(c) for c in classes], "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--qi", nargs="+", required=True)
    ap.add_argument("--sa", nargs="+", required=True)
    ap.add_argument("--drop", nargs="*", default=[])
    ap.add_argument("--age-bands", nargs="*", type=int, default=[5, 10, 20])
    ap.add_argument("--epsilons", nargs="*", type=float, default=[100.0, 10.0, 1.0])
    ap.add_argument("--bins", nargs="*", default=[], help="cot:do_rong, vi du creditscore:50")
    ap.add_argument("--dp-cols", nargs="*", default=[])
    ap.add_argument("--models", nargs="+", default=["lr", "rf", "lgbm", "baseline"])
    ap.add_argument("--no-ner", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    raw, batch_dir = ingest_file(args.file)
    silver, cleaning = clean_and_standardize(raw)
    save_silver(silver, cleaning, batch_dir)
    findings = detect(silver, None if args.no_ner else load_ner())
    missing = [c for c in [args.target] + args.qi + args.sa if c not in silver.columns]
    if missing:
        raise SystemExit(f"Không có cột: {missing}. Cột hiện có: {list(silver.columns)}")

    bins = {b.split(":")[0]: int(b.split(":")[1]) for b in args.bins}
    t0 = time.time()
    res = run(silver, findings, args.target, args.qi, args.sa, args.drop, args.age_bands, args.epsilons,
              bins, args.dp_cols, tuple(args.models), args.seed)
    out = GOLD / Path(args.file).stem
    out.mkdir(parents=True, exist_ok=True)
    (out / "utility_report.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    ms = [m for m in args.models if m != "baseline"]
    print(f"mục tiêu={args.target} lớp={res['classes']} ({time.time() - t0:.0f}s)")
    print(f"{'cấu hình':<14}{'rủi ro TB':>10}{'duy nhất':>10}{'mức':>11}" + "".join(f"{m + ' AUC':>11}{'ΔAUC':>8}" for m in ms))
    for r in res["rows"]:
        p = r["privacy"]
        line = f"{r['config']:<14}{p['avg_risk']:>10.4f}{p['pct_unique']:>10.2%}{p['level']:>11}"
        for m in ms:
            line += f"{r['utility'][m]['auc']:>11.4f}{r['delta_auc'][m]:>8.4f}"
        print(line)
    if "baseline" in args.models:
        print("baseline AUC:", res["rows"][0]["utility"]["baseline"]["auc"])
    print(f"[utility] -> {out / 'utility_report.json'}")


if __name__ == "__main__":
    main()