import argparse
import json
import time

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from gold import GOLD, _group_ids

LEVELS = ["Thấp", "Trung bình", "Cao"]


def build_features(scored, qi, sa):
    cols = list(qi) + list(sa)
    X = scored[cols].copy()
    for c in cols:
        X[f"{c}__freq"] = X[c].map(X[c].value_counts(normalize=True)).astype(float)
    return X


def split_indices(scored, qi, split="group", test_size=0.25, seed=0):
    idx = np.arange(len(scored))
    if split == "random":
        return train_test_split(idx, test_size=test_size, random_state=seed, stratify=scored["level"])
    groups = _group_ids(scored, qi).to_numpy()
    return next(GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed).split(idx, groups=groups))


def make_models(seed=0):
    return {
        "lr": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "rf": RandomForestClassifier(n_estimators=200, min_samples_leaf=2, class_weight="balanced", n_jobs=-1, random_state=seed),
        "xgb": XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
                             eval_metric="mlogloss", n_jobs=-1, random_state=seed),
        "lgbm": LGBMClassifier(n_estimators=300, learning_rate=0.1, num_leaves=31, class_weight="balanced",
                               n_jobs=-1, random_state=seed, verbose=-1),
        "baseline": DummyClassifier(strategy="most_frequent"),
    }


def _preprocessor(X):
    num = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat = [c for c in X.columns if c not in num]
    return ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
    ])


def run_experiment(scored, qi, sa, seed=0, split="group", models=None):
    X = build_features(scored, qi, sa)
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].astype(str)
    classes = sorted(scored["level"].unique())
    y = scored["level"].map({c: i for i, c in enumerate(classes)}).to_numpy()
    tr, te = split_indices(scored, qi, split, seed=seed)
    weights = compute_sample_weight("balanced", y[tr])

    out = []
    for name, model in make_models(seed).items():
        if models and name not in models:
            continue
        pipe = Pipeline([("prep", _preprocessor(X)), ("clf", model)])
        t0 = time.time()
        fit_args = {"clf__sample_weight": weights} if name in ("xgb", "baseline") else {}
        pipe.fit(X.iloc[tr], y[tr], **fit_args)
        secs = time.time() - t0
        pred = pipe.predict(X.iloc[te])
        try:
            auc = float(roc_auc_score(y[te], pipe.predict_proba(X.iloc[te]), multi_class="ovr", average="macro"))
        except ValueError:
            auc = None
        out.append({
            "model": name,
            "accuracy": round(float(accuracy_score(y[te], pred)), 4),
            "macro_f1": round(float(f1_score(y[te], pred, average="macro", zero_division=0)), 4),
            "weighted_f1": round(float(f1_score(y[te], pred, average="weighted", zero_division=0)), 4),
            "roc_auc": None if auc is None else round(auc, 4),
            "fit_seconds": round(secs, 1),
            "confusion": confusion_matrix(y[te], pred, labels=range(len(classes))).tolist(),
        })
    return {
        "classes": [LEVELS[int(c)] for c in classes],
        "n_train": int(len(tr)),
        "n_test": int(len(te)),
        "test_distribution": {LEVELS[int(c)]: round(float((y[te] == i).mean()), 4) for i, c in enumerate(classes)},
        "results": out,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--qi", nargs="+", required=True)
    ap.add_argument("--sa", nargs="+", required=True)
    ap.add_argument("--split", choices=["group", "random"], default="group")
    ap.add_argument("--models", nargs="+")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    folder = GOLD / args.name
    report = {}
    for tag in ("before", "after"):
        scored = pd.read_parquet(folder / f"risk_{tag}.parquet")
        res = run_experiment(scored, args.qi, args.sa, args.seed, args.split, args.models)
        report[tag] = res
        print(f"[{tag}] chia {args.split}: train={res['n_train']} test={res['n_test']} phân bố test={res['test_distribution']}")
        print(f"  {'mô hình':<9}{'acc':>8}{'macroF1':>9}{'wF1':>8}{'AUC':>8}{'giây':>7}")
        for r in res["results"]:
            auc = "-" if r["roc_auc"] is None else f"{r['roc_auc']:.4f}"
            print(f"  {r['model']:<9}{r['accuracy']:>8.4f}{r['macro_f1']:>9.4f}{r['weighted_f1']:>8.4f}{auc:>8}{r['fit_seconds']:>7}")
    path = folder / f"ml_report_{args.split}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ml] -> {path}")


if __name__ == "__main__":
    main()