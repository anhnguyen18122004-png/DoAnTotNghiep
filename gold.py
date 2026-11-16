import argparse
import json
from pathlib import Path

import pandas as pd

from anonymization import DEFAULT_POLICY, anonymize
from cleaning import clean_and_standardize, save_silver
from config import DATA
from detection import detect, load_ner
from ingestion import ingest_file

GOLD = DATA / "gold"


def column_profile(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    rows = []
    for c in df.columns:
        nu = int(df[c].nunique(dropna=True))
        rows.append({
            "column": c,
            "dtype": str(df[c].dtype),
            "missing_rate": round(float(df[c].isna().mean()), 4),
            "n_unique": nu,
            "uniqueness_ratio": round(nu / n, 4) if n else 0.0,
        })
    return pd.DataFrame(rows)


def dataset_profile(df, findings=None, cleaning=None, masked=()):
    findings = findings or []
    pii_cols = {f["column"] for f in findings}
    rows_in = cleaning["rows_in"] if cleaning else len(df)
    dups = cleaning["duplicates_removed"] if cleaning else 0
    return {
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "missing_rate": round(float(df.isna().mean().mean()), 4),
        "duplicate_rate": round(dups / rows_in, 4) if rows_in else 0.0,
        "n_pii_cols": len(pii_cols),
        "n_pii_unmasked": len(pii_cols - set(masked)),
        "pii_types": sorted({f["type"] for f in findings}),
    }


def _group_ids(df, qi):
    return df.groupby(list(qi), dropna=False, observed=True).ngroup()


def record_risk(df, qi) -> pd.DataFrame:
    gid = _group_ids(df, qi)
    size = gid.map(gid.value_counts()).astype(int)
    return pd.DataFrame({"class_size": size, "risk": 1.0 / size})


def k_anonymity(df, qi):
    gid = _group_ids(df, qi)
    size = gid.map(gid.value_counts())
    return {
        "qi": list(qi),
        "n_classes": int(gid.nunique()),
        "k_min": int(size.min()),
        "pct_unique": round(float((size == 1).mean()), 4),
        "pct_k_ge_2": round(float((size >= 2).mean()), 4),
        "pct_k_ge_5": round(float((size >= 5).mean()), 4),
        "pct_k_ge_10": round(float((size >= 10).mean()), 4),
        "avg_risk": round(float((1.0 / size).mean()), 4),
        "max_risk": round(float(1.0 / size.min()), 4),
    }


def l_diversity(df, qi, sa):
    gid = _group_ids(df, qi)
    per_class = df[sa].groupby(gid).nunique(dropna=True)
    per_record = gid.map(per_class)
    return {
        "sa": sa,
        "l_min": int(per_class.min()),
        "mean_l": round(float(per_record.mean()), 3),
        "pct_homogeneous": round(float((per_record == 1).mean()), 4),
    }


def assess(df, qi, sa, findings=None, cleaning=None, masked=()):
    return {
        "profile": dataset_profile(df, findings, cleaning, masked),
        "k_anonymity": k_anonymity(df, qi),
        "l_diversity": [l_diversity(df, qi, c) for c in sa],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--qi", nargs="+", required=True)
    ap.add_argument("--sa", nargs="+", required=True)
    ap.add_argument("--age-band", type=int)
    ap.add_argument("--no-ner", action="store_true")
    ap.add_argument("--seed", type=int)
    args = ap.parse_args()

    raw, batch_dir = ingest_file(args.file)
    silver, cleaning = clean_and_standardize(raw)
    save_silver(silver, cleaning, batch_dir)
    findings = detect(silver, None if args.no_ner else load_ner())

    policy = dict(DEFAULT_POLICY)
    if args.age_band:
        policy["AGE"] = ("generalize_numeric", {"width": args.age_band})
    anon, applied = anonymize(silver, findings, b"gold-assessment", policy=policy, seed=args.seed)
    masked = {a["column"] for a in applied}

    missing = [c for c in args.qi + args.sa if c not in silver.columns]
    if missing:
        raise SystemExit(f"Không có cột: {missing}. Cột hiện có: {list(silver.columns)}")

    result = {
        "before": assess(silver, args.qi, args.sa, findings, cleaning),
        "after": assess(anon, args.qi, args.sa, findings, cleaning, masked),
    }
    out = GOLD / Path(args.file).stem
    out.mkdir(parents=True, exist_ok=True)
    (out / "assessment.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    column_profile(silver).to_csv(out / "column_profile.csv", index=False, encoding="utf-8-sig")

    for tag in ("before", "after"):
        r = result[tag]
        k, p = r["k_anonymity"], r["profile"]
        print(f"[{tag}] dòng={p['n_rows']} PII={p['n_pii_cols']} chưa che={p['n_pii_unmasked']} | "
              f"k_min={k['k_min']} duy nhất={k['pct_unique']:.2%} rủi ro TB={k['avg_risk']:.4f} k>=5={k['pct_k_ge_5']:.2%}")
        for l in r["l_diversity"]:
            print(f"        l-diversity({l['sa']}): l_min={l['l_min']} đồng nhất={l['pct_homogeneous']:.2%}")
    print(f"[gold] -> {out}")


if __name__ == "__main__":
    main()