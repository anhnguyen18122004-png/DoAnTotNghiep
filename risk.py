import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from anonymization import DEFAULT_POLICY, anonymize
from cleaning import clean_and_standardize, save_silver
from detection import detect, load_ner
from gold import GOLD, _group_ids
from ingestion import ingest_file

LEVELS = ["Thấp", "Trung bình", "Cao"]
K_HIGH, K_MED = 5, 10


def sa_confidence(df, gid, sa):
    size = gid.map(gid.value_counts())
    conf = pd.Series(0.0, index=df.index)
    for c in sa:
        cnt = df.groupby([gid, df[c]], dropna=False).size()
        top = cnt.groupby(level=0).max()
        conf = np.maximum(conf, gid.map(top) / size)
    return conf


def score_records(df, qi, sa, k_high=K_HIGH, k_med=K_MED):
    gid = _group_ids(df, qi)
    size = gid.map(gid.value_counts()).astype(int)
    conf = sa_confidence(df, gid, sa)
    homogeneous = (conf >= 1.0) & (size >= 2)
    level = np.where(size < k_high, 2, np.where(size < k_med, 1, 0))
    level = np.where(homogeneous, np.minimum(level + 1, 2), level)
    out = df[list(qi) + list(sa)].copy()
    out["class_size"] = size
    out["risk"] = 1.0 / size
    out["sa_conf"] = conf.round(4)
    out["homogeneous"] = homogeneous
    out["level"] = level.astype(int)
    return out


def dataset_risk(scored, n_pii_unmasked=0):
    dist = {LEVELS[i]: round(float((scored["level"] == i).mean()), 4) for i in range(3)}
    avg = float(scored["risk"].mean())
    if n_pii_unmasked > 0 or dist["Cao"] >= 0.2:
        level = "Cao"
    elif dist["Cao"] >= 0.05 or avg > 0.1:
        level = "Trung bình"
    else:
        level = "Thấp"
    return {"level": level, "avg_risk": round(avg, 4), "distribution": dist, "n_pii_unmasked": int(n_pii_unmasked)}


def simulate_attack(scored, n_rounds=20, seed=0):
    rng = np.random.default_rng(seed)
    p = scored["risk"].to_numpy()
    hit = (rng.random((n_rounds, len(p))) < p).mean(axis=0)
    rate = pd.Series(hit).groupby(scored["level"].to_numpy()).mean()
    return {LEVELS[int(i)]: round(float(v), 4) for i, v in rate.items()}


def label_stability(df, qi, sa, base_level, alts=((4, 8), (6, 12))):
    return {
        f"k{kh}-{km}": round(float((score_records(df, qi, sa, kh, km)["level"].to_numpy() == base_level.to_numpy()).mean()), 4)
        for kh, km in alts
    }


def validate(df, qi, sa, scored, seed=0):
    attack = simulate_attack(scored, seed=seed)
    rates = [attack[l] for l in LEVELS if l in attack]
    return {
        "attack_success": attack,
        "monotonic": all(a < b for a, b in zip(rates, rates[1:])),
        "stability": label_stability(df, qi, sa, scored["level"]),
    }


def prepare(path, age_band=None, no_ner=False, seed=None):
    raw, batch_dir = ingest_file(path)
    silver, cleaning = clean_and_standardize(raw)
    save_silver(silver, cleaning, batch_dir)
    findings = detect(silver, None if no_ner else load_ner())
    policy = dict(DEFAULT_POLICY)
    if age_band:
        policy["AGE"] = ("generalize_numeric", {"width": age_band})
    anon, applied = anonymize(silver, findings, b"gold-assessment", policy=policy, seed=seed)
    return silver, anon, findings, {a["column"] for a in applied}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--qi", nargs="+", required=True)
    ap.add_argument("--sa", nargs="+", required=True)
    ap.add_argument("--age-band", type=int)
    ap.add_argument("--no-ner", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    silver, anon, findings, masked = prepare(args.file, args.age_band, args.no_ner, args.seed)
    missing = [c for c in args.qi + args.sa if c not in silver.columns]
    if missing:
        raise SystemExit(f"Không có cột: {missing}. Cột hiện có: {list(silver.columns)}")

    pii = {f["column"] for f in findings}
    out = GOLD / Path(args.file).stem
    out.mkdir(parents=True, exist_ok=True)
    report = {}
    for tag, df, unmasked in (("before", silver, len(pii)), ("after", anon, len(pii - masked))):
        scored = score_records(df, args.qi, args.sa)
        scored.to_parquet(out / f"risk_{tag}.parquet", index=False)
        report[tag] = {"dataset": dataset_risk(scored, unmasked), "validation": validate(df, args.qi, args.sa, scored, args.seed)}
        d, v = report[tag]["dataset"], report[tag]["validation"]
        print(f"[{tag}] mức bộ dữ liệu={d['level']} | Thấp={d['distribution']['Thấp']:.2%} "
              f"Trung bình={d['distribution']['Trung bình']:.2%} Cao={d['distribution']['Cao']:.2%}")
        print(f"        tấn công mô phỏng: {v['attack_success']} | đơn điệu={v['monotonic']} | ổn định nhãn={v['stability']}")
    (out / "risk_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[risk] -> {out}")


if __name__ == "__main__":
    main()