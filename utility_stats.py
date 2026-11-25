import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from cleaning import clean_and_standardize, save_silver
from detection import detect, load_ner
from gold import GOLD
from ingestion import ingest_file
from utility import apply_config, build_configs, privacy_metrics

BAND = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*-\s*(-?\d+(?:\.\d+)?)\s*$")


def numeric_view(s):
    def conv(v):
        if pd.isna(v):
            return np.nan
        m = BAND.match(str(v))
        return (float(m.group(1)) + float(m.group(2)) + 1) / 2 if m else float(v)

    return s.map(conv).astype(float)


def ks_distance(a, b):
    a, b = np.sort(a.dropna().to_numpy()), np.sort(b.dropna().to_numpy())
    grid = np.concatenate([a, b])
    return float(np.max(np.abs(np.searchsorted(a, grid, side="right") / len(a) - np.searchsorted(b, grid, side="right") / len(b))))


def rel_err(x, y):
    return abs(y - x) / abs(x) if x else 0.0


def column_utility(orig, anon):
    o, a = numeric_view(orig), numeric_view(anon)
    rng = float(o.max() - o.min()) or 1.0
    return {
        "mean_rel_err": round(rel_err(o.mean(), a.mean()), 4),
        "std_rel_err": round(rel_err(o.std(), a.std()), 4),
        "ks": round(ks_distance(o, a), 4),
        "record_mae_norm": round(float((o - a).abs().mean() / rng), 4),
    }


def group_mean_error(orig, anon, value, by):
    g_o = numeric_view(orig[value]).groupby(orig[by]).mean()
    g_a = numeric_view(anon[value]).groupby(anon[by]).mean()
    return round(float(((g_a - g_o).abs() / g_o.abs()).mean()), 4)


def run(silver, findings, numeric_cols, groups, qi, sa, age_bands=(5, 10, 20), epsilons=(100.0, 10.0, 1.0),
        bins=None, dp_cols=(), seed=0):
    pii = {f["column"] for f in findings}
    rows = []
    for name, over, extra in build_configs(age_bands, epsilons, bins or {}, dp_cols):
        df, masked = apply_config(silver, findings, over, extra, seed)
        rows.append({
            "config": name,
            "privacy": privacy_metrics(df, qi, sa, len(pii - masked)),
            "columns": {c: column_utility(silver[c], df[c]) for c in numeric_cols},
            "group_mean_err": {f"{v}~{b}": group_mean_error(silver, df, v, b) for v, b in groups},
        })
    return {"rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--numeric", nargs="+", required=True)
    ap.add_argument("--groups", nargs="*", default=[], help="gia_tri:nhom, vi du billing_amount:medical_condition")
    ap.add_argument("--qi", nargs="+", required=True)
    ap.add_argument("--sa", nargs="+", required=True)
    ap.add_argument("--age-bands", nargs="*", type=int, default=[5, 10, 20])
    ap.add_argument("--epsilons", nargs="*", type=float, default=[100.0, 10.0, 1.0])
    ap.add_argument("--bins", nargs="*", default=[], help="cot:do_rong")
    ap.add_argument("--dp-cols", nargs="*", default=[])
    ap.add_argument("--no-ner", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    raw, batch_dir = ingest_file(args.file)
    silver, cleaning = clean_and_standardize(raw)
    save_silver(silver, cleaning, batch_dir)
    findings = detect(silver, None if args.no_ner else load_ner())
    groups = [tuple(g.split(":")) for g in args.groups]
    missing = [c for c in args.numeric + args.qi + args.sa + [x for g in groups for x in g] if c not in silver.columns]
    if missing:
        raise SystemExit(f"Không có cột: {missing}. Cột hiện có: {list(silver.columns)}")

    bins = {b.split(":")[0]: int(b.split(":")[1]) for b in args.bins}
    res = run(silver, findings, args.numeric, groups, args.qi, args.sa, args.age_bands, args.epsilons, bins, args.dp_cols, args.seed)
    out = GOLD / Path(args.file).stem
    out.mkdir(parents=True, exist_ok=True)
    (out / "utility_stats.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    head = f"{'cấu hình':<14}{'rủi ro TB':>10}{'duy nhất':>10}"
    for c in args.numeric:
        head += f"{c[:9] + ' KS':>13}{'lệch TB':>9}{'MAE/rng':>9}"
    for v, b in groups:
        head += f"{(v[:7] + '~' + b[:5]):>14}"
    print(head)
    for r in res["rows"]:
        p = r["privacy"]
        line = f"{r['config']:<14}{p['avg_risk']:>10.4f}{p['pct_unique']:>10.2%}"
        for c in args.numeric:
            u = r["columns"][c]
            line += f"{u['ks']:>13.4f}{u['mean_rel_err']:>9.4f}{u['record_mae_norm']:>9.4f}"
        for v, b in groups:
            line += f"{r['group_mean_err'][f'{v}~{b}']:>14.4f}"
        print(line)
    print(f"[utility_stats] -> {out / 'utility_stats.json'}")


if __name__ == "__main__":
    main()