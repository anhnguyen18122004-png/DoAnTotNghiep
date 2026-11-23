import argparse
import json
from pathlib import Path

import pandas as pd

from anonymization import DEFAULT_POLICY, anonymize
from cleaning import clean_and_standardize
from config import DATA
from detection import PATTERNS, detect, load_ner
from ingestion import ingest_file

GOLD = DATA / "gold"
LEAK_TYPES = {"PHONE", "EMAIL", "CCCD", "CMND", "PERSON", "ID"}


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4), "tp": tp, "fp": fp, "fn": fn}


def evaluate(findings, truth):
    pred = {(f["column"], f["type"]) for f in findings}
    truth = set(truth)
    per_type = {}
    for t in sorted({t for _, t in pred | truth}):
        pt = {x for x in pred if x[1] == t}
        tt = {x for x in truth if x[1] == t}
        per_type[t] = prf(len(pt & tt), len(pt - tt), len(tt - pt))
    macro = sum(v["f1"] for v in per_type.values()) / len(per_type) if per_type else 0.0
    return {
        "micro": prf(len(pred & truth), len(pred - truth), len(truth - pred)),
        "macro_f1": round(macro, 4),
        "per_type": per_type,
        "false_positives": sorted(pred - truth),
        "false_negatives": sorted(truth - pred),
    }


def leakage(silver, anon, truth):
    out = {}
    for col, typ in sorted(set(truth)):
        if typ not in LEAK_TYPES or col not in anon.columns:
            continue
        pat = PATTERNS.get(typ)
        a, b = silver[col].astype(str), anon[col].astype(str)
        if pat is not None and a.str.contains(pat).mean() < 0.9:
            before = int(a.str.contains(pat).sum())
            after = int(b.str.contains(pat).sum())
            rate = after / before if before else 0.0
        else:
            rate = float((a == b)[silver[col].notna()].mean())
        out[f"{col}:{typ}"] = round(rate, 4)
    return out


def run(silver, truth, ner=None, blind=False, seed=0):
    names = list(silver.columns)
    if blind:
        mapping = {c: f"c{i + 1}" for i, c in enumerate(names)}
        silver = silver.rename(columns=mapping)
        truth = [(mapping[c], t) for c, t in truth]
    findings = detect(silver, ner)
    res = evaluate(findings, truth)
    anon, _ = anonymize(silver, findings, b"detection-eval", policy=dict(DEFAULT_POLICY), seed=seed)
    res["leakage"] = leakage(silver, anon, truth)
    return res


def _print(tag, res):
    m = res["micro"]
    print(f"[{tag}] micro P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} | macro F1={res['macro_f1']:.3f}")
    for t, v in res["per_type"].items():
        print(f"    {t:<8} P={v['precision']:.2f} R={v['recall']:.2f} F1={v['f1']:.2f}  (TP={v['tp']} FP={v['fp']} FN={v['fn']})")
    print(f"    nhầm (FP): {res['false_positives']}")
    print(f"    bỏ sót (FN): {res['false_negatives']}")
    print(f"    còn rò rỉ sau che: {res['leakage']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--truth", nargs="+", required=True, help="cot:LOAI, vi du name:PERSON phone:PHONE")
    ap.add_argument("--no-ner", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    raw, _ = ingest_file(args.file)
    silver, _ = clean_and_standardize(raw)
    truth = [tuple(x.split(":", 1)) for x in args.truth]
    missing = [c for c, _ in truth if c not in silver.columns]
    if missing:
        raise SystemExit(f"Không có cột: {missing}. Cột hiện có: {list(silver.columns)}")

    ner = None if args.no_ner else load_ner()
    report = {}
    for tag, blind in (("tên cột gốc", False), ("tên cột bị ẩn", True)):
        report[tag] = run(silver, truth, ner, blind, args.seed)
        _print(tag, report[tag])
    out = GOLD / Path(args.file).stem
    out.mkdir(parents=True, exist_ok=True)
    (out / "detection_eval.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[eval] -> {out / 'detection_eval.json'}")


if __name__ == "__main__":
    main()