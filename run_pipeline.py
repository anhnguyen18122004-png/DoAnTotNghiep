import argparse
import json
import os
import secrets

import pandas as pd

from anonymization import anonymize
from cleaning import clean_and_standardize, save_silver
from config import OUTPUT
from detection import detect, load_ner
from ingestion import ingest_file, ingest_sql


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--file")
    src.add_argument("--sql", help="SQLAlchemy URL")
    src.add_argument("--demo", action="store_true")
    ap.add_argument("--table")
    ap.add_argument("--query")
    ap.add_argument("--no-ner", action="store_true")
    ap.add_argument("--seed", type=int)
    args = ap.parse_args()

    if args.demo:
        from sample_data import make_sample
        args.file = make_sample()

    raw, batch_dir = ingest_file(args.file) if args.file else ingest_sql(args.sql, args.table, args.query)
    print(f"[bronze] {len(raw)} dòng -> {batch_dir}")

    silver, report = clean_and_standardize(raw)
    silver_dir = save_silver(silver, report, batch_dir)
    print(f"[silver] {report['rows_out']} dòng, bỏ {report['duplicates_removed']} trùng -> {silver_dir}")

    findings = detect(silver, None if args.no_ner else load_ner())
    for f in findings:
        print(f"[detect] {f['column']:<12} {f['type']:<8} {f['scope']:<7} {f['method']:<11} {f['score']}")

    key = os.environ.get("PII_TOKEN_KEY", "").encode() or secrets.token_bytes(32)
    if "PII_TOKEN_KEY" not in os.environ:
        print("[warn] chưa đặt PII_TOKEN_KEY: token sẽ khác nhau giữa các lần chạy")
    anon, applied = anonymize(silver, findings, key, seed=args.seed)

    out = OUTPUT / batch_dir.name
    out.mkdir(parents=True, exist_ok=True)
    anon.to_csv(out / "anonymized.csv", index=False, encoding="utf-8-sig")
    (out / "detection.json").write_text(json.dumps(findings, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "anonymization.json").write_text(json.dumps(applied, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[anonymize] -> {out}")
    print(anon.head(5).to_string())


if __name__ == "__main__":
    main()