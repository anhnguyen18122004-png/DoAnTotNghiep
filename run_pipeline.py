import argparse
import json
import os
import secrets

from anonymization import anonymize
from cleaning import clean_and_standardize, save_silver
from config import OUTPUT
from detection import detect, load_ner
from evaluation import evaluate, save_gold
from ingestion import ingest_file, ingest_sql
from ml_experiment import run_privacy_utility_experiment


def _split_columns(value):
    return None if not value else [c.strip() for c in value.split(",") if c.strip()]


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
    ap.add_argument("--quasi-identifiers", help="Comma-separated column names")
    ap.add_argument("--sensitive-columns", help="Comma-separated sensitive attribute columns")
    ap.add_argument("--utility-target", help="Optional categorical target column for pre/post ML utility comparison")
    args = ap.parse_args()

    if args.demo:
        from sample_data import make_sample
        args.file = make_sample()

    raw, batch_dir = ingest_file(args.file) if args.file else ingest_sql(args.sql, args.table, args.query)
    print(f"[bronze] {len(raw)} rows -> {batch_dir}")
    silver, cleaning_report = clean_and_standardize(raw)
    silver_dir = save_silver(silver, cleaning_report, batch_dir)
    print(f"[silver] {cleaning_report['rows_out']} rows; removed {cleaning_report['duplicates_removed']} duplicates -> {silver_dir}")

    findings = detect(silver, None if args.no_ner else load_ner())
    for item in findings:
        print(f"[detect] {item['column']:<20} {item['type']:<8} {item['scope']:<7} {item['method']:<11} {item['score']}")

    key = os.environ.get("PII_TOKEN_KEY", "").encode() or secrets.token_bytes(32)
    if "PII_TOKEN_KEY" not in os.environ:
        print("[warn] PII_TOKEN_KEY is unset; token values will differ between runs")
    anonymized, applied = anonymize(silver, findings, key, seed=args.seed)
    transformed_columns = {x["column"] for x in applied}
    findings_after = [f for f in detect(anonymized, None) if f["column"] not in transformed_columns]
    evaluation = evaluate(silver, anonymized, findings, findings_after,
                          _split_columns(args.quasi_identifiers), _split_columns(args.sensitive_columns))
    if args.utility_target:
        evaluation["ml_privacy_utility"] = run_privacy_utility_experiment(silver, anonymized, args.utility_target, seed=args.seed or 42)
    gold_dir = save_gold(anonymized, batch_dir.name, evaluation)

    out = OUTPUT / batch_dir.name
    out.mkdir(parents=True, exist_ok=True)
    anonymized.to_csv(out / "anonymized.csv", index=False, encoding="utf-8-sig")
    (out / "detection.json").write_text(json.dumps(findings, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "anonymization.json").write_text(json.dumps(applied, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "evaluation.json").write_text(json.dumps(evaluation, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    if "ml_privacy_utility" in evaluation:
        (out / "ml_experiment.json").write_text(json.dumps(evaluation["ml_privacy_utility"], ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[anonymize] applied to {len(applied)} columns -> {out}")
    print(f"[gold] k={evaluation['privacy_before']['k_anonymity']} "
          f"risk={evaluation['privacy_before']['risk_score']} ({evaluation['privacy_before']['risk_label']}) -> {gold_dir}")


if __name__ == "__main__":
    main()
