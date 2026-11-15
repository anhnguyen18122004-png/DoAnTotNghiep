import argparse
import json
import os
import secrets
from pathlib import Path

from anonymization import anonymize
from cleaning import clean_and_standardize, save_silver
from config import OUTPUT
from detection import detect, load_ner
from evaluation import evaluate, save_gold
from ingestion import ingest_file
from ml_experiment import run_privacy_utility_experiment

ROOT = Path(__file__).resolve().parent
EXTS = {".csv", ".tsv", ".xlsx", ".xls"}


def _split_columns(value):
    return None if not value else [c.strip() for c in value.split(",") if c.strip()]


def process(path, out_dir, key, ner, seed, quasi_identifiers=None,
            sensitive_columns=None, utility_target=None):
    raw, batch_dir = ingest_file(str(path))
    silver, cleaning = clean_and_standardize(raw)
    save_silver(silver, cleaning, batch_dir)
    findings = detect(silver, ner)
    anon, applied = anonymize(silver, findings, key, seed=seed)
    changed = {x["column"] for x in applied}
    remaining = [f for f in detect(anon, None) if f["column"] not in changed]
    evaluation = evaluate(silver, anon, findings, remaining,
                          quasi_identifiers, sensitive_columns)
    if utility_target:
        evaluation["ml_privacy_utility"] = run_privacy_utility_experiment(
            silver, anon, utility_target, seed=seed or 42
        )
    save_gold(anon, batch_dir.name, evaluation)

    anon.to_csv(out_dir / f"{path.stem}_anonymized.csv", index=False, encoding="utf-8-sig")
    report = {"batch_id": batch_dir.name, "detection": findings,
              "anonymization": applied, "cleaning": cleaning, "evaluation": evaluation}
    (out_dir / f"{path.stem}_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    return len(raw), len(anon), len(applied), evaluation


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(ROOT / "input"))
    ap.add_argument("--output", default=str(ROOT / "output"))
    ap.add_argument("--no-ner", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quasi-identifiers", help="Comma-separated column names")
    ap.add_argument("--sensitive-columns", help="Comma-separated sensitive columns")
    ap.add_argument("--utility-target", help="Optional categorical target for ML utility comparison")
    args = ap.parse_args()

    in_dir, out_dir = Path(args.input), Path(args.output)
    in_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in in_dir.iterdir() if p.suffix.lower() in EXTS)
    if not files:
        print(f"No CSV or Excel files found in {in_dir}")
        return

    key = os.environ.get("PII_TOKEN_KEY", "").encode() or secrets.token_bytes(32)
    if "PII_TOKEN_KEY" not in os.environ:
        print("[warn] PII_TOKEN_KEY is unset; token values differ between runs")
    ner = None if args.no_ner else load_ner()
    for file in files:
        try:
            n_in, n_out, n_cols, report = process(
                file, out_dir, key, ner, args.seed,
                _split_columns(args.quasi_identifiers),
                _split_columns(args.sensitive_columns), args.utility_target,
            )
            risk = report["privacy_before"]
            print(f"[ok] {file.name}: {n_in}->{n_out} rows; {n_cols} columns transformed; "
                  f"risk={risk['risk_score']} ({risk['risk_label']}); k={risk['k_anonymity']}")
        except Exception as exc:
            print(f"[error] {file.name}: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
