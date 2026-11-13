import argparse
import json
import os
import secrets
import time
from pathlib import Path

from anonymization import anonymize
from cleaning import clean_and_standardize, save_silver
from detection import detect, load_ner
from ingestion import ingest_file

ROOT = Path(__file__).resolve().parent
EXTS = {".csv", ".tsv", ".xlsx", ".xls"}


def process(path: Path, out_dir: Path, key: bytes, ner, seed):
    raw, batch_dir = ingest_file(str(path))
    silver, cleaning = clean_and_standardize(raw)
    save_silver(silver, cleaning, batch_dir)
    findings = detect(silver, ner)
    anon, applied = anonymize(silver, findings, key, seed=seed)

    anon.to_csv(out_dir / f"{path.stem}_anonymized.csv", index=False, encoding="utf-8-sig")
    report = {"detection": findings, "anonymization": applied, "cleaning": cleaning}
    (out_dir / f"{path.stem}_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return len(raw), len(anon), len(applied)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(ROOT / "input"))
    ap.add_argument("--output", default=str(ROOT / "output"))
    ap.add_argument("--no-ner", action="store_true")
    ap.add_argument("--seed", type=int)
    args = ap.parse_args()

    in_dir, out_dir = Path(args.input), Path(args.output)
    in_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in in_dir.iterdir() if p.suffix.lower() in EXTS)
    if not files:
        print(f"Không có file CSV/Excel nào trong {in_dir}")
        return

    key = os.environ.get("PII_TOKEN_KEY", "").encode() or secrets.token_bytes(32)
    if "PII_TOKEN_KEY" not in os.environ:
        print("[warn] chưa đặt PII_TOKEN_KEY: token sẽ khác nhau giữa các lần chạy")
    ner = None if args.no_ner else load_ner()

    for i, f in enumerate(files):
        if i:
            time.sleep(1.1)  # batch_id của ingestion tính theo giây
        try:
            n_in, n_out, n_cols = process(f, out_dir, key, ner, args.seed)
            print(f"[ok]   {f.name}: {n_in} -> {n_out} dòng, che {n_cols} cột -> {f.stem}_anonymized.csv")
        except Exception as e:
            print(f"[loi]  {f.name}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()