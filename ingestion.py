import hashlib
import json
import re
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd

from config import BRONZE


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _store(df: pd.DataFrame, source: str, kind: str, raw: Path | None = None):
    batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir = BRONZE / batch_id
    batch_dir.mkdir(parents=True)
    manifest = {
        "batch_id": batch_id,
        "kind": kind,
        "source": source,
        "ingested_at": datetime.now().isoformat(timespec="seconds"),
        "rows": len(df),
        "columns": list(df.columns),
    }
    if raw is not None:
        shutil.copy2(raw, batch_dir / raw.name)
        manifest["sha256"] = _sha256(raw)
    df.to_parquet(batch_dir / "data.parquet", index=False)
    (batch_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return df, batch_dir


def ingest_file(path: str):
    path = Path(path)
    ext = path.suffix.lower()
    if ext in (".csv", ".tsv"):
        df = pd.read_csv(
            path,
            sep="\t" if ext == ".tsv" else ",",
            dtype="string",
            keep_default_na=False,
            encoding="utf-8-sig",
        )
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(path, dtype="string", keep_default_na=False)
    else:
        raise ValueError(f"Định dạng không hỗ trợ: {ext}")
    return _store(df, str(path), "file", raw=path)


def ingest_sql(url: str, table: str | None = None, query: str | None = None):
    from sqlalchemy import create_engine, text

    if (table is None) == (query is None):
        raise ValueError("Chỉ truyền một trong hai: table hoặc query")
    if table is not None:
        if not re.fullmatch(r"[\w.]+", table):
            raise ValueError("Tên bảng không hợp lệ")
        query = f"SELECT * FROM {table}"
    if not query.lstrip().lower().startswith("select"):
        raise ValueError("Chỉ cho phép câu lệnh SELECT")
    engine = create_engine(url)
    with engine.connect() as conn:
        df = pd.read_sql(text(query), conn)
    df = df.astype("string")
    safe_url = engine.url.render_as_string(hide_password=True)
    return _store(df, f"{safe_url} :: {query}", "sql")