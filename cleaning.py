import json
import re
import unicodedata

import pandas as pd

from config import SILVER

NULL_TOKENS = {"", "na", "n/a", "null", "none", "nan", "-", "--"}
EMAIL_RE = re.compile(r"^[\w.+-]+@[\w-]+(?:\.[\w-]+)+$")
PHONE_RE = re.compile(r"^0[35789]\d{8}$")
CCCD_RE = re.compile(r"^0\d{11}$")
NUM_RE = re.compile(r"^-?\d+(\.\d+)?$")
DATE_RE = re.compile(r"^\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}$")


def snake(name: str) -> str:
    s = unicodedata.normalize("NFD", str(name).replace("đ", "d").replace("Đ", "D"))
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def _basic(s: pd.Series) -> pd.Series:
    s = s.astype("string").str.normalize("NFC").str.strip()
    s = s.str.replace(r"\s+", " ", regex=True)
    return s.mask(s.str.lower().isin(NULL_TOKENS), pd.NA)


DIGITISH_RE = re.compile(r"^\+?[\d\s.\-()]+$")


def _digits(v: str) -> str:
    return re.sub(r"\D", "", v) if DIGITISH_RE.match(v) else v


def _phone(v: str) -> str:
    if not DIGITISH_RE.match(v):
        return v
    d = re.sub(r"\D", "", v)
    if d.startswith("84") and len(d) == 11:
        d = "0" + d[2:]
    elif len(d) == 9 and d[0] in "35789":
        d = "0" + d
    return d


def _ratio(s: pd.Series, fn) -> float:
    nn = s.dropna()
    return 0.0 if nn.empty else nn.map(fn).mean()


def clean_and_standardize(df: pd.DataFrame):
    report = {"rows_in": len(df), "standardized": {}, "invalid": {}}
    df = df.copy()
    df.columns = [snake(c) for c in df.columns]
    df = df.loc[:, ~df.columns.duplicated()]

    for c in df.columns:
        df[c] = _basic(df[c])

    df = df.dropna(how="all")
    n = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    report["duplicates_removed"] = n - len(df)

    skip_numeric = set()
    for c in df.columns:
        s = df[c]
        if _ratio(s, lambda v: bool(PHONE_RE.match(_phone(v)))) >= 0.5:
            df[c] = s.map(_phone, na_action="ignore")
            report["standardized"][c] = "phone"
            report["invalid"][c] = int((~df[c].dropna().str.match(PHONE_RE)).sum())
        elif _ratio(s, lambda v: bool(EMAIL_RE.match(v.lower()))) >= 0.5:
            df[c] = s.str.lower()
            report["standardized"][c] = "email"
            report["invalid"][c] = int((~df[c].dropna().str.match(EMAIL_RE)).sum())
        elif _ratio(s, lambda v: bool(CCCD_RE.match(_digits(v)))) >= 0.5:
            df[c] = s.map(_digits, na_action="ignore")
            report["standardized"][c] = "cccd"
            report["invalid"][c] = int((~df[c].dropna().str.match(CCCD_RE)).sum())
        else:
            continue
        skip_numeric.add(c)

    for c in df.columns:
        if c in skip_numeric:
            continue
        s = df[c].dropna()
        if s.empty:
            continue
        if _ratio(df[c], lambda v: bool(DATE_RE.match(v))) >= 0.8:
            parsed = pd.to_datetime(df[c], dayfirst=True, errors="coerce", format="mixed")
            report["invalid"][c] = int(parsed.isna().sum() - df[c].isna().sum())
            df[c] = parsed
            report["standardized"][c] = "date"
        elif s.map(lambda v: bool(NUM_RE.match(v))).all():
            fixed_width_id = s.str.len().nunique() == 1 and s.str.len().iloc[0] >= 9
            leading_zero = s.str.match(r"^0\d").any()
            if not fixed_width_id and not leading_zero:
                df[c] = pd.to_numeric(df[c])
                report["standardized"][c] = "numeric"

    report["rows_out"] = len(df)
    report["nulls"] = {c: int(v) for c, v in df.isna().sum().items() if v}
    return df, report


def save_silver(df: pd.DataFrame, report: dict, batch_dir):
    out = SILVER / batch_dir.name
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "data.parquet", index=False)
    (out / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return out