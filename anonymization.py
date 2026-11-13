import hashlib
import hmac
import re

import numpy as np
import pandas as pd

from detection import PATTERNS


def mask(v: str, keep_first: int = 0, keep_last: int = 3, char: str = "*") -> str:
    v = str(v)
    if len(v) <= keep_first + keep_last:
        return char * len(v)
    return v[:keep_first] + char * (len(v) - keep_first - keep_last) + v[len(v) - keep_last:]


def mask_email(v: str) -> str:
    local, _, domain = str(v).partition("@")
    return f"{local[:1]}***@{domain}"


def tokenize(v: str, key: bytes, prefix: str = "TOK") -> str:
    return f"{prefix}_{hmac.new(key, str(v).encode('utf-8'), hashlib.sha256).hexdigest()[:12]}"


def generalize_numeric(s: pd.Series, width: int = 5) -> pd.Series:
    lo = (s // width * width).astype("Int64")
    return (lo.astype("string") + "-" + (lo + width - 1).astype("string")).where(s.notna())


def generalize_date(s: pd.Series, level: str = "year") -> pd.Series:
    if level == "year":
        return s.dt.year.astype("Int64").astype("string")
    if level == "month":
        return s.dt.to_period("M").astype("string")
    raise ValueError(level)


def generalize_address(v: str, keep: int = 2) -> str:
    parts = [p.strip() for p in str(v).split(",") if p.strip()]
    return ", ".join(parts[-keep:]) if len(parts) > keep else ", ".join(parts)


def laplace_perturb(s: pd.Series, epsilon: float, lower: float | None = None,
                    upper: float | None = None, seed: int | None = None) -> pd.Series:
    lower = s.min() if lower is None else lower
    upper = s.max() if upper is None else upper
    rng = np.random.default_rng(seed)
    scale = (upper - lower) / epsilon
    out = s.clip(lower, upper) + rng.laplace(0.0, scale, size=len(s))
    return out.clip(lower, upper).round(0).astype("Int64")


def dp_histogram(s: pd.Series, epsilon: float, seed: int | None = None) -> pd.Series:
    rng = np.random.default_rng(seed)
    counts = s.value_counts()
    noisy = counts + rng.laplace(0.0, 1.0 / epsilon, size=len(counts))
    return noisy.clip(lower=0).round(0).astype(int)


DEFAULT_POLICY = {
    "PHONE": ("mask", {"keep_last": 3}),
    "EMAIL": ("mask_email", {}),
    "CCCD": ("mask", {"keep_last": 0}),
    "CMND": ("mask", {"keep_last": 0}),
    "PERSON": ("mask", {"keep_last": 0}),
    "ID": ("mask", {"keep_last": 0}),
    "ADDRESS": ("generalize_address", {"keep": 2}),
    "DOB": ("generalize_date", {"level": "year"}),
    "SALARY": ("dp", {"epsilon": 1.0}),
}


def _apply_value(method: str, params: dict, key: bytes):
    if method == "mask":
        return lambda v: mask(v, **params)
    if method == "mask_email":
        return mask_email
    if method == "tokenize":
        return lambda v: tokenize(v, key, **params)
    if method == "generalize_address":
        return lambda v: generalize_address(v, **params)
    raise ValueError(method)


def anonymize(df: pd.DataFrame, findings: list[dict], key: bytes,
              policy: dict | None = None, seed: int | None = None):
    policy = policy or DEFAULT_POLICY
    out = df.copy()
    applied = []
    for f in findings:
        col, typ = f["column"], f["type"]
        if typ not in policy:
            continue
        method, params = policy[typ]
        s = out[col]

        if f["scope"] == "inline":
            fn = _apply_value(method, params, key)
            pat = PATTERNS[typ]
            out[col] = s.map(lambda t: pat.sub(lambda m: fn(m.group(0)), t), na_action="ignore")
        elif method == "generalize_numeric":
            out[col] = generalize_numeric(pd.to_numeric(s), **params)
        elif method == "generalize_date":
            out[col] = generalize_date(pd.to_datetime(s), **params)
        elif method == "dp":
            out[col] = laplace_perturb(pd.to_numeric(s), seed=seed, **params)
        else:
            out[col] = s.map(_apply_value(method, params, key), na_action="ignore")
        applied.append({"column": col, "type": typ, "method": method, **params})
    return out, applied