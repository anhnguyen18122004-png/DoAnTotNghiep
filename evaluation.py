"""Dataset profiling, privacy-risk indicators, and Gold-layer persistence."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from config import GOLD

QI_NAME_RE = re.compile(r"(^|_)(age|dob|birth|gender|sex|address|addr|city|district|province|state|zip|postal|occupation|marital_status)(_|$)")
SENSITIVE_NAME_RE = re.compile(r"(salary|income|diagnos|condition|disease|blood_type|religion|ethnicity)")
DIRECT_TYPES = {"PERSON", "PHONE", "EMAIL", "CCCD", "CMND", "ID"}
QI_TYPES = {"AGE", "DOB", "ADDRESS"}
WEIGHTS = {"direct_identifier": 0.35, "k_anonymity": 0.30,
           "unique_quasi_identifiers": 0.20, "sensitive_attribute_disclosure": 0.15}


def _safe_float(value):
    if value is None or pd.isna(value):
        return None
    return float(value)


def _infer_columns(df: pd.DataFrame, findings: list[dict], explicit: list[str] | None, kind: str) -> list[str]:
    if explicit is not None:
        missing = [c for c in explicit if c not in df.columns]
        if missing:
            raise ValueError(f"Unknown {kind} columns: {', '.join(missing)}")
        return list(dict.fromkeys(explicit))
    types = QI_TYPES if kind == "quasi-identifier" else {"SALARY"}
    found = {f["column"] for f in findings if f.get("type") in types}
    regex = QI_NAME_RE if kind == "quasi-identifier" else SENSITIVE_NAME_RE
    return [c for c in df.columns if c in found or regex.search(str(c).lower())]


def profile_data(df: pd.DataFrame, findings: list[dict] | None = None) -> dict:
    findings = findings or []
    nrows, ncols = df.shape
    columns = {}
    for col in df.columns:
        s = df[col]
        columns[str(col)] = {
            "dtype": str(s.dtype),
            "missing_count": int(s.isna().sum()),
            "missing_rate": _safe_float(s.isna().mean()),
            "distinct_count": int(s.nunique(dropna=True)),
            "distinct_rate": _safe_float(s.nunique(dropna=True) / max(nrows, 1)),
            "is_constant": bool(s.nunique(dropna=True) <= 1),
        }
    return {
        "rows": int(nrows), "columns_count": int(ncols),
        "cells_count": int(nrows * ncols),
        "missing_cells": int(df.isna().sum().sum()),
        "missing_rate": _safe_float(df.isna().to_numpy().mean()) if nrows and ncols else 0.0,
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_rate": _safe_float(df.duplicated().mean()) if nrows else 0.0,
        "pii_findings_count": len(findings),
        "pii_columns": list(dict.fromkeys(str(f["column"]) for f in findings)),
        "columns": columns,
    }


def _equivalence_classes(df: pd.DataFrame, qi_cols: list[str]) -> pd.Series | None:
    if not qi_cols or df.empty:
        return None
    return df.groupby(qi_cols, dropna=False, observed=True).size()


def _sensitive_homogeneity(df: pd.DataFrame, qi_cols: list[str], sensitive_cols: list[str]) -> float | None:
    if not qi_cols or not sensitive_cols or df.empty:
        return None
    risks = []
    for col in sensitive_cols:
        group_diversity = df.groupby(qi_cols, dropna=False, observed=True)[col].nunique(dropna=False)
        if len(group_diversity):
            risks.append(float((group_diversity <= 1).mean()))
    return float(np.mean(risks)) if risks else None


def privacy_metrics(df: pd.DataFrame, findings: list[dict], qi_cols: list[str], sensitive_cols: list[str]) -> dict:
    groups = _equivalence_classes(df, qi_cols)
    k_value = int(groups.min()) if groups is not None and len(groups) else None
    unique_rate = float((groups == 1).sum() / max(len(df), 1)) if groups is not None else None
    below_2 = float(groups[groups < 2].sum() / max(len(df), 1)) if groups is not None else None
    below_5 = float(groups[groups < 5].sum() / max(len(df), 1)) if groups is not None else None
    direct_count = sum(f.get("type") in DIRECT_TYPES for f in findings)
    homogeneity = _sensitive_homogeneity(df, qi_cols, sensitive_cols)

    components = {}
    if findings:
        components["direct_identifier"] = min(1.0, direct_count / 3.0)
    if k_value is not None:
        components["k_anonymity"] = 1.0 / max(1, k_value)
        components["unique_quasi_identifiers"] = unique_rate or 0.0
    if homogeneity is not None:
        components["sensitive_attribute_disclosure"] = homogeneity
    active_weight = sum(WEIGHTS[k] for k in components)
    score = (100.0 * sum(WEIGHTS[k] * v for k, v in components.items()) / active_weight) if active_weight else None
    label = "UNKNOWN" if score is None else ("LOW" if score < 30 else "MEDIUM" if score < 60 else "HIGH")
    return {
        "quasi_identifier_columns": qi_cols,
        "sensitive_attribute_columns": sensitive_cols,
        "k_anonymity": k_value,
        "equivalence_classes": int(len(groups)) if groups is not None else None,
        "records_in_classes_below_2_rate": _safe_float(below_2),
        "records_in_classes_below_5_rate": _safe_float(below_5),
        "unique_quasi_identifier_record_rate": _safe_float(unique_rate),
        "direct_identifier_findings": int(direct_count),
        "sensitive_attribute_homogeneity_rate": _safe_float(homogeneity),
        "risk_score": _safe_float(score),
        "risk_label": label,
        "risk_method": "transparent weighted heuristic; not a trained or calibrated model",
        "risk_components": components,
    }


def _distribution_distance(a: pd.Series, b: pd.Series) -> float | None:
    if a.empty or b.empty:
        return None
    ac = a.astype("string").fillna("<NA>").value_counts(normalize=True)
    bc = b.astype("string").fillna("<NA>").value_counts(normalize=True)
    keys = ac.index.union(bc.index)
    return float((ac.reindex(keys, fill_value=0) - bc.reindex(keys, fill_value=0)).abs().sum() / 2)


def utility_comparison(before: pd.DataFrame, after: pd.DataFrame) -> dict:
    results = {}
    distances = []
    for col in before.columns.intersection(after.columns):
        a, b = before[col], after[col]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            am, bm = _safe_float(pd.to_numeric(a, errors="coerce").mean()), _safe_float(pd.to_numeric(b, errors="coerce").mean())
            rel = None if am is None or bm is None else float(abs(bm - am) / max(abs(am), 1e-12))
            results[str(col)] = {"kind": "numeric", "mean_before": am, "mean_after": bm,
                                 "relative_mean_change": rel}
        else:
            tvd = _distribution_distance(a, b)
            results[str(col)] = {"kind": "categorical_distribution", "total_variation_distance": tvd,
                                 "distribution_similarity": None if tvd is None else 1.0 - tvd}
            if tvd is not None:
                distances.append(tvd)
    return {
        "columns_compared": len(results),
        "categorical_distribution_similarity_mean": None if not distances else float(1 - np.mean(distances)),
        "per_column": results,
        "note": "Descriptive distribution comparison only; not a downstream ML utility experiment.",
    }


def evaluate(before: pd.DataFrame, after: pd.DataFrame, findings_before: list[dict],
             findings_after: list[dict] | None = None, quasi_identifier_columns: list[str] | None = None,
             sensitive_attribute_columns: list[str] | None = None) -> dict:
    findings_after = findings_after or []
    qis = _infer_columns(before, findings_before, quasi_identifier_columns, "quasi-identifier")
    qis = [c for c in qis if c in after.columns]
    sensitive = _infer_columns(before, findings_before, sensitive_attribute_columns, "sensitive-attribute")
    sensitive = [c for c in sensitive if c in after.columns]
    return {
        "profile_before": profile_data(before, findings_before),
        "profile_after": profile_data(after, findings_after),
        "privacy_before": privacy_metrics(before, findings_before, qis, sensitive),
        "privacy_after": privacy_metrics(after, findings_after, qis, sensitive),
        "utility_comparison": utility_comparison(before, after),
    }


def save_gold(df: pd.DataFrame, batch_id: str, evaluation: dict) -> Path:
    out = GOLD / batch_id
    out.mkdir(parents=True, exist_ok=False)
    df.to_parquet(out / "data.parquet", index=False)
    (out / "evaluation.json").write_text(
        json.dumps(evaluation, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    return out
