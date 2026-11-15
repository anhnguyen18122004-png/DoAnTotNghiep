"""Human-reviewed labels and supervised privacy-risk classification workflow."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

FEATURE_COLUMNS = [
    "risk_score", "k_anonymity", "records_below_2_rate", "records_below_5_rate",
    "unique_qi_rate", "direct_identifier_findings", "sensitive_homogeneity_rate",
    "missing_rate", "duplicate_rate", "pii_columns_count", "qi_columns_count",
    "sensitive_columns_count",
]
RISK_LABELS = {"LOW", "MEDIUM", "HIGH"}


def evaluation_to_features(path: str | Path) -> dict:
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    privacy = data["privacy_before"]
    profile = data["profile_before"]
    return {
        "batch_id": path.parent.name,
        "risk_score": privacy.get("risk_score"),
        "heuristic_risk_label": privacy.get("risk_label"),
        "reviewed_risk_label": "",
        "k_anonymity": privacy.get("k_anonymity"),
        "records_below_2_rate": privacy.get("records_in_classes_below_2_rate"),
        "records_below_5_rate": privacy.get("records_in_classes_below_5_rate"),
        "unique_qi_rate": privacy.get("unique_quasi_identifier_record_rate"),
        "direct_identifier_findings": privacy.get("direct_identifier_findings"),
        "sensitive_homogeneity_rate": privacy.get("sensitive_attribute_homogeneity_rate"),
        "missing_rate": profile.get("missing_rate"),
        "duplicate_rate": profile.get("duplicate_rate"),
        "pii_columns_count": len(profile.get("pii_columns", [])),
        "qi_columns_count": len(privacy.get("quasi_identifier_columns", [])),
        "sensitive_columns_count": len(privacy.get("sensitive_attribute_columns", [])),
    }


def export_review_template(evaluation_paths: list[str | Path], output_path: str | Path) -> pd.DataFrame:
    if not evaluation_paths:
        raise ValueError("Provide at least one evaluation.json file")
    rows = [evaluation_to_features(p) for p in evaluation_paths]
    frame = pd.DataFrame(rows)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False, encoding="utf-8-sig")
    return frame


def _model_factories(n_classes: int):
    models = {
        "LogisticRegression": lambda: LogisticRegression(max_iter=600, class_weight="balanced"),
        "RandomForest": lambda: RandomForestClassifier(n_estimators=200, max_depth=10,
                                                         min_samples_leaf=2,
                                                         class_weight="balanced_subsample",
                                                         n_jobs=-1, random_state=42),
    }
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = lambda: XGBClassifier(
            n_estimators=150, max_depth=4, learning_rate=0.08, n_jobs=2,
            objective="binary:logistic" if n_classes == 2 else "multi:softprob",
            eval_metric="logloss" if n_classes == 2 else "mlogloss",
            **({} if n_classes == 2 else {"num_class": n_classes}),
        )
    except Exception:
        pass
    try:
        from lightgbm import LGBMClassifier
        models["LightGBM"] = lambda: LGBMClassifier(
            n_estimators=150, learning_rate=0.08, num_leaves=15,
            class_weight="balanced", n_jobs=2, verbosity=-1,
        )
    except Exception:
        pass
    return models


def train_risk_classifier(labels_csv: str | Path, target_column: str = "reviewed_risk_label",
                          seed: int = 42, test_size: float = 0.25,
                          model_output: str | Path | None = None) -> dict:
    frame = pd.read_csv(labels_csv)
    if target_column not in frame:
        raise ValueError(f"Missing target column: {target_column}")
    frame[target_column] = frame[target_column].astype("string").str.upper().str.strip()
    frame = frame[frame[target_column].isin(RISK_LABELS)].copy()
    if len(frame) < 20:
        raise ValueError("At least 20 rows with reviewed LOW/MEDIUM/HIGH labels are required")
    counts = frame[target_column].value_counts()
    if counts.min() < 5:
        raise ValueError("Each reviewed risk class needs at least 5 examples")
    feature_cols = [c for c in FEATURE_COLUMNS if c in frame.columns]
    if len(feature_cols) < 3:
        raise ValueError("Not enough risk feature columns; export the review template first")
    x = frame[feature_cols].apply(pd.to_numeric, errors="coerce")
    y_encoder = LabelEncoder()
    y = y_encoder.fit_transform(frame[target_column])
    train_idx, test_idx = train_test_split(
        np.arange(len(frame)), test_size=test_size, random_state=seed, stratify=y
    )
    best_name, best_score, best_model = None, -1.0, None
    results = {}
    classes = np.unique(y[test_idx])
    for name, factory in _model_factories(len(y_encoder.classes_)).items():
        try:
            model = Pipeline([
                ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                ("scale", StandardScaler()),
                ("classifier", factory()),
            ])
            model.fit(x.iloc[train_idx], y[train_idx])
            pred = model.predict(x.iloc[test_idx])
            score = float(f1_score(y[test_idx], pred, average="weighted", zero_division=0))
            results[name] = {
                "accuracy": float(accuracy_score(y[test_idx], pred)),
                "precision_weighted": float(precision_score(y[test_idx], pred, average="weighted", zero_division=0)),
                "recall_weighted": float(recall_score(y[test_idx], pred, average="weighted", zero_division=0)),
                "f1_weighted": score,
                "confusion_matrix": confusion_matrix(y[test_idx], pred, labels=classes).tolist(),
            }
            if score > best_score:
                best_name, best_score, best_model = name, score, model
        except Exception as exc:
            results[name] = {"error": f"{type(exc).__name__}: {exc}"}
    if best_model is None:
        raise RuntimeError("All risk classifiers failed")
    heuristic_agreement = None
    if "heuristic_risk_label" in frame:
        heuristic_agreement = float((frame["heuristic_risk_label"].astype(str).str.upper() == frame[target_column]).mean())
    if model_output:
        joblib.dump({"pipeline": best_model, "label_encoder": y_encoder,
                     "feature_columns": feature_cols, "best_model": best_name}, model_output)
    return {
        "target_column": target_column, "rows_reviewed": int(len(frame)),
        "rows_train": int(len(train_idx)), "rows_test": int(len(test_idx)),
        "class_counts": {str(k): int(v) for k, v in counts.items()},
        "classes": list(y_encoder.classes_), "feature_columns": feature_cols,
        "heuristic_vs_reviewed_agreement": heuristic_agreement,
        "models": results, "best_model": best_name,
        "label_source": "human-reviewed CSV; heuristic labels are retained only for agreement analysis",
    }


def main():
    parser = argparse.ArgumentParser(description="Export risk-label review sheets or train a reviewed-label model")
    parser.add_argument("--export-review-template", nargs="+", help="evaluation.json paths from multiple batches")
    parser.add_argument("--out", default="risk_review_template.csv")
    parser.add_argument("--train", help="CSV containing reviewed risk labels")
    parser.add_argument("--target", default="reviewed_risk_label")
    parser.add_argument("--model-out", default="risk_classifier.joblib")
    args = parser.parse_args()
    if args.export_review_template:
        table = export_review_template(args.export_review_template, args.out)
        print(f"Wrote {len(table)} review rows to {args.out}; fill reviewed_risk_label before training.")
    elif args.train:
        report = train_risk_classifier(args.train, args.target, model_output=args.model_out)
        report_path = Path(args.model_out).with_suffix(".metrics.json")
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Best model: {report['best_model']}; report: {report_path}")
    else:
        parser.error("Choose --export-review-template or --train")


if __name__ == "__main__":
    main()
