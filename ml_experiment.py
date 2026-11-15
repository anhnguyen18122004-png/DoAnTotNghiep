"""Train comparable utility models on data before and after anonymization."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def _model_factories(n_classes: int):
    models = {
        "LogisticRegression": lambda: LogisticRegression(max_iter=600, class_weight="balanced"),
        "RandomForest": lambda: RandomForestClassifier(n_estimators=100, max_depth=18,
                                                         min_samples_leaf=2,
                                                         class_weight="balanced_subsample",
                                                         n_jobs=-1, random_state=42),
    }
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = lambda: XGBClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.08, subsample=0.85,
            colsample_bytree=0.85, n_jobs=2, verbosity=0, random_state=42,
            objective="binary:logistic" if n_classes == 2 else "multi:softprob",
            eval_metric="logloss" if n_classes == 2 else "mlogloss",
            **({} if n_classes == 2 else {"num_class": n_classes}),
        )
    except Exception:
        pass
    try:
        from lightgbm import LGBMClassifier
        models["LightGBM"] = lambda: LGBMClassifier(
            n_estimators=100, learning_rate=0.08, max_depth=-1, num_leaves=31,
            class_weight="balanced", n_jobs=2, verbosity=-1, random_state=42,
        )
    except Exception:
        pass
    return models


def _pipeline(model, x: pd.DataFrame):
    numeric = [c for c in x.columns if pd.api.types.is_numeric_dtype(x[c])]
    categorical = [c for c in x.columns if c not in numeric]
    transformers = []
    if numeric:
        transformers.append(("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), numeric))
    if categorical:
        transformers.append(("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                      min_frequency=5, max_categories=50)),
        ]), categorical))
    prep = ColumnTransformer(transformers, remainder="drop")
    return Pipeline([("features", prep), ("model", model)])


def _metrics(model, x_test, y_test, classes):
    pred = model.predict(x_test)
    result = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision_weighted": float(precision_score(y_test, pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_test, pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_test, pred, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, pred, labels=classes).tolist(),
        "class_labels_encoded": [int(c) for c in classes],
    }
    try:
        prob = model.predict_proba(x_test)
        if len(classes) == 2:
            result["roc_auc"] = float(roc_auc_score(y_test, prob[:, 1]))
        else:
            result["roc_auc"] = float(roc_auc_score(y_test, prob, multi_class="ovr", average="weighted"))
    except (AttributeError, ValueError):
        result["roc_auc"] = None
    return result


def run_privacy_utility_experiment(before: pd.DataFrame, after: pd.DataFrame,
                                   target_column: str, seed: int = 42,
                                   max_rows: int = 20000, test_size: float = 0.2) -> dict:
    """Compare predictive utility on a shared stratified split.

    The target must remain unchanged by anonymization. High-cardinality categorical
    features are grouped by OneHotEncoder to keep memory use bounded.
    """
    if target_column not in before or target_column not in after:
        raise ValueError(f"Target column '{target_column}' was not found")
    if list(before.columns) != list(after.columns):
        raise ValueError("Before/after tables must have matching columns")
    target_before = before[target_column].astype("string")
    target_after = after[target_column].astype("string")
    valid = target_before.notna() & target_after.notna()
    if not target_before[valid].reset_index(drop=True).equals(target_after[valid].reset_index(drop=True)):
        raise ValueError("Target values changed during anonymization; choose an unchanged outcome column")
    valid_indices = np.flatnonzero(valid.to_numpy())
    labels = target_before.iloc[valid_indices].astype(str)
    encoder = LabelEncoder()
    y_all = encoder.fit_transform(labels)
    if len(encoder.classes_) < 2 or len(encoder.classes_) > 50:
        raise ValueError("Choose a categorical target with 2 to 50 classes")
    counts = np.bincount(y_all)
    if counts.min() < 5:
        raise ValueError("Each target class needs at least 5 non-null rows")
    indices = valid_indices
    if len(indices) > max_rows:
        indices, _, y_all, _ = train_test_split(
            indices, y_all, train_size=max_rows, random_state=seed, stratify=y_all
        )
    sampled_y = encoder.transform(target_before.iloc[indices].astype(str))
    train_idx, test_idx = train_test_split(
        indices, test_size=test_size, random_state=seed, stratify=sampled_y,
    )
    y = np.full(len(before), -1, dtype=int)
    y[valid_indices] = encoder.transform(target_before.iloc[valid_indices].astype(str))
    y_train, y_test = y[train_idx], y[test_idx]
    features = [c for c in before.columns if c != target_column]
    if not features:
        raise ValueError("No feature columns remain after removing the target")
    x_before = before.loc[:, features].copy()
    x_after = after.loc[:, features].copy()
    for frame in (x_before, x_after):
        for col in frame.columns:
            if not pd.api.types.is_numeric_dtype(frame[col]):
                frame[col] = frame[col].astype("object").where(frame[col].notna(), np.nan)
    factories = _model_factories(len(encoder.classes_))
    results = {}
    classes = np.unique(y_test)
    for name, factory in factories.items():
        version_results = {}
        for version, x in (("before", x_before), ("after", x_after)):
            try:
                model = _pipeline(factory(), x)
                model.fit(x.iloc[train_idx], y_train)
                version_results[version] = _metrics(model, x.iloc[test_idx], y_test, classes)
            except Exception as exc:
                version_results[version] = {"error": f"{type(exc).__name__}: {exc}"}
        if "error" not in version_results.get("before", {}) and "error" not in version_results.get("after", {}):
            version_results["delta_after_minus_before"] = {
                metric: (None if version_results["before"].get(metric) is None or version_results["after"].get(metric) is None
                         else version_results["after"][metric] - version_results["before"][metric])
                for metric in ("accuracy", "precision_weighted", "recall_weighted", "f1_weighted", "roc_auc")
            }
        results[name] = version_results
    for optional in ("XGBoost", "LightGBM"):
        if optional not in results:
            results[optional] = {"status": "skipped", "reason": "Optional package is not installed"}
    return {
        "target_column": target_column,
        "target_classes": [str(c) for c in encoder.classes_],
        "rows_used": int(len(indices)), "train_rows": int(len(train_idx)),
        "test_rows": int(len(test_idx)), "seed": seed, "test_size": test_size,
        "feature_columns": features,
        "models": results,
        "method": "Same stratified holdout split before/after; target excluded from features; categorical cardinality capped.",
    }
