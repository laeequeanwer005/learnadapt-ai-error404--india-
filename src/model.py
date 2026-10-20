"""Reproducible scikit-learn demonstration model for next-session success.

Target: whether the next recorded practice session has score >= 0.60.
Training features refer ONLY to earlier same-learner/same-skill sessions.
Synthetic evaluation is not evidence of real-world educational efficacy.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data_processing import (
    CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES,
    DataValidationError, dataset_fingerprint, load_csv, make_training_table, validate_records,
)

TARGET_DESCRIPTION = "Session score >= 0.60, based on past practice sessions (not mastery)."
MODEL_FORMAT_VERSION = 2


class ModelError(ValueError):
    """Invalid training task, serialized model, or prediction input."""


def create_model(seed: int = 42) -> Pipeline:
    """Build a numeric/categorical preprocessing and logistic classifier pipeline."""
    transformer = ColumnTransformer(transformers=[
        ("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), list(NUMERIC_FEATURES)),
        ("categories", OneHotEncoder(handle_unknown="ignore"), list(CATEGORICAL_FEATURES)),
    ])
    return Pipeline([
        ("preprocess", transformer),
        ("classifier", LogisticRegression(max_iter=1000, random_state=seed)),
    ])


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, probabilities: np.ndarray | None) -> dict[str, Any]:
    """Calculate actual held-out binary classification metrics."""
    result: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).astype(int).tolist(),
        "roc_auc": None,
    }
    if probabilities is not None and len(np.unique(y_true)) == 2:
        result["roc_auc"] = float(roc_auc_score(y_true, probabilities))
    return result


def train_and_evaluate(
    records: pd.DataFrame, *, source_type: str, seed: int = 42, test_size: float = 0.25,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Train and test on disjoint LEARNERS to avoid same-learner leakage.

    The test set contains complete held-out learners; generated history
    features for those learners use only records preceding their target rows.
    Output is a model bundle and a report computed from actual predictions.
    """
    if source_type not in ("synthetic", "real"):
        raise ModelError("source_type must explicitly be 'synthetic' or 'real'.")
    if not (0.1 <= test_size <= 0.5):
        raise ModelError("test_size must be between 0.1 and 0.5.")
    data = validate_records(records)
    table = make_training_table(data)
    groups = table["learner_id"]
    if groups.nunique() < 6 or len(table) < 30:
        raise ModelError("Training needs at least 6 learners and 30 usable sequential sessions.")
    X = table.loc[:, list(FEATURE_COLUMNS)]
    y = table["target_success"].astype(int)
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    train_learners = set(groups.iloc[train_idx])
    test_learners = set(groups.iloc[test_idx])
    if not train_learners.isdisjoint(test_learners):
        raise ModelError("Internal split error: learner leakage.")
    if y_train.nunique() != 2:
        raise ModelError("Training split has only one target class; add representative sessions.")
    model = create_model(seed)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, list(model.classes_).index(1)]

    baseline = DummyClassifier(strategy="prior")
    baseline.fit(X_train, y_train)
    baseline_predictions = baseline.predict(X_test)
    baseline_probabilities = baseline.predict_proba(X_test)[:, list(baseline.classes_).index(1)]

    report: dict[str, Any] = {
        "target": TARGET_DESCRIPTION,
        "source_type": source_type,
        "data_label": "FICTIONAL SYNTHETIC DATA" if source_type == "synthetic" else "USER-SUPPLIED DATA (source/license not verified)",
        "dataset_fingerprint_prefix": dataset_fingerprint(data)[:12],
        "seed": seed,
        "test_size": test_size,
        "train_rows": len(train_idx), "test_rows": len(test_idx),
        "train_learners": len(train_learners), "test_learners": len(test_learners),
        "model": _metrics(y_test.to_numpy(), predictions, probabilities),
        "majority_prior_baseline": _metrics(y_test.to_numpy(), baseline_predictions, baseline_probabilities),
        "important_note": (
            "Metrics measured on a held-out learner group. These do not establish reliability "
            "on real students; probability calibration has NOT been evaluated."
        ),
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    bundle = {
        "format_version": MODEL_FORMAT_VERSION,
        "model": model,
        "training_source": source_type,
        "dataset_fingerprint": dataset_fingerprint(data),
        "feature_columns": list(FEATURE_COLUMNS),
        "target": TARGET_DESCRIPTION,
        "trained_at_utc": report["evaluated_at_utc"],
    }
    return bundle, report


def save_bundle(bundle: dict[str, Any], path: str | Path) -> Path:
    """Save a locally trained model. Never load model files from untrusted users."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, destination)
    return destination


def load_bundle(path: str | Path) -> dict[str, Any]:
    """Load only locally trusted model artifacts and check expected metadata."""
    destination = Path(path)
    if not destination.is_file():
        raise ModelError(f"Model artifact not available at {destination}.")
    try:
        bundle = joblib.load(destination)
    except Exception as exc:
        raise ModelError(f"Cannot load local model artifact: {type(exc).__name__}") from exc
    if (
        not isinstance(bundle, dict)
        or bundle.get("format_version") != MODEL_FORMAT_VERSION
        or bundle.get("feature_columns") != list(FEATURE_COLUMNS)
        or bundle.get("training_source") not in ("synthetic", "real")
        or not isinstance(bundle.get("dataset_fingerprint"), str)
        or len(bundle.get("dataset_fingerprint", "")) != 64
        or not isinstance(bundle.get("model"), Pipeline)
    ):
        raise ModelError("Model artifact is incompatible with the current features or format.")
    return bundle


def predict_probability(bundle: dict[str, Any], feature_data: pd.DataFrame) -> np.ndarray:
    """Return uncalibrated predicted likelihood of next-session success.

    Values are classifier probabilities, NOT confidence or validated mastery.
    No PII, IDs, target column, or scores from future sessions are accepted.
    """
    if not isinstance(bundle, dict) or not isinstance(bundle.get("model"), Pipeline):
        raise ModelError("No trained model bundle available.")
    if not isinstance(feature_data, pd.DataFrame):
        raise ModelError("Expected a DataFrame containing predictor fields.")
    missing = set(FEATURE_COLUMNS) - set(feature_data.columns)
    if missing:
        raise ModelError("Missing predictor features: " + ", ".join(sorted(missing)))
    if feature_data.empty:
        raise ModelError("Prediction input has zero rows.")
    clean = feature_data.loc[:, list(FEATURE_COLUMNS)].copy()
    for name in NUMERIC_FEATURES:
        clean[name] = pd.to_numeric(clean[name], errors="coerce")
        if clean[name].isna().any() or not bool(np.isfinite(clean[name]).all()):
            raise ModelError(f"Feature {name} must contain finite numbers.")
    for name in CATEGORICAL_FEATURES:
        if clean[name].isna().any() or clean[name].astype(str).str.strip().eq("").any():
            raise ModelError(f"Categorical feature {name} must be present.")
    if (clean["prior_sessions"] < 1).any() or (clean["prior_attempts"] < 1).any():
        raise ModelError("Insufficient prior evidence: do not predict for new skill histories.")
    for name in ("prior_accuracy", "recent_accuracy", "last_score"):
        if ((clean[name] < 0) | (clean[name] > 1)).any():
            raise ModelError(f"Feature {name} must be within 0 and 1.")
    model = bundle["model"]
    try:
        values = model.predict_proba(clean)[:, list(model.classes_).index(1)]
    except (ValueError, AttributeError, KeyError) as exc:
        raise ModelError(f"Prediction failed: {exc}") from exc
    if not bool(np.isfinite(values).all()) or not bool(((values >= 0) & (values <= 1)).all()):
        raise ModelError("Model returned invalid probability values.")
    return values


def _write_report(report: dict[str, Any], destination: str | Path) -> None:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def main() -> None:
    """CLI supports training+holdout evaluation and printing saved evaluations."""
    parser = argparse.ArgumentParser(description="LearnAdapt AI classifier training and evaluation")
    commands = parser.add_subparsers(dest="command", required=True)
    train_parser = commands.add_parser("train", help="Train and evaluate using disjoint learner groups")
    train_parser.add_argument("--data", required=True)
    train_parser.add_argument("--source", choices=["synthetic", "real"], required=True)
    train_parser.add_argument("--artifact", default="models/learnadapt_model.joblib")
    train_parser.add_argument("--report", default="models/evaluation.json")
    train_parser.add_argument("--seed", type=int, default=42)
    evaluate_parser = commands.add_parser("evaluate", help="Print ACTUAL saved held-out predictions report")
    evaluate_parser.add_argument("--report", default="models/evaluation.json")
    args = parser.parse_args()
    if args.command == "train":
        frame = load_csv(args.data)
        bundle, report = train_and_evaluate(frame, source_type=args.source, seed=args.seed)
        save_bundle(bundle, args.artifact)
        _write_report(report, args.report)
        print(f"Trained {args.source} model; saved artifact to {args.artifact}")
        print(json.dumps(report, indent=2))
    else:
        path = Path(args.report)
        if not path.exists():
            parser.error("No evaluation report exists. Run train first.")
        report = json.loads(path.read_text(encoding="utf-8"))
        print("Saved holdout evaluation from training (not a newly calculated independent test):")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
