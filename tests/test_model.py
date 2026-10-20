"""Model training, reproducibility, serialization, and inference safety tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_processing import FEATURE_COLUMNS, dataset_fingerprint, generate_synthetic_records, make_training_table
from src.model import ModelError, load_bundle, predict_probability, save_bundle, train_and_evaluate


@pytest.fixture(scope="module")
def training_records() -> pd.DataFrame:
    return generate_synthetic_records(n_learners=12, seed=15)


def test_training_evaluates_actual_held_out_predictions(training_records: pd.DataFrame) -> None:
    bundle, report = train_and_evaluate(training_records, source_type="synthetic", seed=10)
    assert report["train_rows"] > 0
    assert report["test_rows"] > 0
    assert report["train_learners"] + report["test_learners"] == 12
    assert report["model"]["confusion_matrix"] and len(report["model"]["confusion_matrix"]) == 2
    assert 0 <= report["model"]["accuracy"] <= 1
    assert 0 <= report["majority_prior_baseline"]["accuracy"] <= 1
    assert bundle["training_source"] == "synthetic"


def test_predictions_are_valid_and_repeatable(training_records: pd.DataFrame) -> None:
    bundle_1, report_1 = train_and_evaluate(training_records, source_type="synthetic", seed=42)
    bundle_2, report_2 = train_and_evaluate(training_records, source_type="synthetic", seed=42)
    table = make_training_table(training_records)
    input_table = table.loc[:, list(FEATURE_COLUMNS)].iloc[:10]
    p1 = predict_probability(bundle_1, input_table)
    p2 = predict_probability(bundle_2, input_table)
    assert p1.shape == (10,)
    assert np.all((p1 >= 0) & (p1 <= 1))
    np.testing.assert_allclose(p1, p2, rtol=0, atol=1e-10)
    assert report_1["model"] == report_2["model"]


def test_model_bundle_roundtrip(training_records: pd.DataFrame, tmp_path) -> None:
    bundle, _ = train_and_evaluate(training_records, source_type="synthetic")
    path = tmp_path / "safe_local_model.joblib"
    save_bundle(bundle, path)
    loaded = load_bundle(path)
    X = make_training_table(training_records).loc[:, list(FEATURE_COLUMNS)].head(2)
    np.testing.assert_allclose(predict_probability(bundle, X), predict_probability(loaded, X))


def test_missing_or_corrupt_artifact(tmp_path) -> None:
    with pytest.raises(ModelError, match="not available"):
        load_bundle(tmp_path / "missing.joblib")
    bad = tmp_path / "wrong.joblib"
    bad.write_text("not a model", encoding="utf-8")
    with pytest.raises(ModelError, match="Cannot load"):
        load_bundle(bad)


def test_model_rejects_missing_features_and_no_history(training_records: pd.DataFrame) -> None:
    bundle, _ = train_and_evaluate(training_records, source_type="synthetic")
    input_row = make_training_table(training_records).loc[:, list(FEATURE_COLUMNS)].head(1).copy()
    with pytest.raises(ModelError, match="Missing predictor features"):
        predict_probability(bundle, input_row.drop(columns=["skill"]))
    input_row["prior_sessions"] = 0
    with pytest.raises(ModelError, match="Insufficient prior evidence"):
        predict_probability(bundle, input_row)


def test_rejects_insufficient_learners() -> None:
    frame = generate_synthetic_records(n_learners=4)
    with pytest.raises(ModelError, match="at least 6 learners"):
        train_and_evaluate(frame, source_type="synthetic")


def test_rejects_inaccurate_source(training_records: pd.DataFrame) -> None:
    with pytest.raises(ModelError, match="source_type"):
        train_and_evaluate(training_records, source_type="unknown")


def test_model_tracks_canonical_dataset_fingerprint(training_records: pd.DataFrame) -> None:
    bundle, report = train_and_evaluate(training_records, source_type="synthetic")
    assert bundle["dataset_fingerprint"] == dataset_fingerprint(training_records)
    assert report["dataset_fingerprint_prefix"] == bundle["dataset_fingerprint"][:12]
    changed = training_records.copy()
    changed.loc[0, "correct_attempts"] = (int(changed.loc[0, "correct_attempts"]) + 1) % (int(changed.loc[0, "attempts"]) + 1)
    changed.loc[0, "score"] = changed.loc[0, "correct_attempts"] / changed.loc[0, "attempts"]
    assert dataset_fingerprint(changed) != bundle["dataset_fingerprint"]
