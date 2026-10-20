"""Tests for reproducible synthetic records, input validation and temporal features."""
from __future__ import annotations

import pandas as pd
import pytest

from src.data_processing import (
    DataValidationError, features_for_candidate, generate_synthetic_records,
    learner_summary, load_csv, load_local_records, make_training_table,
    save_local_record, skill_summary, validate_records,
)


@pytest.fixture
def example() -> pd.DataFrame:
    return generate_synthetic_records(n_learners=8, seed=41)


def test_synthetic_reproducible_and_fictional() -> None:
    a = generate_synthetic_records(6, seed=19)
    b = generate_synthetic_records(6, seed=19)
    pd.testing.assert_frame_equal(a, b)
    assert a["learner_id"].str.startswith("SYN-").all()
    assert len(a) == 6 * 4 * 6
    assert 0 <= a["score"].min() <= a["score"].max() <= 1


def test_missing_required_columns(example: pd.DataFrame) -> None:
    with pytest.raises(DataValidationError, match="Missing required columns: learner_id"):
        validate_records(example.drop(columns=["learner_id"]))


@pytest.mark.parametrize("column,value", [
    ("score", 1.5), ("duration_minutes", -2),
    ("attempts", 0), ("attempts", 2.5), ("correct_attempts", -1),
    ("difficulty", "Impossible"), ("timestamp", "not-a-date"),
    ("learner_id", None),
])
def test_invalid_fields_have_actionable_errors(example: pd.DataFrame, column: str, value: object) -> None:
    bad = example.copy()
    bad[column] = bad[column].astype(object)
    bad.loc[0, column] = value
    with pytest.raises(DataValidationError, match="Invalid"):
        validate_records(bad)


def test_score_mismatch_rejected(example: pd.DataFrame) -> None:
    bad = example.copy()
    bad.loc[0, "score"] = 1.0 if float(bad.loc[0, "score"]) < 0.6 else 0.0
    with pytest.raises(DataValidationError, match="must agree"):
        validate_records(bad)


def test_optional_fields_derived_and_deduplicated(example: pd.DataFrame) -> None:
    no_optional = example.drop(columns=["score", "duration_minutes", "prerequisite_skill"])
    repeated = pd.concat([no_optional, no_optional.iloc[[0]]], ignore_index=True)
    clean = validate_records(repeated)
    assert len(clean) == len(example)
    assert clean["duration_minutes"].isna().all()
    assert clean["score"].between(0, 1).all()
    assert any("Removed 1" in msg for msg in clean.attrs["validation_warnings"])


def test_csv_loading_and_aggregation(example: pd.DataFrame, tmp_path) -> None:
    path = tmp_path / "records.csv"
    example.to_csv(path, index=False)
    data = load_csv(path)
    summary = learner_summary(data)
    selected = skill_summary(data, "SYN-001")
    assert len(summary) == 8
    assert len(selected) == 4
    assert selected["attempts"].sum() == data.loc[data["learner_id"] == "SYN-001", "attempts"].sum()
    assert selected["accuracy"].between(0, 1).all()


def test_temporal_features_use_only_preceding_records(example: pd.DataFrame) -> None:
    table = make_training_table(example)
    sample = table.loc[(table["learner_id"] == "SYN-001") & (table["skill"] == "Fractions")]
    assert len(sample) == 5
    assert sample["prior_sessions"].tolist() == [1, 2, 3, 4, 5]
    prior = example.loc[(example["learner_id"] == "SYN-001") & (example["skill"] == "Fractions")].sort_values("timestamp")
    assert sample.iloc[0]["prior_attempts"] == prior.iloc[0]["attempts"]
    assert sample.iloc[0]["last_score"] == pytest.approx(prior.iloc[0]["score"])
    assert "score" not in table.columns
    features = features_for_candidate(prior, "Fractions", "Beginner")
    assert features["prior_sessions"] == 6
    assert features["prior_attempts"] == int(prior["attempts"].sum())


def test_insufficient_sequential_sessions_fails(example: pd.DataFrame) -> None:
    first = example.groupby(["learner_id", "skill"], sort=False).head(1)
    with pytest.raises(DataValidationError, match="No sessions have prior"):
        make_training_table(first)


def test_local_sqlite_save_and_read_is_parameterized(example: pd.DataFrame, tmp_path) -> None:
    db = tmp_path / "local.db"
    row = example.iloc[0].to_dict()
    row["learner_id"] = "person'; DROP TABLE practice_updates; --"
    save_local_record(db, "context-A", row)
    loaded = load_local_records(db, "context-A", row["learner_id"])
    assert len(loaded) == 1
    assert loaded.iloc[0]["learner_id"] == row["learner_id"]
    assert load_local_records(db, "context-B", row["learner_id"]).empty


def test_invalid_csv_rejected(tmp_path) -> None:
    file = tmp_path / "malformed.csv"
    file.write_text("skill,score\nFractions,0.7\n", encoding="utf-8")
    with pytest.raises(DataValidationError, match="Missing required columns"):
        load_csv(file)


def test_blank_optional_duration_is_allowed(example: pd.DataFrame) -> None:
    sample = example.head(2).copy()
    sample["duration_minutes"] = ["", " "]
    clean = validate_records(sample)
    assert clean["duration_minutes"].isna().all()


def test_recent_accuracy_weighted_by_attempts(example: pd.DataFrame) -> None:
    one = example.loc[(example["learner_id"] == "SYN-001") & (example["skill"] == "Fractions")].sort_values("timestamp")
    features = features_for_candidate(one, "Fractions", "Beginner")
    tail = one.tail(3)
    expected = tail["correct_attempts"].sum() / tail["attempts"].sum()
    assert features["recent_accuracy"] == pytest.approx(expected)


def test_extra_columns_are_explicitly_ignored(example: pd.DataFrame) -> None:
    raw = example.head(2).copy()
    raw["unused_sensitive_field"] = ["not-for-model", "not-for-model"]
    clean = validate_records(raw)
    assert "unused_sensitive_field" not in clean
    assert any("Ignored 1 unsupported" in warning for warning in clean.attrs["validation_warnings"])


def test_ambiguous_same_skill_timestamps_fail_training(example: pd.DataFrame) -> None:
    first = example.iloc[[0]].copy()
    second = first.copy()
    second["activity_id"] = "DIFFERENT-ACTIVITY"
    ambiguous = pd.concat([example, second], ignore_index=True)
    with pytest.raises(DataValidationError, match="timestamp ties"):
        make_training_table(ambiguous)
