"""Cold-start, eligibility constraints, output schema and model-gated suggestions."""
from __future__ import annotations

import pandas as pd

from src.data_processing import generate_synthetic_records, validate_records
from src.model import train_and_evaluate
from src.recommendation import (
    ACTIVITY_CATALOGUE, empty_history, prerequisite_check, recommend_activity,
)


def _history(skill: str, attempts: int, correct: int, prerequisite: str | None = None) -> pd.DataFrame:
    return pd.DataFrame([{
        "learner_id": "TEST-1", "skill": skill, "activity_id": "LOCAL",
        "difficulty": "Beginner", "attempts": attempts,
        "correct_attempts": correct, "score": correct / attempts,
        "duration_minutes": 10, "timestamp": "2026-10-01T12:00:00Z",
        "prerequisite_skill": prerequisite,
    }])


def test_cold_start_is_diagnostic_and_does_not_invent_confidence() -> None:
    result = recommend_activity(empty_history())
    assert result["method"] == "diagnostic_cold_start"
    assert result["confidence"] is None
    assert result["estimated_success_probability"] is None
    assert "diagnostic" in result["activity_title"].lower()


def test_prerequisites_are_not_bypassed() -> None:
    activity = next(a for a in ACTIVITY_CATALOGUE if a.activity_id == "ALG-101")
    allowed, blocked = prerequisite_check(activity, validate_records(_history("Fractions", 6, 6)))
    assert not allowed and "Fractions" in blocked  # too few attempts
    allowed, blocked = prerequisite_check(activity, validate_records(_history("Fractions", 10, 5)))
    assert not allowed and "Fractions" in blocked  # low observed fraction accuracy
    evidence = validate_records(pd.concat([
        _history("Fractions", 10, 8), _history("Foundations", 10, 8),
    ], ignore_index=True))
    allowed, blocked = prerequisite_check(activity, evidence)
    assert allowed and not blocked  # both direct and transitive prerequisites met


def test_unknown_declared_prerequisite_is_conservative() -> None:
    activity = next(a for a in ACTIVITY_CATALOGUE if a.activity_id == "ALG-101")
    row = _history("Algebra", 10, 10, prerequisite="UnlistedSkill")
    history = validate_records(row)
    ok, blocked = prerequisite_check(activity, history)
    assert not ok
    assert "UnlistedSkill" in blocked


def test_fallback_recommendation_structured_and_deterministic() -> None:
    history = validate_records(pd.concat([
        _history("Foundations", 10, 9),
        _history("Fractions", 10, 2),
    ], ignore_index=True))
    first = recommend_activity(history, model_bundle=None)
    second = recommend_activity(history, model_bundle=None)
    assert first == second
    required = {
        "activity_id", "activity_title", "skill", "difficulty", "estimated_minutes",
        "reason", "method", "confidence", "estimated_success_probability",
        "evidence_level", "prerequisites_satisfied", "blocked_skills",
    }
    assert required.issubset(first)
    assert first["method"] == "rule_based_fallback"
    assert first["confidence"] is None
    assert first["estimated_success_probability"] is None
    assert first["skill"] != "Algebra"
    assert first["prerequisites_satisfied"] is True
    assert "recorded" in first["reason"]


def test_incompatible_synthetic_model_cannot_score_real_data() -> None:
    fictional = generate_synthetic_records(n_learners=12, seed=8)
    bundle, _ = train_and_evaluate(fictional, source_type="synthetic")
    history = fictional.loc[fictional["learner_id"] == "SYN-001"]
    result = recommend_activity(history, model_bundle=bundle, source_type="real")
    assert result["method"] == "rule_based_fallback"
    assert result["estimated_success_probability"] is None


def test_model_assisted_when_compatible_and_evidence_exists() -> None:
    fictional = generate_synthetic_records(n_learners=12, seed=8)
    bundle, _ = train_and_evaluate(fictional, source_type="synthetic")
    history = fictional.loc[fictional["learner_id"] == "SYN-001"]
    result = recommend_activity(history, model_bundle=bundle, source_type="synthetic")
    assert result["method"] == "model_assisted"
    assert 0 <= result["estimated_success_probability"] <= 1
    assert result["confidence"] is None
    assert "Uncalibrated" in str(result["probability_note"])


def test_weak_fraction_session_changes_evidence_linked_reasons() -> None:
    baseline = validate_records(pd.concat([
        _history("Foundations", 10, 10),
        _history("Fractions", 10, 7),
    ], ignore_index=True))
    extra = _history("Fractions", 5, 2)
    extra.loc[0, "timestamp"] = "2026-10-02T12:00:00Z"
    updated = validate_records(pd.concat([baseline, extra], ignore_index=True))
    before = recommend_activity(baseline)
    after = recommend_activity(updated)
    assert before != after
    assert after["method"] == "rule_based_fallback"
    assert "Fractions" in after["reason"] or after["skill"] == "Foundations"


def test_transitive_prerequisite_gate_is_not_bypassed() -> None:
    algebra = next(a for a in ACTIVITY_CATALOGUE if a.activity_id == "ALG-101")
    history = validate_records(pd.concat([
        _history("Foundations", 10, 2),
        _history("Fractions", 10, 10),
    ], ignore_index=True))
    allowed, blocked = prerequisite_check(algebra, history)
    assert not allowed
    assert "Foundations" in blocked
    assert "Fractions" not in blocked
