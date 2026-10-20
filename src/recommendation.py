"""Evidence-linked, prerequisite-safe activity ranking for LearnAdapt AI.

The catalogue defines practice IDEAS, not verified external learning links.
No recommendation establishes learner ability or educational outcomes.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from src.data_processing import ALL_COLUMNS, features_for_candidate, validate_records
from src.model import ModelError, predict_probability

MIN_GATE_ATTEMPTS = 8
GATE_ACCURACY = 0.70


@dataclass(frozen=True)
class Activity:
    """One locally defined activity suggestion, not hosted learning content."""
    activity_id: str
    title: str
    skill: str
    difficulty: str
    estimated_minutes: int
    prerequisite_skill: str | None = None


ACTIVITY_CATALOGUE: tuple[Activity, ...] = (
    Activity("FND-DIAG", "Try a short number-sense diagnostic", "Foundations", "Beginner", 8),
    Activity("FND-101", "Practice number fluency", "Foundations", "Beginner", 10),
    Activity("FRA-101", "Explore fraction basics", "Fractions", "Beginner", 10, "Foundations"),
    Activity("FRA-102", "Practice fraction fluency", "Fractions", "Beginner", 12, "Foundations"),
    Activity("FRA-201", "Practice equivalent fractions", "Fractions", "Intermediate", 12, "Foundations"),
    Activity("ALG-101", "Practice introductory algebra", "Algebra", "Beginner", 12, "Fractions"),
    Activity("ALG-201", "Solve simple linear equations", "Algebra", "Intermediate", 15, "Fractions"),
    Activity("GRA-101", "Learn to read graphs", "Graphs", "Beginner", 10, "Algebra"),
    Activity("GRA-201", "Interpret trends in graphs", "Graphs", "Intermediate", 15, "Algebra"),
)


def _observed(history: pd.DataFrame, skill: str) -> dict[str, float | int]:
    subset = history.loc[history["skill"] == skill].sort_values("timestamp")
    if subset.empty:
        return {"sessions": 0, "attempts": 0, "accuracy": 0.0, "recent_accuracy": 0.0}
    recent = subset.tail(3)
    return {
        "sessions": len(subset),
        "attempts": int(subset["attempts"].sum()),
        "accuracy": float(subset["correct_attempts"].sum() / subset["attempts"].sum()),
        "recent_accuracy": float(recent["correct_attempts"].sum() / recent["attempts"].sum()),
    }


def _direct_requirements(skill: str, history: pd.DataFrame) -> set[str]:
    """Get every explicitly declared direct prerequisite for a skill."""
    requirements = {
        item.prerequisite_skill for item in ACTIVITY_CATALOGUE
        if item.skill == skill and item.prerequisite_skill
    }
    if not history.empty and "prerequisite_skill" in history:
        relevant = history.loc[history["skill"] == skill, "prerequisite_skill"].dropna()
        requirements.update(str(value) for value in relevant if str(value).strip())
    return requirements


def _requirements(activity: Activity, history: pd.DataFrame) -> list[str]:
    """Return transitive prerequisite closure; cycles remain conservatively blocked."""
    required: set[str] = set()
    pending = list(_direct_requirements(activity.skill, history))
    while pending:
        skill = pending.pop()
        if skill in required:
            continue
        required.add(skill)
        pending.extend(_direct_requirements(skill, history) - required)
    return sorted(required)


def prerequisite_check(activity: Activity, history: pd.DataFrame) -> tuple[bool, list[str]]:
    """Never unlock an explicit prerequisite without sufficient observed evidence."""
    blocked: list[str] = []
    for skill in _requirements(activity, history):
        info = _observed(history, skill)
        if info["attempts"] < MIN_GATE_ATTEMPTS or info["accuracy"] < GATE_ACCURACY:
            blocked.append(skill)
    return not blocked, blocked


def _cold_start_result() -> dict[str, Any]:
    activity = ACTIVITY_CATALOGUE[0]
    return {
        "activity_id": activity.activity_id,
        "activity_title": activity.title,
        "skill": activity.skill,
        "difficulty": activity.difficulty,
        "estimated_minutes": activity.estimated_minutes,
        "reason": (
            "No prior practice is available. Start with a short optional diagnostic, "
            "then update the recommendation from actual practice evidence."
        ),
        "method": "diagnostic_cold_start",
        "confidence": None,
        "estimated_success_probability": None,
        "probability_note": None,
        "evidence_level": "none",
        "prerequisites_satisfied": True,
        "blocked_skills": [],
    }


def recommend_activity(
    history: pd.DataFrame,
    *,
    model_bundle: dict[str, Any] | None = None,
    source_type: str = "synthetic",
) -> dict[str, Any]:
    """Choose an eligible activity with an evidence-linked explanation.

    With a compatible model: model-assisted ranking of candidates with some
    history; returned p is explicitly UNCALIBRATED. Without one: deterministic
    evidence-gap ranking. Cold start: a diagnostic recommendation, not a guess.
    """
    if source_type not in ("synthetic", "real"):
        raise ValueError("source_type must be synthetic or real.")
    if history is None or history.empty:
        return _cold_start_result()
    frame = validate_records(history)

    model_available = (
        model_bundle is not None
        and model_bundle.get("training_source") == source_type
    )
    scored: list[tuple[float, Activity, dict[str, float | int], float | None, list[str]]] = []
    locked_skills: set[str] = set()
    for activity in ACTIVITY_CATALOGUE:
        if activity.activity_id == "FND-DIAG":
            continue  # only an intentional new-learner diagnostic
        allowed, blocked = prerequisite_check(activity, frame)
        if not allowed:
            locked_skills.update(blocked)
            continue
        info = _observed(frame, activity.skill)
        sessions = int(info["sessions"])
        if sessions:
            gap = 1.0 - float(info["accuracy"])
            recent_gap = 1.0 - float(info["recent_accuracy"])
            evidence_priority = 0.65 * gap + 0.35 * recent_gap
            evidence_priority += 0.09 if float(info["recent_accuracy"]) < 0.6 else 0
        else:
            evidence_priority = 0.35  # gentle exploration, never a mastery claim

        if sessions == 0 or float(info["accuracy"]) < 0.65:
            preferred_difficulty = "Beginner"
        else:
            preferred_difficulty = "Intermediate"
        if activity.difficulty == preferred_difficulty:
            evidence_priority += 0.07
        else:
            evidence_priority -= 0.08
        # Reduce repetition of an identical practice activity if it was just logged.
        if not frame.empty and str(frame.iloc[-1]["activity_id"]) == activity.activity_id:
            evidence_priority -= 0.08

        probability: float | None = None
        ranking = evidence_priority
        if model_available and sessions >= 1 and int(info["attempts"]) >= 3:
            try:
                features = pd.DataFrame([features_for_candidate(frame, activity.skill, activity.difficulty)])
                probability = float(predict_probability(model_bundle, features)[0])
                # Near ~0.7 is used only as a heuristic challenge target, not a
                # validated pedagogical cutoff or calibrated probability.
                ranking += 0.18 * (1.0 - abs(probability - 0.70))
            except (ModelError, ValueError, TypeError):
                probability = None
        scored.append((ranking, activity, info, probability, blocked))

    if not scored:
        # Catalogue always contains a no-prerequisite foundation activity.
        return _cold_start_result()
    scored.sort(key=lambda result: (-result[0], result[1].activity_id))
    _, activity, info, probability, _ = scored[0]
    sessions = int(info["sessions"])
    if sessions:
        explanation = (
            f"In {sessions} recorded {activity.skill} sessions, "
            f"{int(round(float(info['accuracy']) * 100))}% of {info['attempts']} attempts were correct; "
            f"the most recent three sessions show {int(round(float(info['recent_accuracy']) * 100))}% correct. "
            "This observed history informed the practice ranking; it does not determine mastery."
        )
    else:
        explanation = (
            f"There are no recorded {activity.skill} sessions yet. This introductory activity "
            "offers a cautious way to gather evidence."
        )
    required = _requirements(activity, frame)
    if required:
        explanation += " Observed prerequisites passed: " + ", ".join(required) + "."
    if sessions < 3:
        explanation += " Evidence is limited; review or override this suggestion."
    if locked_skills:
        explanation += " Other advanced options remain restricted until their prerequisites have enough evidence."
    method = "model_assisted" if probability is not None else "rule_based_fallback"
    if probability is not None:
        explanation += (
            " An uncalibrated classifier probability also influenced the rank. "
            "This is NOT a confidence score or an educational outcome guarantee."
        )
    return {
        "activity_id": activity.activity_id,
        "activity_title": activity.title,
        "skill": activity.skill,
        "difficulty": activity.difficulty,
        "estimated_minutes": activity.estimated_minutes,
        "reason": explanation,
        "method": method,
        "confidence": None,
        "estimated_success_probability": probability,
        "probability_note": (
            "Uncalibrated estimate of score >= 60% on the NEXT session, derived from training data."
            if probability is not None else None
        ),
        "evidence_level": "limited" if sessions < 3 else "observational",
        "prerequisites_satisfied": True,
        "blocked_skills": sorted(locked_skills),
    }


def catalogue_as_records() -> list[dict[str, Any]]:
    """Export the exact activities supported in Practice Update forms."""
    return [asdict(activity) for activity in ACTIVITY_CATALOGUE]


def empty_history() -> pd.DataFrame:
    """Prepare a schema-compatible empty history for cold-start learners."""
    return pd.DataFrame(columns=list(ALL_COLUMNS))
