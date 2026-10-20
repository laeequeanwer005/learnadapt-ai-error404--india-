"""Validation, synthetic development data, leakage-safe features, and optional SQLite.

A row represents a completed practice SESSION with a count of correct attempts.
The prediction feature table uses only rows that precede the target session.
"""
from __future__ import annotations

import argparse
import hashlib
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = (
    "learner_id", "skill", "activity_id", "difficulty",
    "attempts", "correct_attempts", "timestamp",
)
OPTIONAL_COLUMNS = ("score", "duration_minutes", "prerequisite_skill")
ALL_COLUMNS = (*REQUIRED_COLUMNS, *OPTIONAL_COLUMNS)
DIFFICULTIES = ("Beginner", "Intermediate", "Advanced")
FEATURE_COLUMNS = (
    "skill", "difficulty", "prior_sessions", "prior_attempts",
    "prior_accuracy", "recent_accuracy", "last_score",
)
NUMERIC_FEATURES = (
    "prior_sessions", "prior_attempts", "prior_accuracy",
    "recent_accuracy", "last_score",
)
CATEGORICAL_FEATURES = ("skill", "difficulty")


class DataValidationError(ValueError):
    """Input data cannot safely be used in the learning pipeline."""


def _fail_rows(mask: pd.Series, field: str, problem: str) -> None:
    """Raise a concise validation error with zero-based row indexes."""
    if bool(mask.any()):
        indices = list(mask[mask].index[:5])
        raise DataValidationError(
            f"Invalid {field}: {problem}. Example row indexes: {indices}. "
            f"Affected rows: {int(mask.sum())}. No records were silently removed."
        )


def validate_records(raw: pd.DataFrame) -> pd.DataFrame:
    """Return canonical records or reject malformed data with actionable errors.

    `score` is a fraction from 0 to 1, verified against the attempt counts.
    Identical duplicate rows are removed with an attached warning. Missing
    optional duration remains NaN; it is not silently reported as zero time.
    """
    if not isinstance(raw, pd.DataFrame):
        raise DataValidationError("Expected a pandas DataFrame.")
    missing = sorted(set(REQUIRED_COLUMNS).difference(raw.columns))
    if missing:
        raise DataValidationError("Missing required columns: " + ", ".join(missing))
    if raw.empty:
        raise DataValidationError("Dataset has no records.")

    frame = raw.loc[:, [c for c in ALL_COLUMNS if c in raw.columns]].copy()
    warnings: list[str] = []
    ignored_columns = [c for c in raw.columns if c not in ALL_COLUMNS]
    if ignored_columns:
        warnings.append(
            f"Ignored {len(ignored_columns)} unsupported CSV column(s). "
            "Only documented schema fields are used (extra columns are never model features)."
        )
    for column in ("learner_id", "skill", "activity_id", "difficulty"):
        frame[column] = frame[column].astype("string").str.strip()
        _fail_rows(frame[column].isna() | frame[column].eq(""), column, "missing text")

    frame["difficulty"] = frame["difficulty"].str.title()
    _fail_rows(~frame["difficulty"].isin(DIFFICULTIES), "difficulty", "use Beginner, Intermediate, or Advanced")

    for column in ("attempts", "correct_attempts"):
        original = frame[column]
        frame[column] = pd.to_numeric(original, errors="coerce")
        invalid = frame[column].isna() | ~np.isfinite(frame[column])
        _fail_rows(invalid, column, "must be a finite integer")
        _fail_rows((frame[column] % 1) != 0, column, "must be an integer")
        frame[column] = frame[column].astype("int64")
    _fail_rows(frame["attempts"] < 1, "attempts", "must be at least 1")
    _fail_rows(
        (frame["correct_attempts"] < 0) | (frame["correct_attempts"] > frame["attempts"]),
        "correct_attempts", "must be between zero and attempts",
    )

    if "score" in frame:
        submitted_score = pd.to_numeric(frame["score"], errors="coerce")
        nonempty = frame["score"].notna() & frame["score"].astype("string").str.strip().ne("")
        _fail_rows(nonempty & submitted_score.isna(), "score", "must be numeric from 0 to 1")
        _fail_rows(
            submitted_score.notna() & (
                ~np.isfinite(submitted_score) | (submitted_score < 0) | (submitted_score > 1)
            ), "score", "must be between zero and one",
        )
        expected = frame["correct_attempts"] / frame["attempts"]
        _fail_rows(
            submitted_score.notna() & (submitted_score - expected).abs().gt(0.011),
            "score", "must agree with correct_attempts / attempts (within 0.011)",
        )
    else:
        submitted_score = pd.Series(np.nan, index=frame.index)
    frame["score"] = frame["correct_attempts"] / frame["attempts"]
    if submitted_score.isna().any():
        warnings.append("Missing scores were derived from correct_attempts / attempts.")

    if "duration_minutes" not in frame:
        frame["duration_minutes"] = np.nan
    else:
        original_duration = frame["duration_minutes"].copy()
        absent = original_duration.isna() | original_duration.astype("string").str.strip().eq("")
        frame["duration_minutes"] = pd.to_numeric(original_duration, errors="coerce")
        _fail_rows(
            ~absent & frame["duration_minutes"].isna(),
            "duration_minutes", "must be numeric or blank",
        )
        _fail_rows(
            frame["duration_minutes"].notna() & (
                ~np.isfinite(frame["duration_minutes"]) | (frame["duration_minutes"] <= 0)
            ), "duration_minutes", "must be a positive number when supplied",
        )
    if "prerequisite_skill" not in frame:
        frame["prerequisite_skill"] = pd.NA
    else:
        frame["prerequisite_skill"] = (
            frame["prerequisite_skill"].astype("string").str.strip().replace("", pd.NA)
        )

    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce", format="mixed")
    _fail_rows(frame["timestamp"].isna(), "timestamp", "must be a parseable date/time")

    before = len(frame)
    frame = frame.drop_duplicates(subset=list(ALL_COLUMNS), keep="first")
    if len(frame) != before:
        warnings.append(f"Removed {before - len(frame)} exactly identical duplicate records.")
    frame = frame.sort_values(["timestamp", "learner_id", "skill"], kind="stable").reset_index(drop=True)
    frame.attrs["validation_warnings"] = warnings
    return frame.loc[:, list(ALL_COLUMNS)]


def load_csv(path_or_file: str | Path | Any) -> pd.DataFrame:
    """Read and validate local CSV paths or Streamlit UploadedFile objects."""
    try:
        raw = pd.read_csv(path_or_file)
    except (OSError, UnicodeError, pd.errors.ParserError, ValueError) as exc:
        raise DataValidationError(f"Cannot read CSV: {exc}") from exc
    return validate_records(raw)


def generate_synthetic_records(n_learners: int = 60, seed: int = 42) -> pd.DataFrame:
    """Create reproducible fictional practice sessions; NEVER real learner data."""
    if n_learners < 4:
        raise ValueError("Synthetic training demo requires at least four fictional learners.")
    rng = np.random.default_rng(seed)
    skills = ("Foundations", "Fractions", "Algebra", "Graphs")
    prereqs = {"Foundations": None, "Fractions": "Foundations", "Algebra": "Fractions", "Graphs": "Algebra"}
    start = pd.Timestamp("2026-09-01T08:00:00Z")
    records: list[dict[str, Any]] = []
    for learner_no in range(1, n_learners + 1):
        skill_ability = rng.normal(0.0, 0.75, size=len(skills))
        for round_no in range(6):
            for skill_idx, skill in enumerate(skills):
                difficulty = DIFFICULTIES[(round_no + skill_idx // 2) % 3]
                attempts = int(rng.integers(3, 9))
                difficulty_effect = {"Beginner": 0.65, "Intermediate": 0.0, "Advanced": -0.55}[difficulty]
                logit = -0.25 + skill_ability[skill_idx] + 0.19 * round_no + difficulty_effect
                probability = 1 / (1 + np.exp(-logit))
                correct = int(rng.binomial(attempts, probability))
                timestamp = start + pd.Timedelta(days=round_no * 5 + skill_idx, hours=learner_no % 16)
                records.append({
                    "learner_id": f"SYN-{learner_no:03d}",
                    "skill": skill,
                    "activity_id": f"DEMO-{skill.upper()}-{difficulty.upper()}",
                    "difficulty": difficulty,
                    "attempts": attempts,
                    "correct_attempts": correct,
                    "score": correct / attempts,
                    "duration_minutes": round(float(rng.uniform(5, 20)), 2),
                    "timestamp": timestamp.isoformat(),
                    "prerequisite_skill": prereqs[skill],
                })
    return validate_records(pd.DataFrame.from_records(records))


def learner_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Group weighted accuracy and practice counts by fictional/opaque ID."""
    if frame.empty:
        return pd.DataFrame(columns=["learner_id", "sessions", "attempts", "correct_attempts", "accuracy"])
    grouped = frame.groupby("learner_id", as_index=False, observed=True).agg(
        sessions=("score", "size"), attempts=("attempts", "sum"),
        correct_attempts=("correct_attempts", "sum"),
    )
    grouped["accuracy"] = grouped["correct_attempts"] / grouped["attempts"]
    return grouped


def skill_summary(frame: pd.DataFrame, learner_id: str) -> pd.DataFrame:
    """Compute observed performance (not a mastery score) per skill."""
    subset = frame.loc[frame["learner_id"] == learner_id].sort_values("timestamp")
    columns = ["skill", "sessions", "attempts", "correct_attempts", "accuracy", "recent_accuracy"]
    if subset.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    for skill, group in subset.groupby("skill", sort=True):
        recent = group.tail(3)
        rows.append({
            "skill": str(skill), "sessions": len(group),
            "attempts": int(group["attempts"].sum()),
            "correct_attempts": int(group["correct_attempts"].sum()),
            "accuracy": float(group["correct_attempts"].sum() / group["attempts"].sum()),
            "recent_accuracy": float(recent["correct_attempts"].sum() / recent["attempts"].sum()),
        })
    return pd.DataFrame(rows, columns=columns)


def make_training_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Predict CURRENT session success using only PREVIOUS same-skill sessions.

    Chronologically this is a next-session prediction from the preceding point.
    Groups are learner+skill. Rows without at least one earlier session are
    excluded from training but retained for descriptive analytics.
    """
    data = validate_records(frame)
    tied = data.duplicated(subset=["learner_id", "skill", "timestamp"], keep=False)
    if tied.any():
        raise DataValidationError(
            f"Found {int(tied.sum())} same-learner/same-skill timestamp ties. "
            "Provide unambiguous chronological session timestamps before training."
        )
    data = data.sort_values(["learner_id", "skill", "timestamp"], kind="stable").reset_index(drop=True)
    by_history = data.groupby(["learner_id", "skill"], sort=False)
    data["prior_sessions"] = by_history.cumcount()
    data["prior_attempts"] = by_history["attempts"].cumsum() - data["attempts"]
    previous_correct = by_history["correct_attempts"].cumsum() - data["correct_attempts"]
    data["prior_accuracy"] = np.where(
        data["prior_attempts"] > 0, previous_correct / data["prior_attempts"].clip(lower=1), 0.5,
    )
    recent_correct = by_history["correct_attempts"].transform(
        lambda counts: counts.shift(1).rolling(3, min_periods=1).sum()
    )
    recent_attempts = by_history["attempts"].transform(
        lambda counts: counts.shift(1).rolling(3, min_periods=1).sum()
    )
    data["recent_accuracy"] = (recent_correct / recent_attempts).fillna(0.5)
    data["last_score"] = by_history["score"].shift(1).fillna(0.5)
    data["target_success"] = (data["score"] >= 0.6).astype("int64")
    training = data.loc[data["prior_sessions"] > 0].copy().reset_index(drop=True)
    if training.empty:
        raise DataValidationError("No sessions have prior same-skill history; cannot build next-session training rows.")
    return training.loc[:, ["learner_id", *FEATURE_COLUMNS, "target_success"]]


def features_for_candidate(history: pd.DataFrame, skill: str, difficulty: str) -> dict[str, Any]:
    """Compute a model input from observed, already completed sessions."""
    if difficulty not in DIFFICULTIES:
        raise DataValidationError(f"Unknown difficulty: {difficulty}")
    subset = history.loc[history["skill"] == skill].sort_values("timestamp")
    if subset.empty:
        return {
            "skill": skill, "difficulty": difficulty,
            "prior_sessions": 0, "prior_attempts": 0,
            "prior_accuracy": 0.5, "recent_accuracy": 0.5, "last_score": 0.5,
        }
    recent = subset.tail(3)
    return {
        "skill": skill, "difficulty": difficulty,
        "prior_sessions": len(subset), "prior_attempts": int(subset["attempts"].sum()),
        "prior_accuracy": float(subset["correct_attempts"].sum() / subset["attempts"].sum()),
        "recent_accuracy": float(recent["correct_attempts"].sum() / recent["attempts"].sum()),
        "last_score": float(subset.iloc[-1]["score"]),
    }


def dataset_fingerprint(records: pd.DataFrame) -> str:
    """SHA-256 fingerprint of the canonical session table (including provenance).

    Used only to match a local trained model to the uploaded/base dataset it
    was trained on. A hash is not a verification of consent or data licensing.
    """
    canonical = validate_records(records)
    signature = canonical.to_csv(index=False, date_format="%Y-%m-%dT%H:%M:%S.%fZ", na_rep="")
    return hashlib.sha256(signature.encode("utf-8")).hexdigest()


def _connect(database_path: str | Path) -> sqlite3.Connection:
    """Initialize local-only SQLite without persisting any imported CSV rows."""
    path = Path(database_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=5)
    conn.execute("""CREATE TABLE IF NOT EXISTS practice_updates (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        context_key TEXT NOT NULL,
        learner_id TEXT NOT NULL,
        skill TEXT NOT NULL,
        activity_id TEXT NOT NULL,
        difficulty TEXT NOT NULL,
        attempts INTEGER NOT NULL,
        correct_attempts INTEGER NOT NULL,
        score REAL NOT NULL,
        duration_minutes REAL,
        timestamp TEXT NOT NULL,
        prerequisite_skill TEXT
    )""")
    return conn


def save_local_record(database_path: str | Path, context_key: str, row: dict[str, Any]) -> None:
    """Validate and insert one user-entered record via a parameterized query."""
    canonical = validate_records(pd.DataFrame([row]))
    record = canonical.iloc[0]
    with _connect(database_path) as connection:
        connection.execute("""
            INSERT INTO practice_updates
              (context_key, learner_id, skill, activity_id, difficulty,
               attempts, correct_attempts, score, duration_minutes, timestamp, prerequisite_skill)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            context_key, str(record["learner_id"]), str(record["skill"]), str(record["activity_id"]),
            str(record["difficulty"]), int(record["attempts"]), int(record["correct_attempts"]),
            float(record["score"]),
            None if pd.isna(record["duration_minutes"]) else float(record["duration_minutes"]),
            record["timestamp"].isoformat(),
            None if pd.isna(record["prerequisite_skill"]) else str(record["prerequisite_skill"]),
        ))


def load_local_records(database_path: str | Path, context_key: str, learner_id: str) -> pd.DataFrame:
    """Load only updates for the explicitly selected context and learner."""
    if not Path(database_path).exists():
        return pd.DataFrame(columns=ALL_COLUMNS)
    with _connect(database_path) as connection:
        rows = connection.execute("""
            SELECT learner_id, skill, activity_id, difficulty, attempts, correct_attempts,
                   score, duration_minutes, timestamp, prerequisite_skill
            FROM practice_updates WHERE context_key = ? AND learner_id = ? ORDER BY id
        """, (context_key, learner_id)).fetchall()
    return pd.DataFrame(rows, columns=ALL_COLUMNS)


def main() -> None:
    """CLI for generating an explicitly labelled synthetic CSV."""
    parser = argparse.ArgumentParser(description="LearnAdapt AI synthetic data generator (fictional records only)")
    parser.add_argument("command", choices=["generate-demo"])
    parser.add_argument("--output", default="data/synthetic_demo.csv")
    parser.add_argument("--learners", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    arguments = parser.parse_args()
    frame = generate_synthetic_records(n_learners=arguments.learners, seed=arguments.seed)
    output = Path(arguments.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print(f"SYNTHETIC DEMO ONLY: wrote {len(frame)} fictional practice sessions to {output}")


if __name__ == "__main__":
    main()
