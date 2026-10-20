"""LearnAdapt AI | TEAM ERROR-404 | Streamlit Community Cloud-ready frontend.

The app works without a model artifact: all unavailable model results are
explicitly replaced with documented rule-based suggestions, not fake numbers.
"""
from __future__ import annotations

import base64
import hashlib
import io
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

from src.data_processing import (
    ALL_COLUMNS, DataValidationError, dataset_fingerprint, generate_synthetic_records,
    load_csv, load_local_records, save_local_record, skill_summary, validate_records,
)
from src.model import ModelError, load_bundle, train_and_evaluate
from src.recommendation import ACTIVITY_CATALOGUE, empty_history, recommend_activity

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "learnadapt_model.joblib"
DATABASE_PATH = ROOT / ".local" / "practice_updates.db"
LOCAL_SQLITE_ENABLED = os.getenv("LEARNADAPT_ENABLE_SQLITE", "0") == "1"

st.set_page_config(
    page_title="LearnAdapt AI | TEAM ERROR-404",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
:root {--blue:#1D4ED8;--dark:#142340;--muted:#52647C;--mint:#14B8A6;}
html, body, [class*=st-] {font-family:Inter,system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;}
[data-testid="stAppViewContainer"] {background:#F6F9FE;color:#142340;}
[data-testid="stSidebar"] {background:#ECF3FF;}
.block-container {max-width:1240px;padding-top:1.4rem;}
h1,h2,h3,h4 {letter-spacing:-.035em;color:#142340;}
p,li {line-height:1.6;}
.hero-card {background:linear-gradient(110deg,#122854 0%,#1D4ED8 68%,#2563EB 100%);
 padding:26px 32px;border-radius:23px;color:#F6F9FF;margin-bottom:20px;box-shadow:0 18px 36px #20407820;}
.hero-card img {max-width:320px;width:75%;height:auto;object-fit:contain;}
.hero-card h1 {color:#fff;font-size:clamp(2rem,3.5vw,3rem);margin:8px 0 4px;}
.hero-card p {color:#E5EEFF;margin:6px 0 0;max-width:780px;}
.hero-kicker {font-size:12px;letter-spacing:.22em;font-weight:700;color:#A9DFFF;}
.source-pill {display:inline-block;padding:7px 13px;border-radius:999px;
 background:#DBEAFE;color:#18408B;font-size:.84rem;font-weight:700;margin:2px 0 12px;}
[data-testid="stMetric"] {background:#fff;border:1px solid #DFEAF9;border-radius:16px;
 padding:18px;box-shadow:0 6px 20px #1B3A6510;}
[data-testid="stMetricLabel"] {color:#425777;}
[data-testid="stMetricValue"] {color:#173979;}
div[data-testid="stVerticalBlockBorderWrapper"]:has(>div>div[data-testid="stVerticalBlock"]) {
 border-color:#DDE9F6;}
.stButton>button[kind="primary"], .stFormSubmitButton>button[kind="primary"] {
 background:#1D4ED8;color:white;border-radius:10px;border:0;}
.stButton>button, .stDownloadButton>button, .stFormSubmitButton>button {
 border-radius:10px;font-weight:650;}
[data-testid="stAlert"] {border-radius:12px;}
small {color:#50647B;}
@media(max-width:640px){.hero-card {padding:20px 18px;border-radius:16px;}.block-container {padding-top:1rem;}}


[data-testid="stSidebar"] button {
    height: auto !important;
    min-height: 44px;
    white-space: normal !important;
    overflow-wrap: anywhere;
    line-height: 1.4 !important;
    padding: 0.65rem 0.8rem !important;
}

[data-testid="stSidebar"] button p {
    white-space: normal !important;
    overflow-wrap: anywhere;
    line-height: 1.4 !important;
}

</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def demo_data() -> pd.DataFrame:
    """Generate one stable fictional dataset, never external learner data."""
    return generate_synthetic_records(n_learners=60, seed=42)


@st.cache_data(show_spinner=False)
def logo_as_data_uri() -> str:
    """Embed a trusted local asset so the app needs no external image host."""
    path = ROOT / "assets" / "learnadapt_logo.png"
    if not path.is_file():
        return ""
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


@st.cache_resource(show_spinner=False)
def load_trusted_local_model(path: str, mtime: float) -> dict[str, Any]:
    """Cache the serialized local model; mtime invalidates on replacement."""
    return load_bundle(path)


def brand_banner() -> None:
    """Render locally embedded, consistent LearnAdapt branding."""
    logo = logo_as_data_uri()
    image = f'<img src="{logo}" alt="LearnAdapt AI logo" />' if logo else ""
    st.markdown(
        '<div class="hero-card">'
        '<div class="hero-kicker">AI INNOVATION HACKATHON 2026 / CHALLENGE 12</div>'
        + image
        + '<h1>Every next lesson should fit the learner.</h1>'
        + '<p>Explainable learning-path suggestions informed by recorded practice, '
          'explicit prerequisites, and transparent uncertainty.</p>'
        + '<p><strong>TEAM ERROR-404</strong> &nbsp; | &nbsp; LearnAdapt AI</p>'
        + '</div>',
        unsafe_allow_html=True,
    )


def describe_source(source: str) -> None:
    """Always disclose synthetic vs uploaded records."""
    if source == "synthetic":
        st.markdown('<span class="source-pill">SYNTHETIC DEMO DATA | FICTIONAL LEARNERS</span>', unsafe_allow_html=True)
        st.caption("Demonstration records were generated locally; they do not represent real students.")
    else:
        st.markdown('<span class="source-pill">UPLOADED CSV | SOURCE NOT VERIFIED</span>', unsafe_allow_html=True)
        st.caption("User-provided records. Data collection, consent, and licensing have not been independently verified.")


def _load_current_model(source_type: str, fingerprint: str) -> tuple[dict[str, Any] | None, str]:
    """Require matching data source and dataset fingerprint before inference."""
    memory = st.session_state.get("synthetic_model_bundle")
    if source_type == "synthetic" and memory is not None:
        if memory.get("dataset_fingerprint") == fingerprint:
            return memory, "Synthetic classifier trained in this session; held-out learner evaluation available."
        return None, "Session model refers to a different dataset. Using prerequisite-aware rules."
    if not MODEL_PATH.is_file():
        return None, "No trained model artifact found. Explainable rule-based fallback is active."
    try:
        bundle = load_trusted_local_model(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)
    except (ModelError, OSError) as exc:
        return None, f"Model not usable ({exc}). Rule-based fallback is active."
    if bundle["training_source"] != source_type or bundle.get("dataset_fingerprint") != fingerprint:
        return None, (
            "The saved model does not match this dataset and source type. "
            "To avoid misleading estimates, using prerequisite-aware rules instead."
        )
    return bundle, f"Matching {source_type} model artifact loaded; classifier probabilities are not calibrated."


def _get_updates(context: str, learner_id: str) -> pd.DataFrame:
    """Combine private session entries with optional opt-in local SQLite rows."""
    records = [
        item for item in st.session_state.get("practice_updates", [])
        if item["context"] == context and item["row"]["learner_id"] == learner_id
    ]
    session = pd.DataFrame([item["row"] for item in records], columns=ALL_COLUMNS)
    if not LOCAL_SQLITE_ENABLED:
        return session
    try:
        saved = load_local_records(DATABASE_PATH, context, learner_id)
    except sqlite3.Error as exc:
        st.warning(f"Local SQLite could not be read: {type(exc).__name__}. Session entries remain usable.")
        return session
    if saved.empty:
        return session
    return pd.concat([session, saved], ignore_index=True)


def _selected_history(base: pd.DataFrame, learner_id: str, context: str) -> pd.DataFrame:
    """Get exactly the selected learner's records plus user-entered updates."""
    subset = base.loc[base["learner_id"] == learner_id].copy()
    updates = _get_updates(context, learner_id)
    if updates.empty:
        return subset if not subset.empty else empty_history()
    combined = pd.concat([subset, updates], ignore_index=True)
    return validate_records(combined)


def accuracy_chart(skills: pd.DataFrame) -> None:
    """Plot measured practice accuracy; never present it as mastery."""
    if skills.empty:
        st.info("No skill observations yet. Record a practice session to view your history.")
        return
    chart_data = skills.copy()
    chart_data["Observed correct (%)"] = (chart_data["accuracy"] * 100).round(1)
    fig = px.bar(
        chart_data, x="skill", y="Observed correct (%)", text="Observed correct (%)",
        color="skill", color_discrete_sequence=["#1D4ED8", "#14B8A6", "#7C3AED", "#F59E0B"],
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside", marker_line_width=0)
    fig.update_layout(
        showlegend=False, template="plotly_white", margin=dict(l=12, r=12, t=25, b=0),
        height=340, yaxis_range=[0, 110], xaxis_title="Skill", yaxis_title="Correct responses (%)",
        font=dict(family="Arial", color="#1F3656"), plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, width="stretch")
    st.caption("Observed performance: correct responses / attempts; not a validated mastery measurement.")


def display_activity(recommendation: dict[str, Any], *, model_status: str) -> None:
    """Render a structured, honest recommendation and evidence explanation."""
    with st.container(border=True):
        st.subheader(recommendation["activity_title"])
        c1, c2, c3 = st.columns(3)
        c1.metric("Skill", recommendation["skill"])
        c2.metric("Difficulty", recommendation["difficulty"])
        c3.metric("Estimated activity time", f"{recommendation['estimated_minutes']} min")
        st.write("**Why this activity?**")
        st.write(recommendation["reason"])
        st.caption(f"Selection method: {recommendation['method'].replace('_', ' ')}")
        if recommendation["estimated_success_probability"] is not None:
            st.info(
                "Classifier estimate for next-session score >= 60%: "
                f"{recommendation['estimated_success_probability']:.1%}. "
                "Probability calibration was not evaluated; this is not a confidence or mastery score."
            )
        if recommendation["blocked_skills"]:
            st.warning(
                "Prerequisites are not yet established for some advanced options: "
                + ", ".join(recommendation["blocked_skills"])
            )
        st.caption("Catalogue suggestions are descriptions, not links to verified learning materials.")
    st.caption("Model status: " + model_status)


def practice_table(history: pd.DataFrame, *, count: int | None = None) -> None:
    """Display session observations without exposing IDs or raw private fields."""
    if history.empty:
        st.info("No practice sessions recorded. Use Practice Update to add one.")
        return
    shown = history.sort_values("timestamp", ascending=False)
    if count:
        shown = shown.head(count)
    table = shown.loc[:, ["timestamp", "skill", "activity_id", "attempts", "correct_attempts", "score", "duration_minutes"]].copy()
    table["timestamp"] = table["timestamp"].dt.strftime("%Y-%m-%d %H:%M UTC")
    table["score"] = (table["score"] * 100).round(1)
    table = table.rename(columns={
        "timestamp": "Practice time", "skill": "Skill", "activity_id": "Activity ID",
        "attempts": "Attempts", "correct_attempts": "Correct", "score": "Correct (%)",
        "duration_minutes": "Minutes",
    })
    st.dataframe(table, width="stretch", hide_index=True)


def dashboard(history: pd.DataFrame, source: str, model_bundle: dict | None, model_status: str) -> None:
    """Show practice summary, recent activity, and a real recommendation."""
    st.header("Learning dashboard")
    skill_frame = skill_summary(history, str(history.iloc[0]["learner_id"])) if not history.empty else pd.DataFrame()
    attempts = int(history["attempts"].sum()) if not history.empty else 0
    correct = int(history["correct_attempts"].sum()) if not history.empty else 0
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Practice sessions", len(history))
    m2.metric("Skills practiced", len(skill_frame))
    m3.metric("Attempts recorded", attempts)
    m4.metric("Observed correct", f"{100 * correct / attempts:.1f}%" if attempts else "N/A")
    st.subheader("Skill-level evidence")
    accuracy_chart(skill_frame)
    st.subheader("Suggested next activity")
    recommendation = recommend_activity(history, model_bundle=model_bundle, source_type=source)
    display_activity(recommendation, model_status=model_status)
    st.subheader("Recent practice")
    practice_table(history, count=5)


def progress_page(history: pd.DataFrame) -> None:
    """Skill summary and historical trend of actual recorded practice sessions."""
    st.header("Skill progress")
    if history.empty:
        st.info("No practice data yet. Use Practice Update to start collecting evidence.")
        return
    learner_id = str(history.iloc[0]["learner_id"])
    metrics = skill_summary(history, learner_id)
    accuracy_chart(metrics)
    table = metrics.rename(columns={
        "skill": "Skill", "sessions": "Sessions", "attempts": "Attempts",
        "correct_attempts": "Correct attempts", "accuracy": "Overall correct",
        "recent_accuracy": "Recent correct",
    }).copy()
    table["Overall correct"] = (table["Overall correct"] * 100).round(1)
    table["Recent correct"] = (table["Recent correct"] * 100).round(1)
    st.dataframe(table, hide_index=True, width="stretch")
    chosen_skill = st.selectbox("View historical performance for skill", sorted(history["skill"].unique()))
    subset = history.loc[history["skill"] == chosen_skill].sort_values("timestamp").copy()
    subset["Correct (%)"] = subset["score"] * 100
    if len(subset) >= 2:
        fig = px.line(subset, x="timestamp", y="Correct (%)", markers=True, template="plotly_white")
        fig.update_traces(line_color="#1D4ED8", marker=dict(size=9, color="#14B8A6"))
        fig.update_layout(
            height=305, yaxis_range=[0, 105], xaxis_title="Practice date (UTC)",
            yaxis_title="Correct responses (%)", margin=dict(l=12, r=12, t=18, b=0),
        )
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("At least two sessions for this skill are needed to show a historical trend.")
    st.caption("A changing percentage is a descriptive observation, not a verified improvement or causal effect.")


def history_page(history: pd.DataFrame) -> None:
    """Practice record browser with safe filtering and clear empty states."""
    st.header("Practice history")
    if history.empty:
        st.info("No practice activity is available for this learner.")
        return
    options = ["All skills", *sorted(history["skill"].unique().tolist())]
    selected = st.selectbox("Filter by skill", options)
    filtered = history if selected == "All skills" else history.loc[history["skill"] == selected]
    st.caption(f"Showing {len(filtered)} recorded practice sessions. Times are UTC.")
    practice_table(filtered)


def recommend_page(history: pd.DataFrame, source: str, model_bundle: dict | None, model_status: str) -> None:
    """Explain one ranked candidate, the explicit gate, and fallback approach."""
    st.header("Next learning activity")
    st.write("Recommendations are generated from the selected learner's completed practice sessions.")
    display_activity(recommend_activity(history, model_bundle=model_bundle, source_type=source), model_status=model_status)
    with st.expander("How was this recommendation decided?"):
        st.markdown("""
1. **Observe:** count completed attempts by skill, including recent practice.
2. **Predict when possible:** use a trained classifier for a *next-session* score >= 60%.
3. **Check:** require at least 8 recorded prerequisite attempts and >= 70% observed correct.
4. **Rank:** balance practice gaps, recent errors, and suitable difficulty.
5. **Update:** record another session to refresh the recommendation.

These thresholds are demonstration heuristics, not validated instructional policy.
For new learners, a short diagnostic is suggested rather than a speculative score.
""")


def update_page(learner_id: str, context: str) -> None:
    """Validated practice form with session memory and optional local SQLite."""
    st.header("Practice update")
    st.write("Log a completed activity. Suggestions update after the next rerun.")
    st.info(
        "Entries are **session-only** by default. Optional local SQLite is for a "
        "trusted, single-user local machine only (LEARNADAPT_ENABLE_SQLITE=1); do "
        "not enable it on a public server. Streamlit Community Cloud does not "
        "guarantee durable local files."
    )
    available_skills = list(dict.fromkeys(activity.skill for activity in ACTIVITY_CATALOGUE))
    with st.form("practice_update_form", clear_on_submit=False):
        skill = st.selectbox("Skill practiced", available_skills)
        options = [activity for activity in ACTIVITY_CATALOGUE if activity.skill == skill]
        activity_id = st.selectbox(
            "Activity completed", [activity.activity_id for activity in options],
            format_func=lambda ident: next(a.title for a in options if a.activity_id == ident),
        )
        selected_activity = next(activity for activity in options if activity.activity_id == activity_id)
        st.caption("Difficulty: " + selected_activity.difficulty)
        c1, c2, c3 = st.columns(3)
        attempts = int(c1.number_input("Number of attempts", min_value=1, max_value=100, value=5, step=1))
        correct = int(c2.number_input("Correct attempts", min_value=0, max_value=attempts, value=min(3, attempts), step=1))
        minutes = float(c3.number_input("Time spent (minutes)", min_value=1.0, max_value=180.0, value=10.0, step=1.0))
        save_to_db = False
        if LOCAL_SQLITE_ENABLED:
            save_to_db = st.checkbox("Also save this record in LOCAL SQLite (not permanent in hosted cloud)", value=False)
        submit = st.form_submit_button("Record completed practice", type="primary", width="stretch")
    if not submit:
        return
    row = {
        "learner_id": learner_id, "skill": skill, "activity_id": activity_id,
        "difficulty": selected_activity.difficulty, "attempts": attempts,
        "correct_attempts": correct, "score": correct / attempts,
        "duration_minutes": minutes, "timestamp": datetime.now(timezone.utc).isoformat(),
        "prerequisite_skill": selected_activity.prerequisite_skill,
    }
    try:
        validate_records(pd.DataFrame([row]))
        if save_to_db:
            save_local_record(DATABASE_PATH, context, row)
            notice = "Completed practice saved to local SQLite; cloud storage is not durable."
        else:
            st.session_state.setdefault("practice_updates", []).append({"context": context, "row": row})
            notice = "Completed practice recorded for this browser session."
    except (DataValidationError, sqlite3.Error, OSError) as exc:
        st.error(f"Could not save record: {exc}")
        return
    st.session_state["save_notice"] = notice
    st.rerun()


def main() -> None:
    """Run the app with explicit model and data provenance."""
    brand_banner()
    st.sidebar.title("LearnAdapt AI")
    st.sidebar.caption("TEAM ERROR-404 | AI Innovation Hackathon 2026")
    source_choice = st.sidebar.radio("Data source", ["Synthetic demonstration", "Upload my CSV"], index=0)
    source_type = "synthetic" if source_choice == "Synthetic demonstration" else "real"
    if source_type == "synthetic":
        base = demo_data().copy()
        context = "fictional_seed42_v1"
        st.sidebar.download_button(
            "Download clearly labelled synthetic CSV",
            data=base.to_csv(index=False).encode("utf-8"),
            file_name="learnadapt_SYNTHETIC_demo.csv", mime="text/csv",
        )
    else:
        upload = st.sidebar.file_uploader("Practice records (.csv)", type=["csv"], help="Use fictional or consented data; no names required.")
        if upload is None:
            describe_source(source_type)
            st.info("Upload a CSV with the required columns. No model or scores will be fabricated while awaiting data.")
            with st.expander("Required CSV fields"):
                st.code("learner_id,skill,activity_id,difficulty,attempts,correct_attempts,timestamp", language="text")
                st.caption("Optional: score (0-1), duration_minutes, prerequisite_skill")
            return
        raw = upload.getvalue()
        context = "uploaded_" + hashlib.sha256(raw).hexdigest()[:18]
        try:
            base = load_csv(io.BytesIO(raw))
        except DataValidationError as exc:
            describe_source(source_type)
            st.error(str(exc))
            return
    describe_source(source_type)
    for warning in base.attrs.get("validation_warnings", []):
        st.warning("Data preparation: " + warning)

    distinct_learners = sorted(base["learner_id"].astype(str).unique().tolist())
    learner_options = [*distinct_learners, "__new__"]
    def label_for_learner(item: str) -> str:
        if item == "__new__":
            return "New learner (no history)"
        if source_type == "synthetic":
            return "Aisha (fictional demo learner)" if item == "SYN-001" else f"Fictional learner {item.split('-')[-1]}"
        return f"Uploaded learner #{distinct_learners.index(item) + 1}"
    learner_id = st.sidebar.selectbox("Learner profile", learner_options, format_func=label_for_learner)
    if learner_id == "__new__":
        learner_id = "SYN-NEW" if source_type == "synthetic" else "USER-NEW"
    history = _selected_history(base, learner_id, context)

    st.sidebar.divider()
    section = st.sidebar.radio(
        "Explore", ["Dashboard", "Skill Progress", "Practice History", "Next Learning Activity", "Practice Update"],
    )
    st.sidebar.divider()
    
    if source_type == "synthetic":
        if st.sidebar.button(
            "Train synthetic model in this session",
            key="train_synthetic_model",
            width="stretch",
        ):
            try:
                with st.spinner(
                    "Training a classifier and evaluating on held-out fictional learners..."
                ):
                    bundle, report = train_and_evaluate(
                        base,
                        source_type="synthetic",
                        seed=42,
                    )
                    st.session_state["synthetic_model_bundle"] = bundle
                    st.session_state["synthetic_model_report"] = report

                st.sidebar.success(
                    "Synthetic model trained and evaluated."
                )

            except (ModelError, DataValidationError, ValueError) as exc:
                st.sidebar.error(f"Training unavailable: {exc}")

        if "synthetic_model_report" in st.session_state:
            with st.sidebar.expander("Calculated demo evaluation", expanded=False):
                report = st.session_state["synthetic_model_report"]

                st.markdown("**Held-out learners**")
                st.write(report.get("test_learners", "Not available"))

                model_metrics = report.get("model", {})
                st.markdown("**Synthetic model metrics**")

                for metric_name in ("accuracy", "precision", "recall", "f1", "roc_auc"):
                    metric_value = model_metrics.get(metric_name)

                    if isinstance(metric_value, (int, float)):
                        st.write(f"{metric_name.replace('_', ' ').title()}: {metric_value:.1%}")
                    else:
                        st.write(f"{metric_name.replace('_', ' ').title()}: Not available")

                st.caption(
                    "Results use fictional synthetic data and do not establish "
                    "performance on real students."
                )


    model_bundle, model_status = _load_current_model(source_type, dataset_fingerprint(base))
    st.sidebar.caption(model_status)
    st.sidebar.caption("Privacy: CSV uploads are processed in the running app server's memory. Never upload identifiable student data to a public demo.")

    if st.session_state.get("save_notice"):
        st.success(st.session_state.pop("save_notice"))
    if section == "Dashboard":
        dashboard(history, source_type, model_bundle, model_status)
    elif section == "Skill Progress":
        progress_page(history)
    elif section == "Practice History":
        history_page(history)
    elif section == "Next Learning Activity":
        recommend_page(history, source_type, model_bundle, model_status)
    else:
        update_page(learner_id, context)

    st.divider()
    st.caption(
        "LearnAdapt AI | TEAM ERROR-404 | Mohammad Laeeque Anwar, Hamna Rehan, Tara V., Mehak Shah. "
        "Demo estimates are educational prototypes, not diagnoses or proof of effectiveness."
    )


if __name__ == "__main__":
    main()
