# LearnAdapt AI — Personalized Learning Path Prediction

**Team:** ERROR-404 (India)  
**Event:** AI Innovation Hackathon 2026 — Women in AI Research, Innovation & Entrepreneurship  
**Challenge:** 12 — LearnAdapt AI: Personalized Learning Path Prediction  
**Track:** AI/ML Methodology

**Team members:** Mohammad Laeeque Anwar (team leader), Hamna Rehan, Tara V., Mehak Shah.

> **Status:** Runnable Python application with data validation, a scikit-learn classifier training/evaluation pipeline, prerequisite-aware rule-based recommendations, optional model-assisted ranking, and optional local-only SQLite. The default dataset is **100% synthetic**. Predictions and evaluations are technical demonstrations, **not evidence of educational effectiveness**.

## The problem

A fixed sequence of lessons can ignore differences in practice history and prerequisite readiness. Learners may benefit from different next activities, but a score alone cannot explain what to practice or why. The project explores a careful, explainable next-step selection process rather than declaring fixed student abilities.

## The solution

1. **Observe** completed, skill-tagged practice sessions.
2. **Extract features** from past sessions, never future outcomes.
3. **Predict, when available** an uncalibrated estimate of whether the **next practice session** will have >=60% correct attempts.
4. **Check prerequisites** using explicit demonstration gates.
5. **Rank eligible activities** with observed performance, recent history, practice difficulty and optionally model output.
6. **Explain** the recommended activity with evidence and caveats.
7. **Update** the learner's observed record when a new session is entered.

No prediction should be interpreted as mastery, a diagnosis, or a guarantee of progress.

## Working features

- A blue-and-white, responsive Streamlit dashboard retaining the original five areas: **Dashboard**, **Skill Progress**, **Practice History**, **Next Learning Activity**, and **Practice Update**.
- Synthetic, reproducible dataset generated deterministically (60 fictional learners by default).
- CSV upload with schema validation, descriptive errors, strict field checks and explicit source status.
- Observed per-skill and recent accuracy, attempt counts, timeline visualizations, filtering and empty states.
- Leakage-conscious scikit-learn logistic regression pipeline, grouped **learner-disjoint** train/test split, DummyClassifier baseline, actual holdout metrics and Joblib artifact support.
- A prerequisite-aware, explainable activity catalogue, including **transitive** gates (Foundations -> Fractions -> Algebra -> Graphs), cautious cold starts and no fabricated confidence.
- Optional session-based synthetic model training via the Streamlit sidebar. Models are cached/loaded, not retrained on every rerun.
- A local practice-entry form with validation; entries are session-only unless optional **single-user, local** SQLite is deliberately enabled. SQLite uses parameterized SQL.
- Automated deterministic Pytest cases for input validation, history features, serialization, model predictions, privacy-safe filters and activity gates.

## Technology

Python **3.11** (recommended), Streamlit, pandas, NumPy, scikit-learn, Plotly, pytest, SQLite (Python standard library) and Joblib. There is no separate web framework or frontend build process.

## Repository map

```text
LearnAdapt-AI/
├── app.py                          # all five Streamlit sections
├── requirements.txt                # compatible dependency ranges
├── README.md
├── .gitignore
├── .streamlit/config.toml          # blue-and-white light theme
├── src/
│   ├── __init__.py
│   ├── data_processing.py          # CSV, synthetic data, features, SQLite, CLI
│   ├── model.py                    # training, evaluation, serialization, CLI
│   └── recommendation.py           # activity catalogue, gates, explanations
├── data/README.md                  # input schema and consent guidance
├── models/README.md                # artifact and evaluation notes
├── docs/
│   ├── architecture.md
│   ├── methodology.md
│   ├── limitations.md
│   ├── github_merge.md
│   └── verification.md
├── tests/
│   ├── README.md
│   ├── test_data_processing.py
│   ├── test_model.py
│   └── test_recommendation.py
├── screenshots/README.md          # directions for real screenshots
└── assets/                         # existing team branding PNGs
    ├── README.md
    ├── learnadapt_icon.png
    └── learnadapt_logo.png
```

Generated CSV and models are intentionally excluded from Git by `.gitignore`; you must generate and document an artifact before deploying it. `models/` has only its README in the distribution.

## Install (Windows PowerShell)

From the repository folder, run:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If you only have one Python installation, replace the first command with `py -m venv .venv`.

If activation is blocked, do **not** change your system-wide execution policy. Instead:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## Exact development workflow

Commands below use implemented module entrypoints and are run **from the project root** after activating the environment:

```powershell
# 1. Generate fictional examples (the Streamlit app can also generate them in memory)
python -m src.data_processing generate-demo --output data/synthetic_demo.csv --learners 60 --seed 42

# 2. Train and EVALUATE on learner-disjoint holdout data; save the real calculation
python -m src.model train --data data/synthetic_demo.csv --source synthetic --artifact models/learnadapt_model.joblib --report models/evaluation.json --seed 42

# 3. Display the ACTUAL saved holdout report (this does not retrain/re-evaluate)
python -m src.model evaluate --report models/evaluation.json

# 4. Run deterministic tests
python -m pytest -q

# 5. Launch the app
streamlit run app.py
```

**No CLI training is needed for initial use**: `streamlit run app.py` works without CSV or model files. Select **Synthetic demonstration** for rule-based fallback, or click **Train synthetic model in this session** to train and use a held-out-evaluated fictional classifier. That on-demand training only executes when clicked, not every rerun.

To use your own **de-identified, permissioned** CSV:

1. Prepare a file with the schema in [`data/README.md`](data/README.md).
2. Choose **Upload my CSV** in the sidebar; the UI explicitly labels it *source not verified*.
3. Until an artifact trained on that **same canonical dataset** and marked `real` is present, recommendations remain rule-based.
4. For local experiments only, use:

```powershell
python -m src.model train --data path\to\permitted_dataset.csv --source real --artifact models/learnadapt_model.joblib --report models/evaluation.json
```

The app verifies a model's `source_type` **and a canonical dataset SHA-256 fingerprint** against the chosen base dataset. Do not reuse a synthetic model for another dataset. A fingerprint does **not** verify licenses, consent or validity. Do not upload personal student information to public Streamlit servers.

## Data schema and provenance

Each row represents **one completed practice session** (not one student's permanent ability). Required CSV columns:

| Column | Meaning | Validation |
|---|---|---|
| `learner_id` | Opaque synthetic/pseudonymous identifier (never a predictor) | Non-empty text |
| `skill` | Skill name, e.g. `Fractions` | Non-empty text |
| `activity_id` | Practice catalogue ID or existing activity ID | Non-empty text |
| `difficulty` | Proposed/recorded activity level | `Beginner`, `Intermediate`, `Advanced` |
| `attempts` | Number of questions attempted that session | Positive integer |
| `correct_attempts` | Number correct | Integer from 0 to attempts |
| `timestamp` | When practice occurred | Parseable timestamp, interpreted as UTC if timezone absent |

Optional: `score` (0–1, consistent with counts; derived if missing), `duration_minutes` (strictly positive or blank) and `prerequisite_skill` (skill identifier or blank). There are no required names, emails, demographics or confidential assessments. Missing required fields and invalid values are rejected with useful error messages rather than silently dropped. Only **identical** duplicate records are removed, with a warning. Unsupported extra columns are ignored **with a warning**; ambiguous same-learner/same-skill timestamp ties are rejected during training.

The app's synthetic generator is in `src/data_processing.py`, seeded and labelled. **No public third-party education dataset is bundled, cited, or claimed licensed.** If you later use one, verify its real schema, permission and license first.

## Model, evaluation and ranking

**Model target:** `target_success = 1` when the **next recorded session** achieves at least 60% correct responses, else `0`. Features describe the *earlier* sessions in the same skill: cumulative attempts, session count, past weighted accuracy, weighted accuracy over at most three recent previous sessions, most recent past score, chosen skill and activity difficulty. The target is **not** correctness of each individual next question, future educational success or measured mastery.

The logistic regression pipeline includes preprocessing. A `GroupShuffleSplit` keeps entire learners out of training when evaluating on the test group. Training excludes the first session per learner/skill because there is no prior same-skill evidence. A majority/prior `DummyClassifier` baseline provides context. Measured holdout accuracy, precision, recall, F1, ROC-AUC (only if both test classes exist) and confusion matrix are emitted by the command; actual values depend on the generated data and software environment. **No real-student evaluation has been performed; efficacy and calibration remain unverified.**

The recommendation engine first excludes candidates whose declared prerequisite chain has fewer than **8 attempts** or fewer than **70% observed correct**. It then ranks available activities using skill/recent practice gaps, cautious difficulty, and optionally the classifier's **uncalibrated** probability. These gate thresholds and weights are demonstration heuristics, not validated educator policy. If no evidence is available, it suggests a diagnostic; if no compatible model is available, it uses deterministic rules. External activity URLs are never fabricated.

See [`models/README.md`](models/README.md), [`docs/methodology.md`](docs/methodology.md), and [`docs/limitations.md`](docs/limitations.md) for technical details. See [`docs/verification.md`](docs/verification.md) for the executed test and synthetic evaluation record.

## Evaluation results and status

The application computes metrics **from held-out predictions**, not invented values. Run the training command and inspect `models/evaluation.json` for the reproducible **synthetic-only** report. There are presently no independently validated real-world model results, learning improvement results, or recommendation-relevance benchmarks to claim. CLI `evaluate` reads the saved report; it does not pretend to be a second external evaluation.

## Responsible use and persistence

- This is an educational decision-support prototype: no automatic grading, student labeling, or consequential placement decisions.
- Never interpret observed % correct as mastery; evidence can be sparse or noisy.
- Learners/instructors should be able to ignore or override suggested activities.
- Synthetic records and uploaded CSVs are processed by the app **server**, not solely in the browser. Avoid real identifiable learner uploads to publicly hosted demos.
- By default, updates last only for the Streamlit session.
- **SQLite is local opt-in and single-user only**: set `LEARNADAPT_ENABLE_SQLITE=1` on your own trusted machine to expose the local-save checkbox. Do **not** use this unprotected demo database on a public multi-user service; it has no identity/authentication boundary. A Streamlit Community Cloud filesystem is ephemeral and not durable database storage.
- Joblib artifacts are pickle-based. **Only load artifacts created by your trusted team**, never downloaded/unknown files.

## Streamlit Community Cloud deployment

1. Review/merge the generated files into your own GitHub repository; follow [`docs/github_merge.md`](docs/github_merge.md). Back up your original `app.py`, `README.md` and `requirements.txt` before replacing anything.
2. Commit `.streamlit/config.toml` **under `.streamlit/`**, all `src/` code, `app.py`, `requirements.txt`, documentation and local image assets. Do **not** commit `.env`, personal data, `.local/`, a private DB or untrusted model files.
3. In Streamlit Community Cloud, create a new app from **your actual GitHub repo**, branch **`main`**, and entrypoint **`app.py`** at the root. Choose a compatible Python runtime (ideally 3.11) in deployment settings if offered.
4. Build from `requirements.txt` and read the app and package-install logs for missing libraries. Common mistakes: incorrect case-sensitive paths, accidental root `config.toml` instead of `.streamlit/config.toml`, missing assets, or noncommitted `src/` files.
5. Because `.joblib` is excluded from Git, the deployed app starts in **rule-based demo mode**. Use the button to train a **fictional session model** in memory if you want to demonstrate ML, but expect it to reset when the app restarts. To make a pre-trained artifact available, deliberately deploy a trusted, version-compatible artifact after inspecting privacy/security concerns and the artifact's source provenance.
6. Avoid enabling SQLite for public cloud deployment; use session-only practice updates. Deployment, access policy, and any hosted URL must be verified by the team.

## Screenshots, links and credits

- **Live Streamlit demo URL:** `[ADD AFTER VERIFIED DEPLOYMENT]`
- **GitHub repository URL:** `[ADD YOUR ACTUAL LINK]`
- **Genuine UI screenshots:** see [`screenshots/README.md`](screenshots/README.md); add only captures from the running Streamlit application.
- **Code/design contributions:** Team ERROR-404 (see names above). Record each member's **actual** contributions before hackathon submission.
- **Libraries:** Streamlit, pandas, NumPy, scikit-learn, Plotly, pytest and Joblib; see dependency metadata and their respective licenses.
- **Logo:** team-provided LearnAdapt AI design assets included for visual identity. Check team permission/licensing before republishing.
- **External datasets:** none packaged. Attribution/license will be recorded if one is adopted.
- **Source license:** not selected by the team. Do not claim MIT/Apache or redistribute as open source under an assumed license.

## Limitations and future work

Fictional data may not reflect learners. Historical accuracy is noisy, predictions are not calibrated, prerequisite mappings are simplified, and neither causal learning improvement nor fairness/generalization has been measured. Before real student use: verify data permissions; obtain representative consented data; evaluate calibration/uncertainty, subgroup fairness and out-of-sample performance; consult teachers about gates; conduct user testing and robust privacy/security assessment. See [`docs/limitations.md`](docs/limitations.md).
