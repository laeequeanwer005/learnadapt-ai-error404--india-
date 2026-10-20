# System architecture — LearnAdapt AI

## Implemented components

- **`app.py` (Streamlit):** rendering, navigation, file upload, chosen (pseudonymous) learner context, Plotly skill charts, practice form and clear model status. No model math or prerequisite decision logic lives in UI code.
- **`src/data_processing.py`:** validated session-schema loading, synthetic generator, chronological feature engineering, learner/skill aggregations, optional local SQLite and CLI data generator.
- **`src/model.py`:** scikit-learn preprocessing/classifier, learner-disjoint heldout split, baseline, actual metrics, versioned Joblib bundle and training/evaluation CLI.
- **`src/recommendation.py`:** declared local activity catalogue, transitive prerequisite checks, skill-gap ranking, optional model-assisted selection, evidence-linked human-readable reasons and conservative cold start.
- **`tests/`:** deterministic contract, model and prerequisite tests.

```mermaid
flowchart TD
    A[Seeded fictional session data] --> C[Validate & normalize records]
    B[User-selected permissioned CSV] --> C
    C --> D[Chronological prior-session features]
    D --> E[Group-disjoint training & holdout evaluation]
    E --> F[Versioned local Joblib model + synthetic-only report]
    C --> G[Selected learner's observed practice history]
    G --> H[Prerequisite gates (direct and transitive)]
    H --> I[Rank eligible catalogue activities]
    F --> J{Model source & dataset fingerprint match?}
    J -->|Yes| I
    J -->|No| K[Transparent rule-based fallback]
    K --> I
    I --> L[Explanation + suggested next activity]
    L --> M[Five-section Streamlit dashboard]
    M --> N[Validated practice update (session memory by default)]
    N --> G
```

## Exact flow

1. `validate_records` rejects missing/invalid fields and discloses removal of identical duplicate rows; displays source label.
2. Streamlit filters practice history to one opaque learner ID. Charts/metrics use observed counts rather than hardcoded synthetic percentages.
3. The model sees only earlier same-learner/same-skill records when training the `>=60%` next-session success target. A grouped learner train/test split avoids the same learner leaking across evaluation folds.
4. A locally trained model is eligible for UI use only if its source type and canonical data fingerprint match the loaded base dataset. If missing/incompatible/unavailable, rules still work.
5. The recommender first closes all candidates whose entire known prerequisite chain has too little evidence (fewer than 8 attempts) or fewer than 70% correct. Only eligible candidates are ranked. Weights/thresholds are heuristic demonstration settings, not validated pedagogy.
6. The dashboard explains evidence and method, and never presents an invented confidence score. For a new learner with no history, it proposes a short diagnostic.
7. On **Practice Update**, a new validated record is appended to Streamlit session memory, or saved via opt-in parameterized SQLite on **single-user local machines only**. Recommendations refresh on rerun.

## Deployment boundaries

Streamlit Community Cloud may restart and lose in-memory or local files. CSV uploads are processed in server memory. The sample SQLite implementation has **no user identity/authentication boundary** and must not be enabled on a public multi-user deployment. The static activity catalogue is not a library of verified external lessons. Model loading uses Joblib serialization and must be restricted to trusted artifacts. Consult `docs/limitations.md` before using any personal learning data.
