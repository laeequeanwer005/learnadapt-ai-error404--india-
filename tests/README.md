# Automated checks

Run from the repository root with `python -m pytest -q`.

`test_data_processing.py` validates required fields, numeric ranges, bad dates, optional values, duplicate handling, deterministic synthetic examples, preceding-session feature construction, CSV loading, grouped statistics, and parameterized SQLite saves.

`test_model.py` trains a classifier on fictional learners with a group-disjoint holdout, checks actual evaluation-report structure, confirms deterministic predictions and probability ranges, verifies artifact roundtrips, and exercises missing and corrupt model handling.

`test_recommendation.py` covers no-history diagnostics, prerequisite gates, conservative unknown prerequisites, reproducible rule fallback, incompatible synthetic-vs-real model handling, and model-assisted ranking.

No tests require the internet, external credentials, user data, or a pre-trained artifact. Unit tests do not establish statistical fairness or educational effectiveness. GUI smoke checks and actual Streamlit Community Cloud deployment are separate verification tasks.
