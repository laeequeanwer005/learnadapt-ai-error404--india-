# Verification record (20 October 2026)

This is a reproducible engineering audit, **not** an externally verified educational evaluation.

## What was executed

In the authoring environment with Python **3.13.5**, pandas 2.2.3, NumPy 2.3.5, scikit-learn 1.8.0, Plotly 6.5.2, Joblib 1.5.3 and pytest 9.0.2:

```bash
python -m compileall -q app.py src tests
python -m pytest -q
python -m src.data_processing generate-demo --output synthetic_demo.csv --learners 60 --seed 42
python -m src.model train --data synthetic_demo.csv --source synthetic --artifact learnadapt_model.joblib --report evaluation.json --seed 42
python -m src.model evaluate --report evaluation.json
```

- **37 tests passed** with no warnings after final code revisions.
- The generator produced **1,440 fictional sessions**.
- The trained model's dataset fingerprint matched the canonical generated CSV and in-memory generator output.
- The model artifact was saved, reloaded, and generated a structured model-assisted recommendation; a cold-start history generated a diagnostic without an invented probability.
- Synthetic-only held-out split: **45 train learners / 15 test learners**; **900 train rows / 300 test rows** after removing first-skill sessions with no prior evidence.
- Actual held-out classifier metrics: accuracy **0.760**, precision **0.791**, recall **0.762**, F1 **0.776**, ROC-AUC **0.827**; confusion matrix `[[103, 33], [39, 125]]`.
- Actual held-out prior/majority baseline accuracy **0.547**; ROC-AUC **0.500**.

**Interpretation:** These outcomes measure performance against generated fictional labels only. They **do not** establish real student prediction reliability, calibrated probabilities, or learning benefit. Metrics may differ with a different dependency environment.

## Not verified in this environment

- **Streamlit UI live/browser smoke test:** Streamlit was unavailable locally, and package installation was blocked by a network/DNS error. The app code compiled, but interactive rendering was not tested here.
- **Python 3.11 runtime:** the project targets Python 3.11 with compatible dependency constraints; the code/tests above were executed with Python 3.13.5 because 3.11 was unavailable in this environment.
- **Hosted app/GitHub integration:** no live deployment, repository changes or authentication were performed.
- **Real educational data/real-world predictive value:** no real dataset supplied, no data license or consent verified, and no human validation conducted.
- **UI screenshots:** none fabricated; see `screenshots/README.md` for instructions to capture genuine ones after launch.

## Artifact policy

The synthetic training artifact and report were created separately for verification. They are **not included** in the default source ZIP because generated Joblib binaries can be version-specific and can execute code if untrusted. Use the documented CLI to generate your own reproducible artifact and report. `models/README.md` explains artifact lifecycle and trust boundaries.
