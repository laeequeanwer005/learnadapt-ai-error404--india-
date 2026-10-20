# Model artifacts and training policy

**Prediction target:** Whether the next completed practice session on a selected skill meets **>=60% correct attempts**. This is a binary target computed from session counts. It is not a prediction of a single question, a mastery estimate, or educational impact.

## Implementation

- `src/model.py` implements a scikit-learn `Pipeline(ColumnTransformer, LogisticRegression)` with median numeric imputation, numeric standardization and one-hot encodings for `skill` and `difficulty`.
- Numeric features: `prior_sessions`, `prior_attempts`, `prior_accuracy`, `recent_accuracy` (weighted over up to 3 earlier sessions) and `last_score` from a previous session.
- Categorical features: `skill`, `difficulty` of the proposed session.
- The same learner/skill's *current* outcome is **never** used to predict itself.
- No learner ID, activity ID, demographic information, future row or personal identity becomes a model feature.
- A deterministic `GroupShuffleSplit` divides the training table **by learner ID** (75% train/25% test default), not by random session rows. The holdout includes entirely disjoint learner IDs.
- The baseline is a `DummyClassifier(strategy="prior")`; both model and baseline are evaluated on exactly the same holdout.
- Metrics reported *only after actual predictions*: accuracy, precision, recall, F1, confusion matrix, and ROC-AUC if both test classes exist. None of these prove educational benefit. Calibration is not established.

## Generate and inspect the artifact

Run at the repository root with installed dependencies:

```powershell
python -m src.data_processing generate-demo --output data/synthetic_demo.csv --learners 60 --seed 42
python -m src.model train --data data/synthetic_demo.csv --source synthetic --artifact models/learnadapt_model.joblib --report models/evaluation.json --seed 42
python -m src.model evaluate --report models/evaluation.json
```

- `models/learnadapt_model.joblib`: created by the training CLI, **not bundled by default**. `joblib` files are pickle-based and can execute code during loading; load trusted team-generated artifacts only.
- `models/evaluation.json`: JSON report of the calculated holdout results, created by CLI. `evaluate` **reads an existing report**; it does not claim to re-evaluate on an independent dataset.
- Both generated files are ignored by `.gitignore`; to host a compatible artifact you must explicitly manage trusted model distribution, compatible Python/dependency versions, data permissions and model provenance. The Streamlit sidebar can train synthetic data on demand without persisting a model.

A bundle contains `format_version`, `training_source`, `dataset_fingerprint` (SHA-256 of canonical data), `feature_columns`, `target`, `trained_at_utc`, and the fitted pipeline. The UI refuses artifacts with a different source or dataset fingerprint; it then uses explainable fallback rules.

## Versioning and future evaluation

Increment the internal format version and retrain when feature or preprocessing semantics change. Keep a record of training seed, dependency versions, approved data source, consent/license, calculated metrics and known failures. Synthetic holdout numbers are not real-student performance. Before deployment with real student data, independently evaluate generalization, error patterns, calibration and subgroup fairness; consider temporal evaluation in addition to learner separation.
