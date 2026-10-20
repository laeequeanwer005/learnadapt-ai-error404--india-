# Methodology and reproducibility

## 1. Data unit, target and important distinctions

One record is a **completed practice session** with `attempts` questions and `correct_attempts`. The demonstrated binary prediction target is whether **a subsequent session** scores >=0.60. It does not measure learner mastery, individual question correctness, long-term progress, or causality. Rows have UTC timestamps; the training table is sorted by learner, skill and time.

## 2. Feature generation and leakage prevention

`make_training_table` computes the following for each learner/skill and **strictly before** the target session:

- `prior_sessions`: cumulative session count (excluding target).
- `prior_attempts`: sum of earlier attempts only.
- `prior_accuracy`: earlier correct attempts / earlier total attempts.
- `recent_accuracy`: weighted correctness over up to 3 previous sessions.
- `last_score`: observed correctness in the immediately preceding session.
- `skill`: skill of the proposed session.
- `difficulty`: proposed session's chosen difficulty.

The target is obtained from **the later/current session's** correctness score only after predictor construction. The first session per skill/learner has zero prior evidence and is excluded from model training. Learner IDs do not appear in features; they are only used to construct earlier history and separated evaluation groups. Activity names are not used as predictor features. Time ties within the same learner/skill are **rejected during training** because their chronological order is ambiguous (the demo generator yields distinct times).

## 3. Model selection, baseline and evaluation

The demonstration fits scikit-learn logistic regression with one-hot categoricals and scaled numeric features (reproducible `seed=42`). This interpretable, regularized linear classifier is a simple **prototype choice**, not a proven optimal model. A majority/prior DummyClassifier provides context. The split is `GroupShuffleSplit(test_size=0.25, random_state=seed)` across learner IDs, meaning **different learners** are used for train and test. The held-out learners still have earlier observed sessions used to form their historical features (simulation of known history for a previously unseen model-user); cold-start learner cases are handled by rules instead.

`train_and_evaluate` calculates holdout accuracy, precision, recall, F1 and confusion matrix; ROC-AUC is calculated only if the test set contains both target classes. Saved evaluation numbers arise from actual predictions. There are no invented metrics, independent external validation or calibration claims. Before real deployment, additional chronological validation, error analyses and probability calibration are required. If data is too sparse or training lacks both target classes, training fails explicitly.

## 4. How the ranking is computed

Every proposed activity is defined in `src/recommendation.py` as a **local catalogue suggestion**, not a verified external learning resource. Candidates whose declared **direct or transitive prerequisites** lack at least 8 attempts or 70% observed correctness are removed. The remaining activities are ranked by weighted observed skill gap and recent gap, with a small difficulty-preference adjustment and repeat-activity penalty. When a compatible model has sufficient same-skill history, an uncalibrated classifier probability can adjust the rank; this is *not* a reliable chance-of-success guarantee and is **not** returned as `confidence`.

For a learner without history, the system suggests a simple optional Foundations diagnostic. If no compatible model artifact is available, the engine returns a deterministic `rule_based_fallback`. Reasons include measured historical attempt counts/percentages and explicit notice of any low evidence. Thresholds and ranking coefficients must be treated as *demonstration heuristics* needing educator review.

## 5. Reproducibility procedure

From the repository root with requirements installed:

```powershell
python -m src.data_processing generate-demo --output data/synthetic_demo.csv --learners 60 --seed 42
python -m src.model train --data data/synthetic_demo.csv --source synthetic --artifact models/learnadapt_model.joblib --report models/evaluation.json --seed 42
python -m src.model evaluate --report models/evaluation.json
python -m pytest -q
streamlit run app.py
```

Generated data is explicitly synthetic and has no external license claim. The training script reports the seed, group counts, feature definition, baseline and actual holdout performance. Floating point / solver details may vary by compatible dependency versions; don't compare scores across environments without recording versions. The JSON is **ignored by Git** unless the team deliberately publishes synthetic-only metrics and their provenance.
