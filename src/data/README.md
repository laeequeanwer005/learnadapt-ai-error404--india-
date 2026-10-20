# Data contract and provenance

**No real student dataset is included.** The project uses reproducible fictional data until a separate, de-identified, permitted dataset is supplied. All `SYN-*` learner identifiers from the generator are synthetic. No dataset license or quality is assumed for uploads. The data folder intentionally contains no confidential student records.

## CSV format

A row is one **completed practice session** and can summarize multiple question attempts.

| Column | Required? | Data type and semantics |
|---|---|---|
| `learner_id` | Yes | Non-empty opaque/pseudonymous learner ID (not a model feature) |
| `skill` | Yes | Non-empty skill label (match the activity catalogue if you want supported recommendations) |
| `activity_id` | Yes | Non-empty ID of the completed activity |
| `difficulty` | Yes | `Beginner`, `Intermediate` or `Advanced` |
| `attempts` | Yes | Strictly positive integer per practice session |
| `correct_attempts` | Yes | Integer, from 0 through attempts |
| `timestamp` | Yes | Parseable date/time; UTC is used for display (naive timestamps assumed UTC) |
| `score` | Optional, derived | Fraction 0–1, checked against `correct_attempts / attempts`; derived if absent |
| `duration_minutes` | Optional | Positive finite number when present; blank is permitted |
| `prerequisite_skill` | Optional | Skill name or blank; contributes to conservative prerequisite checks |

Example, **fictional schema only** (do not treat as actual participant records):

```csv
learner_id,skill,activity_id,difficulty,attempts,correct_attempts,score,duration_minutes,timestamp,prerequisite_skill
SYN-EXAMPLE,Foundations,FND-101,Beginner,5,4,0.8,8,2026-10-01T09:00:00Z,
SYN-EXAMPLE,Fractions,FRA-101,Beginner,5,2,0.4,10,2026-10-02T09:00:00Z,Foundations
```

## Generate deterministic synthetic demonstration CSV

Run from the repository root (Python dependencies installed):

```powershell
python -m src.data_processing generate-demo --output data/synthetic_demo.csv --learners 60 --seed 42
```

Generates `60 × 4 skills × 6 rounds = 1,440` fictional practice sessions with reproducible random outcomes and explicit skills and prerequisites. The application also generates the same data in memory when no CSV is supplied. `data/synthetic_demo.csv` is **generated**, not included or tracked by Git.

## CSV validation and feature creation

`load_csv` reads and `validate_records` rejects missing required columns, empty identifiers, bad difficulty, invalid counts, score/count inconsistency, impossible/negative durations and bad timestamps. Missing optional scores are derived. Missing optional durations remain missing (not zero). Exactly identical duplicate rows are removed with a warning. Extra unsupported columns are ignored **with an explicit warning**; other rows are not silently discarded. Model training rejects ambiguous same-learner/same-skill timestamp ties. Dates are converted to UTC. Any ambiguous source semantics must be resolved before use.

`make_training_table` builds next-session labels **after** selecting previous-session features. Raw `learner_id` is used only to group histories and split evaluation, **never as a predictive feature**. Group-separated holdout evaluation prevents the same learner appearing on both sides of the split. Features based on prior history within held-out learners simulate a learner with already recorded sessions; the first cold-start session has no sufficient history for a model prediction.

## Bringing a real CSV

1. Verify the actual source, schema, licensing and explicit permission/consent. Never invent a public-dataset mapping.
2. Replace direct names, email addresses, student IDs or any personally identifiable information with opaque synthetic/pseudonymous IDs. A pseudonym is still potentially personal data; treat it as such.
3. Match the required session-level schema with **documented** transformations (for question-level events, aggregate only when statistically defensible).
4. Test with `python -m pytest -q` and upload only permitted records. On public cloud, uploads are processed on the app server—prefer entirely fictional data for judging.
5. If locally training, run `python -m src.model train --data path\to\permitted.csv --source real --artifact models/learnadapt_model.joblib --report models/evaluation.json`. This does **not** independently verify the dataset's origin or licensing.

If a data provider does not supply enough sequential sessions or both classes in the training subset, stop rather than silently fabricate a training label. Source names, licenses, collection dates and transformation scripts should be added here if actual external data is ever adopted.
