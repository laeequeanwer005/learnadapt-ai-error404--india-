# Limitations, safety and responsible AI

**LearnAdapt AI is a hackathon prototype. It is not a certified diagnostic or validated educational intervention.**

## 1. Synthetic data and evaluation

The default generator simulates fictional practice; its accuracy and distributions are created algorithmically. A high score on such data does not imply any predictive validity for real students, let alone improved outcomes. No specific external educational dataset, license, accuracy figure, learning gain, or human-subject experiment is claimed.

## 2. Dataset bias and generalization

Practice data may be biased by difficulty, opportunity, access, device, teacher, curriculum, language or sampling. A model trained on a narrow population may generalize poorly across schools, regions, disabilities or learning contexts. Learner-separated holdout evaluation helps with one form of leakage but does **not** establish external validity, stability, fairness or causal impact. A longitudinal/temporal split should be considered for future validation.

## 3. Low evidence and cold starts

One strong or poor attempt may be noisy. There is no hard diagnosis of ability or mastery. The app provides an optional diagnostic on no history and marks recommendations based on fewer than 3 same-skill sessions as limited evidence. It does not invent probabilities in these cases.

## 4. Uncalibrated uncertainty

The logistic classifier can output numbers from 0 to 1, but **probability calibration has not been evaluated**. Those values are not honest calibrated confidence scores. Evaluation metrics must be calculated from actual held-out predictions; missing ROC-AUC should remain missing when class diversity is insufficient. Human instructors should consider context before acting on a suggestion.

## 5. Prerequisite simplification

The built-in chain (Foundations -> Fractions -> Algebra -> Graphs) and minimum 8 attempts/70% observed correct are **heuristics**, not validated curriculum standards. Actual prerequisite structure varies by grade and curriculum, and a low observed score may reflect measurement error, inaccessible materials or missing context. Known prerequisites are treated conservatively. Students and educators must be able to override the suggestion.

## 6. Privacy and persistence

No names, emails, demographics or school IDs are needed. IDs must still be treated carefully: a pseudonym can remain personal data if linked elsewhere. Uploads are processed **server-side** in a Streamlit deployment; never upload identifiable student records to a public hackathon demo. By default, practice changes are held only in session memory. The optional SQLite database has **no authentication/user-isolation guarantee** and is appropriate only for trusted **single-user local use**, never a public or student-facing deployment. Streamlit Community Cloud local disk and in-memory data can reset, so they are not persistent student-record stores. `.gitignore` excludes local DBs, exported demo CSV and generated model files.

## 7. Model and app security

Only locally generated, trusted Joblib bundles may be loaded. Pickle serialization is unsafe with arbitrary externally obtained model files. Uploaded CSVs should be small, consented and subject to the user's own data-governance rules. The app is not a secure student-information system, has no account management or access controls, and should not make high-stakes educational placements or assessments.

## 8. What remains to validate

- Real dataset identification, provenance, licensing, consent and schema mapping.
- Correct time ordering and split design for the chosen classroom/education setting.
- Performance against credible baselines, external validation, calibration and uncertainty bounds.
- Bias/fairness/error checks for relevant subgroups, with appropriate governance.
- Educator feedback on activities, prerequisite policies and override controls.
- Privacy/security review and a properly authenticated durable database if handling real students.
- Genuine demo screenshots, live hosted deployment and any real measured learning impact.
