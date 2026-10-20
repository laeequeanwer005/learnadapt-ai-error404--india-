# Safe integration with your existing GitHub repository

## What was reviewed

The team supplied an original `app.py` (five-section Streamlit frontend), `requirements.txt` and `README.md`. The original UI used a blue-and-white design, Plotly charts, hardcoded skill percentages and synthetic example history, session-only entries, and recommendation placeholders. Its requirements file duplicated `streamlit` and `pandas`. The original README already contained team identity, problem description and planned architecture.

The generated project is **a separate replacement candidate**. Your GitHub account/repository has **not been accessed, modified or deployed**. No claims about its current branch contents are made beyond your own description.

## Keep vs replace vs move vs create

| Existing path | Proposed action | Notes |
|---|---|---|
| `app.py` | **Review and replace/merge** | Keeps all five tabs, overall blue/light aesthetic and Plotly, replaces hardcoded metrics and disabled recommendations with functional integrations. Back up your current version first. |
| `requirements.txt` | **Review and replace** | Removes duplicates, adds numpy, sklearn, joblib and pytest. |
| `README.md` | **Review and replace/merge** | Retains team, challenge and aim, updates status and instructions to match functional implementation. |
| Root `config.toml` | **Move/reconcile** | The functional Streamlit theme file is `.streamlit/config.toml` — confirm any custom settings in your original root file before deleting it. |
| `.gitignore` | Create/merge | Protects generated artifacts, secrets and local DB. |
| `src/` | Create | Reusable processing, training and recommendation modules. |
| `data/` | Create | Documentation only until demo CSV is generated. |
| `models/` | Create | Documentation only; trusted Joblib artifact generated on your machine or via explicit UI action. |
| `docs/` | Create | Architecture, methodology, limitations and this guide. |
| `tests/` | Create | Deterministic automated tests. |
| `screenshots/` | Create | Screenshot instructions; capture actual UI after running locally. |
| `assets/` | Create | Local LearnAdapt AI PNG branding files. |

## Suggested commands (Windows PowerShell)

1. Download/expand the ZIP to a **separate folder**. Check the README and inspect differences with your existing repo, especially `app.py`.
2. Open your existing repo locally. Make a backup branch first:

```powershell
git checkout main
git pull
git checkout -b feature/learnadapt-backend
```

3. Copy the approved generated files to the **same paths** in the checked-out repository root. Do not nest `LearnAdapt-AI/LearnAdapt-AI/` accidentally. Compare/merge any team-specific frontend customizations before replacing `app.py`.
4. If your old `config.toml` was at the repo root, move relevant settings into `.streamlit/config.toml` (not both); verify them in the Streamlit UI.
5. Run the setup, tests, synthetic training and Streamlit app commands from the top-level README.
6. Inspect changes before committing:

```powershell
git status
git diff --stat
git add app.py requirements.txt README.md .gitignore .streamlit src data models docs tests screenshots assets
git commit -m "Integrate LearnAdapt AI validation, ML and recommendations"
git push -u origin feature/learnadapt-backend
```

7. Review and merge the feature branch into `main` through GitHub after teammate approval. `git add models` does not include ignored generated `.joblib`/`evaluation.json` files automatically. No public URL or merged/deployed status should be claimed until verified.

**Important:** The optional local SQLite flag `LEARNADAPT_ENABLE_SQLITE=1` is only for single-user local testing; do not enable on a publicly hosted Streamlit app. Cloud deployments should use session-only updates until authenticated durable persistence is implemented.
