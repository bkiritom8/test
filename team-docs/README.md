# Team Documentation

This directory contains **internal team documentation** for the F1 Strategy Optimizer project.
These files are meant for developers working on the project and are **not part of the course submission**.

## Contents

| File | Purpose |
|---|---|
| `ML_HANDOFF.md` | High-level ML handoff — GCP setup, training pipeline, model status |
| `ml_module_handoff.md` | Detailed ML module handoff — all Vertex AI components, infra, commands |
| `DEV_SETUP.md` | Developer onboarding guide — environment setup, authentication, common tasks |
| `SETUP.md` | Quick setup reference for new contributors |
| `session_summary.md` | Session-by-session progress log (gitignored — local only) |

## For the Course Submission

The grader-facing submission is in [`Data-Pipeline/`](../Data-Pipeline/).
Start there for pipeline setup, DVC reproducibility, and Airflow DAG documentation.

## What Is NOT Here

- Architecture decisions → [`docs/architecture.md`](../docs/architecture.md)
- Model details → [`docs/models.md`](../docs/models.md)
- Bias analysis → [`docs/bias.md`](../docs/bias.md)
- Pipeline submission → [`Data-Pipeline/README.md`](../Data-Pipeline/README.md)
