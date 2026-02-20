# F1 Strategy Optimizer — ML Team Handoff

**Date**: 2026-02-19
**Project**: `f1optimizer` (GCP)
**Region**: `us-central1`
**Repo**: `bkiritom8/test` | Branch: `main`

---

## Table of Contents

1. [Infrastructure Overview](#1-infrastructure-overview)
2. [What Was Deleted and Why](#2-what-was-deleted-and-why)
3. [Vertex AI Workbench Access](#3-vertex-ai-workbench-access)
4. [GCS Paths](#4-gcs-paths)
5. [Cloud SQL Connection](#5-cloud-sql-connection)
6. [API Endpoint](#6-api-endpoint)
7. [Repo Structure — ml/](#7-repo-structure--ml)
8. [Running Training](#8-running-training)
9. [Compute Profiles](#9-compute-profiles)
10. [Known Gaps and Proposed Solutions](#10-known-gaps-and-proposed-solutions)
11. [Day 1 Quickstart](#11-day-1-quickstart)

---

## 1. Infrastructure Overview

### What Exists (verified 2026-02-19)

| Resource | Name | Status | Notes |
|---|---|---|---|
| Cloud SQL (PostgreSQL 15) | `f1-optimizer-dev` | RUNNABLE | Private IP `10.194.0.3`, tier `db-f1-micro` |
| VPC | `f1-optimizer-network-dev` | Active | Private IP routing for Cloud SQL |
| GCS — Terraform state | `gs://f1-optimizer-terraform-state/` | Active | Do not modify |
| GCS — Cloud Build | `gs://f1optimizer_cloudbuild/` | Active | Build artifacts |
| Cloud Run Job | `db-check` | Deployed | Schema connectivity check |
| Secret Manager | `Github-github-oauthtoken-93539d` | Active | CI/CD only |
| Terraform state | `gs://f1-optimizer-terraform-state/terraform/` | Active | |

### What Does NOT Yet Exist (must be created before ML work)

| Resource | Expected Name | Action Required |
|---|---|---|
| GCS — model artifacts | `gs://f1optimizer-models/` | `gsutil mb -p f1optimizer -l us-central1 gs://f1optimizer-models` |
| GCS — training data | `gs://f1optimizer-training/` | `gsutil mb -p f1optimizer -l us-central1 gs://f1optimizer-training` |
| GCS — FastF1 cache | `gs://f1optimizer-fastf1-cache/` | `gsutil mb -p f1optimizer -l us-central1 gs://f1optimizer-fastf1-cache` |
| Cloud Run service | `f1-strategy-api-dev` | `terraform apply` or manual Cloud Run deploy |
| Service Account | `f1-training-dev@f1optimizer.iam.gserviceaccount.com` | `terraform apply` |
| Secret | `f1-db-password-dev` | Create + populate in Secret Manager (see §5) |
| Pub/Sub topic | `f1-predictions-dev` | `gcloud pubsub topics create f1-predictions-dev --project=f1optimizer` |
| Pub/Sub topic | `f1-alerts-dev` | `gcloud pubsub topics create f1-alerts-dev --project=f1optimizer` |
| Artifact Registry | `f1-optimizer` | `terraform apply` or `gcloud artifacts repositories create` |
| Vertex AI Workbench | `f1-ml-workbench` | `terraform apply -target=google_workbench_instance.ml_workbench` |

---

## 2. What Was Deleted and Why

The following Cloud Run Jobs were targeted for deletion (distributed ingestion workers from a previous architecture):

- `f1-data-coordinator`
- `f1-jolpica-worker`
- `f1-fastf1-worker`

These **were not found in the project** — they were either never deployed or previously removed. The current ingestion architecture is a single Cloud Run Job (`f1-data-ingestion`, managed by Terraform) that runs `pipeline/scripts/run_ingestion.sh` sequentially. No action was needed.

---

## 3. Vertex AI Workbench Access

### Current Status

The Workbench instance `f1-ml-workbench` has **not been provisioned** — Terraform apply is required.

### Provisioning

```bash
# From repo root — review plan first
terraform -chdir=infra/terraform plan -var-file=dev.tfvars -target=google_workbench_instance.ml_workbench

# Apply when ready
terraform -chdir=infra/terraform apply -var-file=dev.tfvars -target=google_workbench_instance.ml_workbench
```

### Specs (from `infra/terraform/vertex_workbench.tf`)

- Machine: `n1-standard-8` (8 vCPU, 30 GB RAM)
- GPU: T4 (1x, NVIDIA)
- Disk: 100 GB SSD
- Auto-shutdown: 60 minutes of idle
- Location: `us-central1`

### Access

Once provisioned:
1. Go to [Vertex AI Workbench](https://console.cloud.google.com/vertex-ai/workbench/instances?project=f1optimizer)
2. Click **Open JupyterLab** next to `f1-ml-workbench`
3. The instance has the ML container pre-installed (`docker/Dockerfile.ml`)

---

## 4. GCS Paths

> **Note**: Buckets do not yet exist. Create them before running any ML code (see §1).

### Training Bucket: `gs://f1optimizer-training/`

```
gs://f1optimizer-training/
├── features/                  # Feature manifests from KFP feature_engineering step
│   └── {run_id}/
│       ├── train_features.parquet
│       └── eval_features.parquet
├── checkpoints/               # Model checkpoints written during training
│   ├── strategy_predictor/{run_id}/
│   └── pit_stop_optimizer/{run_id}/
└── pipelines/                 # Compiled KFP pipeline YAML
    └── f1_strategy_pipeline.yaml
```

### Models Bucket: `gs://f1optimizer-models/`

```
gs://f1optimizer-models/
├── strategy_predictor/
│   └── latest/                # Promoted model artifacts (xgb_model.pkl, lgbm_model.pkl)
└── pit_stop_optimizer/
    └── latest/                # Promoted model artifacts (model.keras, scaler.pkl)
```

### FastF1 Cache: `gs://f1optimizer-fastf1-cache/`

Used by `src/ingestion/fastf1_ingestion.py` to cache telemetry downloads.

---

## 5. Cloud SQL Connection

### Connection Details

| Parameter | Value |
|---|---|
| Host | `10.194.0.3` (private IP, VPC only) |
| Port | `5432` |
| Database | `f1_strategy` |
| Superuser | `postgres` |
| App user | `f1_app` (must be created — see gap below) |
| Instance | `f1optimizer:us-central1:f1-optimizer-dev` |

### Password from Secret Manager

```bash
# Retrieve DB password
gcloud secrets versions access latest \
  --secret="f1-db-password-dev" \
  --project=f1optimizer
```

> **Gap**: The secret `f1-db-password-dev` does not yet exist. Create it:

```bash
# Generate and store password
DB_PASS=$(openssl rand -base64 32)
echo -n "$DB_PASS" | gcloud secrets create f1-db-password-dev \
  --data-file=- \
  --project=f1optimizer \
  --replication-policy=automatic
```

### Connecting (from within VPC / Workbench)

```python
import psycopg2

conn = psycopg2.connect(
    host="10.194.0.3",
    port=5432,
    database="f1_strategy",
    user="postgres",
    password="<from Secret Manager above>"
)
```

### Application User Setup (one-time)

The `f1_app` user referenced throughout ML code needs to be created manually after schema apply:

```sql
-- Run against f1-optimizer-dev after connecting as postgres
CREATE USER f1_app WITH PASSWORD '<same-as-secret>';
GRANT CONNECT ON DATABASE f1_strategy TO f1_app;
GRANT USAGE ON SCHEMA public TO f1_app;
GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA public TO f1_app;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT, INSERT, UPDATE ON TABLES TO f1_app;
```

### Schema

Schema is at `src/database/schema.sql`. Applied automatically by `pipeline/scripts/run_ingestion.sh` on first run, or manually:

```bash
psql -h 10.194.0.3 -U postgres -d f1_strategy -f src/database/schema.sql
```

---

## 6. API Endpoint

### Current Status

The Cloud Run service `f1-strategy-api-dev` is **not yet deployed**.

### Deploying the API

```bash
# Build and push image
gcloud builds submit --config cloudbuild.yaml . --project=f1optimizer

# Deploy (adjust env vars as needed)
gcloud run deploy f1-strategy-api-dev \
  --image us-central1-docker.pkg.dev/f1optimizer/f1-optimizer/api:latest \
  --region us-central1 \
  --project f1optimizer \
  --port 8000 \
  --memory 2Gi \
  --set-env-vars PROJECT_ID=f1optimizer,REGION=us-central1
```

### Available Routes (from `src/api/main.py`)

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `GET` | `/` | Root — service info |
| `POST` | `/predict/strategy` | Race strategy recommendation |
| `POST` | `/predict/pit-stop` | Pit stop timing recommendation |
| `GET` | `/models/info` | Loaded model metadata |

> **Known Gap**: Both `/predict/*` endpoints currently raise `NotImplementedError` for standalone inference — see §10 for fix.

---

## 7. Repo Structure — ml/

```
ml/
├── dag/
│   ├── f1_pipeline.py          # KFP v2 pipeline definition (5-step DAG)
│   ├── pipeline_runner.py      # Compile + submit + monitor pipeline runs
│   └── components/
│       ├── validate_data.py    # Step 1: data quality checks vs Cloud SQL
│       ├── feature_engineering.py  # Step 2: feature extraction to GCS parquet
│       ├── train_strategy.py   # Step 3a: XGBoost+LightGBM ensemble (parallel)
│       ├── train_pit_stop.py   # Step 3b: LSTM sequence model (parallel)
│       ├── evaluate.py         # Step 4: score models, log to Vertex Experiments
│       └── deploy.py           # Step 5: promote to gs://f1optimizer-models/*/latest/
│
├── models/
│   ├── strategy_predictor.py   # XGBoost + LightGBM ensemble
│   │                           #   - train(), evaluate(), predict(), save(), load()
│   │                           #   ⚠️ predict() raises NotImplementedError (standalone mode)
│   ├── pit_stop_optimizer.py   # LSTM sequence model (TF/Keras, GPU-capable)
│   │                           #   - train(), evaluate(), predict(), save(), load()
│   │                           #   ⚠️ evaluate() guards against None model (fixed)
│   └── base_model.py           # Abstract base class for both models
│
├── features/
│   ├── feature_store.py        # Cloud SQL → DataFrame (reads via psycopg2, ADC auth)
│   └── feature_pipeline.py     # Transform raw data → model-ready features
│
├── distributed/
│   ├── cluster_config.py       # 4 named compute configs (see §9)
│   ├── data_sharding.py        # Race-level sharding across workers
│   ├── distribution_strategy.py # TF MirroredStrategy / MultiWorkerMirrored wrappers
│   └── aggregator.py           # Metric aggregation across distributed workers
│
├── tests/
│   ├── run_tests_on_vertex.py  # Submit all tests as Vertex AI Custom Job
│   ├── test_dag.py             # KFP component contract tests
│   ├── test_distributed.py     # Sharding + distribution strategy tests
│   ├── test_features.py        # Feature store / pipeline tests
│   └── test_models.py          # Model train/eval/save/load tests
│
└── evaluation/                 # Placeholder — __init__.py only (no logic yet)
```

### Pipeline DAG

```
validate_data
    └── feature_engineering
            ├── train_strategy  ─────┐
            └── train_pit_stop  ─────┤ (parallel)
                    ├── evaluate (strategy)  ┐
                    └── evaluate (pit_stop)  ┘ (parallel)
                            └── deploy
```

---

## 8. Running Training

### Option A — Full KFP Pipeline (Recommended)

```bash
# Compile pipeline to YAML
python ml/dag/pipeline_runner.py --compile-only

# Compile + submit + monitor
python ml/dag/pipeline_runner.py --run-id $(date +%Y%m%d-%H%M%S)

# Via Cloud Run trigger job (if deployed)
gcloud run jobs execute f1-pipeline-trigger --region=us-central1 --project=f1optimizer
```

Pipeline progress: [Vertex AI Pipelines Console](https://console.cloud.google.com/vertex-ai/pipelines?project=f1optimizer)

### Option B — Individual Model Training (Vertex AI Custom Job)

```bash
# Strategy Predictor
gcloud ai custom-jobs create \
  --region=us-central1 \
  --project=f1optimizer \
  --display-name="strategy-predictor-$(date +%Y%m%d)" \
  --worker-pool-spec=machine-type=n1-standard-8,replica-count=1,container-image-uri=us-central1-docker.pkg.dev/f1optimizer/f1-optimizer/ml:latest \
  --args="python,-m,ml.models.strategy_predictor,--mode,train,--training-bucket,gs://f1optimizer-training,--models-bucket,gs://f1optimizer-models"

# Pit Stop Optimizer (GPU)
gcloud ai custom-jobs create \
  --region=us-central1 \
  --project=f1optimizer \
  --display-name="pit-stop-optimizer-$(date +%Y%m%d)" \
  --worker-pool-spec=machine-type=n1-standard-8,accelerator-type=NVIDIA_TESLA_T4,accelerator-count=1,replica-count=1,container-image-uri=us-central1-docker.pkg.dev/f1optimizer/f1-optimizer/ml:latest \
  --args="python,-m,ml.models.pit_stop_optimizer,--mode,train,--training-bucket,gs://f1optimizer-training,--models-bucket,gs://f1optimizer-models"
```

### Option C — Run Tests on Vertex AI

```bash
python ml/tests/run_tests_on_vertex.py
```

---

## 9. Compute Profiles

Defined in `ml/distributed/cluster_config.py`:

| Config Name | Machine | GPUs | Workers | Use Case |
|---|---|---|---|---|
| `SINGLE_NODE_MULTI_GPU` | `n1-standard-16` | 4x T4 | 1 | Standard training run |
| `MULTI_NODE_DATA_PARALLEL` | `n1-standard-8` | 1x T4 each | 4 | Large dataset sharding |
| `HYPERPARAMETER_SEARCH` | `n1-standard-4` | 0 | 8 | HP sweep (XGBoost only) |
| `CPU_DISTRIBUTED` | `n1-standard-8` | 0 | 4 | Feature engineering |

Select a config in `pipeline_runner.py` by passing `--cluster-config <NAME>`.

---

## 10. Known Gaps and Proposed Solutions

### CRITICAL

| # | Gap | Location | Proposed Fix |
|---|---|---|---|
| C1 | **GCS buckets missing** | Infra | Create 3 buckets: `f1optimizer-models`, `f1optimizer-training`, `f1optimizer-fastf1-cache` |
| C2 | **Secret `f1-db-password-dev` missing** | Secret Manager | `gcloud secrets create f1-db-password-dev ...` (see §5) |
| C3 | **`predict()` not implemented** in both models | `ml/models/*.py:357,378` | Implement standalone `predict()` using saved model artifacts from GCS |
| C4 | **SA `f1-training-dev` missing** | IAM | Run `terraform apply` or create manually with Storage + Vertex AI + CloudSQL roles |
| C5 | **Pub/Sub topics missing** | GCP | Create `f1-predictions-dev`, `f1-alerts-dev` (see §1 commands) |

### HIGH

| # | Gap | Location | Proposed Fix |
|---|---|---|---|
| H1 | **`f1_app` DB user not created** | Cloud SQL | Run SQL grants from §5 |
| H2 | **Bcrypt 4.2.1 breaks passlib 1.7.4** | `docker/requirements-api.txt:29` | Pin `bcrypt>=3.2.0,<4.0.0` |
| H3 | **`ml/scripts/run_training.sh` missing** | `ml/scripts/` | Create directory + script (see §8 for equivalent commands) |
| H4 | **Artifact Registry repo missing** | GCP | `terraform apply` or `gcloud artifacts repositories create f1-optimizer --repository-format=docker --location=us-central1 --project=f1optimizer` |

### MEDIUM

| # | Gap | Location | Proposed Fix |
|---|---|---|---|
| M1 | Ray imported in `ml/training/distributed_trainer.py` but not in ML container | `docker/requirements-ml.txt` | Add `ray[default]==2.9.1` to ML requirements or remove Ray usage |
| M2 | LightGBM version drift (4.2.0 vs 4.3.0) | requirements files | Pin to `lightgbm==4.3.0` everywhere |
| M3 | `ml/evaluation/` is empty | `ml/evaluation/` | Move post-training analysis from `ml/dag/components/evaluate.py` |
| M4 | KFP components hardcoded to `:latest` image | `ml/dag/components/*.py` | Pin to git SHA or semver for rollback capability |
| M5 | Hardcoded dev instance name across ML code | Multiple files | Pass `INSTANCE_CONNECTION_NAME` as env var |

---

## 11. Day 1 Quickstart

Complete these steps in order on your first day.

### Step 0 — Clone and configure

```bash
git clone https://github.com/bkiritom8/test.git
cd test
git checkout main

# Install ML dependencies (local dev)
pip install -r requirements-f1.txt

# Configure GCP credentials
gcloud auth application-default login
gcloud config set project f1optimizer
```

### Step 1 — Create missing GCS buckets

```bash
for bucket in f1optimizer-models f1optimizer-training f1optimizer-fastf1-cache; do
  gsutil mb -p f1optimizer -l us-central1 "gs://${bucket}/"
  echo "Created gs://${bucket}/"
done
```

### Step 2 — Create and populate DB secret

```bash
DB_PASS=$(openssl rand -base64 32)
echo -n "$DB_PASS" | gcloud secrets create f1-db-password-dev \
  --data-file=- \
  --project=f1optimizer \
  --replication-policy=automatic
echo "Secret created. Password: $DB_PASS"
# Store this password securely — you'll need it for Cloud SQL setup
```

### Step 3 — Create Pub/Sub topics

```bash
gcloud pubsub topics create f1-predictions-dev --project=f1optimizer
gcloud pubsub topics create f1-alerts-dev --project=f1optimizer
```

### Step 4 — Create Artifact Registry repo (if absent)

```bash
gcloud artifacts repositories create f1-optimizer \
  --repository-format=docker \
  --location=us-central1 \
  --project=f1optimizer \
  --description="F1 Strategy Optimizer container images"
```

### Step 5 — Build and push ML image

```bash
gcloud builds submit --config cloudbuild.yaml . --project=f1optimizer
```

### Step 6 — Provision Workbench (optional, for interactive work)

```bash
terraform -chdir=infra/terraform plan -var-file=dev.tfvars -target=google_workbench_instance.ml_workbench
# Review plan, then:
terraform -chdir=infra/terraform apply -var-file=dev.tfvars -target=google_workbench_instance.ml_workbench
```

Open [Vertex AI Workbench Console](https://console.cloud.google.com/vertex-ai/workbench/instances?project=f1optimizer) → Click **Open JupyterLab**.

### Step 7 — Run ingestion to populate Cloud SQL

```bash
gcloud run jobs execute f1-data-ingestion --region=us-central1 --project=f1optimizer
# Monitor logs
gcloud logging read 'resource.type="cloud_run_job" AND resource.labels.job_name="f1-data-ingestion"' \
  --project=f1optimizer --limit=50 --format="value(textPayload)"
```

### Step 8 — Compile and run the ML pipeline

```bash
# Compile KFP pipeline
python ml/dag/pipeline_runner.py --compile-only

# Submit to Vertex AI Pipelines
python ml/dag/pipeline_runner.py --run-id $(date +%Y%m%d-%H%M%S)
```

Monitor at: https://console.cloud.google.com/vertex-ai/pipelines?project=f1optimizer

### Step 9 — Verify model artifacts were promoted

```bash
gsutil ls gs://f1optimizer-models/strategy_predictor/latest/
gsutil ls gs://f1optimizer-models/pit_stop_optimizer/latest/
```

### Step 10 — Run ML tests

```bash
python ml/tests/run_tests_on_vertex.py
```

---

## Quick Reference

| What | How |
|---|---|
| DB password | `gcloud secrets versions access latest --secret=f1-db-password-dev --project=f1optimizer` |
| Cloud SQL host | `10.194.0.3:5432` |
| DB name | `f1_strategy` |
| Training bucket | `gs://f1optimizer-training/` |
| Models bucket | `gs://f1optimizer-models/` |
| Pipeline trigger | `python ml/dag/pipeline_runner.py --run-id <id>` |
| Build all images | `gcloud builds submit --config cloudbuild.yaml . --project=f1optimizer` |
| Check ingestion logs | `gcloud logging read 'resource.labels.job_name="f1-data-ingestion"' --project=f1optimizer --limit=50` |
| Vertex AI console | https://console.cloud.google.com/vertex-ai?project=f1optimizer |
| Cloud Run console | https://console.cloud.google.com/run?project=f1optimizer |

---

*Generated 2026-02-19. Update `docs/progress.md` after each ML session.*
