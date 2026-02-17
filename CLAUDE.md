# F1 Complete Race Strategy Optimizer - Project Memory

**Status**: Terraform infrastructure complete — ready for `terraform apply`
**Branch**: `claude/f1-strategy-optimizer-lh9No`
**Last Updated**: 2026-02-17

## Project Summary

Build a production-grade, real-time F1 race strategy system delivering lap-by-lap guidance:
- **Pit strategy** (compound, timing, fuel load)
- **Driving mode** (PUSH/BALANCED/CONSERVE)
- **Brake bias** optimization
- **Throttle/braking** patterns
- **Setup recommendations** (wing angle, suspension)

**Core Innovation**: Driver-aware recommendations leveraging 74 years of F1 data (1950-2024), validated against actual race outcomes.

**Target Latency**: <500ms P99 for real-time pit wall decisions

## Core Objectives

1. **Driver Profile Library**: Extract behavioral profiles for 200+ drivers (r > 0.7 correlation)
2. **ML Models**: Train 4 models (tire degradation, fuel consumption, brake bias, driving style)
3. **Real-Time API**: <500ms latency recommendation endpoint
4. **Ground Truth Validation**: ≥70% podium prediction, ≥65% winner prediction
5. **Production MLOps**: DAG-based pipeline, distributed training, operational guarantees

## Success Criteria

### Model Performance
- Tire Degradation MAE: <50ms
- Fuel Consumption RMSE: <0.5 kg/lap
- Brake Bias Accuracy: ±1%
- Driving Style Classifier: ≥75%

### Race Predictions
- Podium Accuracy: ≥70%
- Winner Accuracy: ≥65%
- Pit Timing: ±2 laps (70%+ races)
- Finishing Order Correlation: Spearman >0.75

### Operations
- API P99 Latency: <500ms
- End-to-End Latency: <5s
- System Uptime: 99.5%
- Cost Per Prediction: <$0.001

## High-Level Architecture

```
Data Sources (Ergast, FastF1) → BigQuery → Feature Store
                                    ↓
                            Driver Profiles
                                    ↓
                    ┌───────────────┴────────────────┐
                    ↓                                ↓
            Training Pipeline                Monte Carlo Sim
            (4 models, DAG)                   (10K scenarios)
                    ↓                                ↓
                    └────────────┬───────────────────┘
                                 ↓
                         FastAPI (Cloud Run)
                                 ↓
                    ┌────────────┴─────────────┐
                    ↓                          ↓
                Dashboard                  Monitoring
```

## Tech Stack Summary

- **Data**: Cloud SQL PostgreSQL 15 (operational store), Dataflow (streaming), Ergast + FastF1 (sources)
- **ML**: Distributed containerized training, DAG orchestration, Vertex AI registry
- **Serving**: FastAPI on Cloud Run (serverless)
- **Monitoring**: Centralized logging, Cloud Monitoring, alerting, drift detection
- **Infrastructure**: Terraform-managed GCP resources, autoscaling compute
- **Operational Guarantees**: Task-level failure isolation, cost controls, full audit trail

## Key Constraints

### Performance
- **Latency**: <500ms P99 (pit wall requirement), <5s end-to-end
- **Uptime**: 99.5% during race weekends

### Cost
- **Budget**: $70/month hard cap (set in dev.tfvars and prod.tfvars)
- **Target**: <$0.001 per prediction

### Data
- **Telemetry**: 2018+ only (10Hz), earlier races partial data
- **Temporal Split**: 1950-2022 train, 2023 Q1-Q2 val, 2023 Q3-2024+ test

### Computational
- Monte Carlo requires GPU (10K scenarios)
- Dataflow must handle 10K msgs/sec spikes
- Model quantization required for latency targets

## Critical Bottlenecks

1. **Monte Carlo Simulation**: Compute-intensive (10K strategies × 50 laps)
   - Mitigation: GPU acceleration, caching, reduce to 5K for live races

2. **Real-Time Inference**: Multiple models → latency risk
   - Mitigation: Model caching, quantization (FP32→INT8), TF SavedModel compilation

3. **API Response Time**: 500ms P99 requires sub-100ms model inference
   - Mitigation: Cloud Run autoscaling, aggressive optimization

## Operational Guarantees

### Observability
- Metrics per DAG run and per task (duration, success/failure, retry counts, queue delay)
- Alerting for repeated failures, SLA violations, orchestrator unavailability
- Full pipeline observability across all stages

### Cost Controls
- Resource limits per task (CPU, memory, timeout, max cost)
- Budget thresholds with progressive actions (warning → throttle → emergency stop)
- Cost visibility per DAG run and per stage

### Failure Isolation
- Failures isolated to single task or single DAG run
- Training failures do NOT impact data ingestion or inference
- Retries scoped per task, not per pipeline
- No cascading failures across DAG branches

### Compliance & Auditability
- Immutable records of DAG definitions, task execution metadata, IAM access logs
- 1-year retention for regulatory compliance
- Full reproducibility for any DAG version

## Documentation Structure

- **CLAUDE.md**: This file (high-level strategy, <5000 tokens)
- **docs/data.md**: Data sources, DAG orchestration, operational guarantees
- **docs/training-pipeline.md**: Distributed training, DAG integration, operational guarantees
- **docs/models.md**: ML architectures, training details
- **docs/architecture.md**: System design, deployment
- **docs/metrics.md**: KPIs, targets, validation
- **docs/monitoring.md**: Operational monitoring, alerting
- **docs/roadmap.md**: Timeline, phases, milestones
- **docs/progress.md**: Session log (append-only)

## Component Status

| Phase | Components | Status |
|-------|-----------|--------|
| **Setup** | GCP config, Cloud SQL, IAM, Terraform | Complete — ready for `apply` |
| **Data** | Ingestion (Ergast, FastF1), DAG orchestrator, pipeline logging | DAG fetches data; **write to Cloud SQL missing** |
| **Processing** | Cleaning, feature engineering, driver profiles | Not Started |
| **Training** | 4 ML models (parallel), distributed training infrastructure | Not Started |
| **Serving** | FastAPI, Cloud Run, Monte Carlo sim | Not Started |
| **Ops** | Monitoring, dashboards, alerting | Not Started |

## Terraform Infrastructure (as of 2026-02-17)

### What was done
- **Replaced BigQuery entirely with Cloud SQL (PostgreSQL 15)**
  - Removed `module "bigquery"` and `bigquery.googleapis.com` API
  - Added `google_sql_database_instance` (private IP on VPC, deletion_protection=true, automated backups)
  - Added `google_sql_database` (`f1_data`) and `google_sql_user` (`f1_api`)
  - Password auto-generated via `random_password`, stored in Secret Manager (`google_secret_manager_secret` + `google_secret_manager_secret_version`)
  - `api_sa` IAM binding: `bigquery.dataViewer` → `roles/cloudsql.client`
  - Cloud Run env vars: `BIGQUERY_DATASET` → `DB_HOST`, `DB_NAME`, `DB_PORT`
  - Output: `bigquery_dataset_id` → `cloud_sql_instance_connection_name`
- **Added VPC private IP peering for Cloud SQL**
  - `google_compute_global_address` (VPC_PEERING, /16)
  - `google_service_networking_connection`
  - `servicenetworking.googleapis.com` added to required APIs
  - `networking` module: added `output "network_id"`
- **Added `hashicorp/random ~> 3.0` provider**
- **Removed `var.db_password`** — password is Terraform-generated
- **Fixed `user_labels`** inside `settings {}` (Cloud SQL provider 5.x requirement)
- **Created `dev.tfvars` and `prod.tfvars`** (project: f1optimizer, region: us-central1, budget: $70)
- **Updated README** with GCS state bucket creation commands

### Key files
- `terraform/main.tf` — all infrastructure
- `terraform/variables.tf` — no db_password, no bigquery_location
- `terraform/dev.tfvars` — dev environment (min_instances=0, max=3)
- `terraform/prod.tfvars` — prod environment (min_instances=1, max=10)
- `terraform/modules/networking/main.tf` — exposes `network_id` output

### To deploy
```bash
# One-time: create state bucket
gsutil mb -p f1optimizer gs://f1-optimizer-terraform-state
gsutil versioning set on gs://f1-optimizer-terraform-state

gcloud auth application-default login
terraform -chdir=terraform init
terraform -chdir=terraform apply -var-file=dev.tfvars
```

## Known Gaps (next priorities)

1. **DAG missing write step**: `f1_data_ingestion.py` fetches from Ergast/FastF1 but never writes to Cloud SQL — Cloud SQL will be empty after `terraform apply` until this is implemented
2. **Architecture diagram**: still references BigQuery — needs updating
3. **docker-compose**: `mock-bigquery` service and `BIGQUERY_HOST` env var still reference BigQuery — needs Cloud SQL equivalent for local dev

## Next Steps

1. **Immediate**: `terraform apply -var-file=dev.tfvars` to provision GCP infra
2. **Next**: Add Cloud SQL write step to `f1_data_ingestion.py` DAG
3. **Week 3-4**: Driver profile extraction, feature engineering
4. **Week 5-7**: Model training (4 models in parallel), validation
5. **Week 8+**: API deployment, Monte Carlo sim, dashboard

See docs/ for detailed implementation plans.

---

**Working Principles**:
1. High-signal only in CLAUDE.md; details → docs/
2. Modular context in dedicated documentation files
3. Reference, don't repeat
4. Production-first architecture
5. Data-driven validation

**Compaction Protocol**: Run `/compact` every ~40 messages.
**Session End**: Append summary to docs/progress.md, update session_summary.md.
