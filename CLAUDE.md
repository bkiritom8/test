# F1 Complete Race Strategy Optimizer - Project Memory

**Status**: Initial setup phase
**Branch**: `claude/f1-strategy-optimizer-lh9No`
**Last Updated**: 2026-02-14

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

1. **Driver Profile Library**: Extract behavioral profiles for 200+ drivers (aggression, consistency, pressure response) with r > 0.7 correlation
2. **ML Models**: Train 4 models (tire degradation, fuel consumption, brake bias, driving style) meeting accuracy targets
3. **Real-Time API**: <500ms latency recommendation endpoint on Cloud Run
4. **Ground Truth Validation**: ≥70% podium prediction, ≥65% winner prediction on unseen races
5. **Production MLOps**: Streaming pipeline, auto-retraining, drift detection, monitoring

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
┌─────────────┐     ┌──────────────┐     ┌────────────┐
│  Ergast API │────>│   BigQuery   │────>│  Feature   │
│  FastF1 SDK │     │  (150GB raw) │     │   Store    │
└─────────────┘     └──────────────┘     └────────────┘
                                                │
                    ┌───────────────────────────┘
                    │
┌─────────────┐     ▼              ┌──────────────┐
│  Telemetry  │  ┌──────────┐     │ Driver       │
│  (Pub/Sub)  │─>│ Dataflow │────>│ Profiles     │
└─────────────┘  └──────────┘     └──────────────┘
                                                │
                    ┌───────────────────────────┤
                    │                           │
                    ▼                           ▼
            ┌──────────────┐          ┌──────────────┐
            │  Training    │          │  Monte Carlo │
            │  Pipeline    │          │  Simulator   │
            │ (Distributed)│          └──────────────┘
            └──────────────┘                   │
                    │                          │
                    ▼                          │
            ┌──────────────┐                   │
            │  ML Models   │                   │
            │  (4 models)  │                   │
            └──────────────┘                   │
                    │                          │
                    └───────────┬──────────────┘
                                ▼
                        ┌──────────────┐
                        │   FastAPI    │
                        │  (Cloud Run) │
                        └──────────────┘
                                │
                        ┌───────┴────────┐
                        ▼                ▼
                ┌──────────────┐  ┌──────────────┐
                │   Dashboard  │  │  Monitoring  │
                │    (React)   │  │  (GCP Cloud) │
                └──────────────┘  └──────────────┘
```

## Tech Stack

### Data Layer
- **Warehouse**: Google BigQuery (partitioned by race_year, race_date)
- **Streaming**: Apache Beam / Dataflow (10K msgs/sec)
- **Feature Store**: BigQuery native integration
- **Sources**: Ergast API (1950-2024), FastF1 (2018-2024, 10Hz telemetry)
- **Artifact Storage**: GCS buckets (models, metrics, checkpoints)

### ML Stack
- **Training Pipeline**: Distributed containerized jobs, autoscaling workers
- **Orchestration**: Vertex AI Pipelines, Terraform-managed infrastructure
- **Compute**: Autoscaling worker pools (CPU/GPU abstracted)
- **Registry**: Vertex AI MLflow-compatible
- **Libraries**: pandas, NumPy, scikit-learn, TensorFlow, Dataflow SDK
- **Security**: IAM RBAC, encrypted inter-worker transport (HTTPS)

### Serving
- **API**: FastAPI (async, type hints)
- **Container**: Docker
- **Deployment**: Google Cloud Run (serverless, autoscaling)
- **Dashboard**: React 18 + Next.js (TypeScript), Recharts, Tailwind CSS

### Monitoring
- **Logging**: Cloud Logging
- **Metrics**: Cloud Monitoring
- **Alerting**: Custom Pub/Sub + Slack integration
- **Drift Detection**: Weekly automated validation

## Repository Structure

```
f1-strategy-optimizer/
├── CLAUDE.md              # This file - persistent memory
├── README.md              # Quick start guide
├── requirements.txt       # Python dependencies (pinned)
├── setup.sh              # Environment & GCP setup
├── Dockerfile            # Container for Cloud Run
│
├── data/                 # Data ingestion & preprocessing
│   ├── download.py       # Ergast/FastF1 fetchers
│   ├── preprocess.py     # Cleaning & feature engineering
│   └── schema.sql        # BigQuery table definitions
│
├── drivers/              # Driver profile extraction
│   ├── extract_profiles.py
│   └── profiles.json     # Output: 200+ driver profiles
│
├── models/               # ML model training & serving
│   ├── tire_degradation.py
│   ├── fuel_consumption.py
│   ├── brake_bias.py
│   ├── driving_style.py
│   └── train.py          # Unified training pipeline
│
├── simulation/           # Monte Carlo race simulator
│   ├── race_simulator.py # 10K scenario engine
│   └── optimizer.py      # Strategy selection logic
│
├── pipeline/             # Real-time streaming & training
│   ├── dataflow_job.py   # Beam pipeline
│   ├── feature_extraction.py
│   └── training/         # Distributed training infrastructure
│       ├── orchestrator.py    # Training job scheduler
│       ├── worker.py          # Training worker container
│       └── config.yaml        # Training pipeline config
│
├── serving/              # API deployment
│   └── api.py            # FastAPI endpoints
│
├── dashboard/            # Web interface
│   └── src/              # React components
│
├── monitoring/           # Production monitoring
│   ├── alerts.yaml       # Alert configurations
│   └── dashboards.json   # Cloud Monitoring dashboards
│
├── notebooks/            # Analysis & validation
│   └── validation.ipynb  # Ground truth comparisons
│
├── tests/                # Unit & integration tests
│   ├── test_models.py
│   ├── test_pipeline.py
│   └── test_api.py
│
├── infrastructure/       # Terraform & deployment
│   ├── terraform/        # GCP infrastructure as code
│   │   ├── training.tf   # Training compute resources
│   │   ├── storage.tf    # BigQuery, GCS buckets
│   │   └── iam.tf        # Service accounts, roles
│   └── k8s/             # Kubernetes manifests (if needed)
│
└── docs/                 # Detailed documentation
    ├── data.md           # Data sources, splits, management
    ├── models.md         # ML architectures, training
    ├── architecture.md   # System design, deployment
    ├── metrics.md        # KPIs, targets, validation
    ├── monitoring.md     # Operational monitoring, alerting
    ├── training-pipeline.md  # Distributed training infrastructure
    ├── roadmap.md        # Timeline, phases, milestones
    ├── progress.md       # Session log (append-only)
    └── (session_summary.md at repo root)
```

## Active Components & Status

| Component | Status | Owner | Notes |
|-----------|--------|-------|-------|
| Data Ingestion | Not Started | Data Engineer | Ergast API + FastF1 |
| BigQuery Setup | Not Started | Data Architect | Schema definition needed |
| Driver Profiles | Not Started | Data Scientist | 200+ profiles target |
| Feature Store | Not Started | Data Engineer | f1_features table |
| Training Infrastructure | Not Started | MLOps Engineer | Terraform, Vertex AI Pipelines |
| Training Pipeline | Not Started | ML Engineer | Distributed, containerized jobs |
| ML Models (4) | Not Started | ML Engineer | Train all 4 in parallel |
| Monte Carlo Sim | Not Started | ML Engineer | 10K scenarios |
| FastAPI Server | Not Started | Backend Engineer | Deploy to Cloud Run |
| React Dashboard | Not Started | Frontend Engineer | 4 components |
| Monitoring | Not Started | DevOps | Cloud Monitoring setup |
| Testing | Not Started | QA | 80%+ coverage target |

## Known Bottlenecks & Mitigations

### Critical (🔴 High Severity)

1. **Monte Carlo Simulation**: 10K strategies × 50 lap forward = compute-intensive
   - **Mitigation**: GPU acceleration, reduce to 5K for live races, cache common scenarios

2. **Real-Time Inference Latency**: Multiple models → slow predictions
   - **Mitigation**: Model caching, quantization (FP32→INT8), batch predictions, TF SavedModel compilation

3. **API Response Time**: 500ms P99 target requires sub-100ms model inference
   - **Mitigation**: Cloud Run autoscaling, load testing, aggressive optimization

### Medium-High (🟠)

4. **Model Training**: XGBoost + LSTM on 20M laps = high compute
   - **Mitigation**: Vertex AI distributed training, cache preprocessed features

### Medium (🟡)

5. **Data Ingestion**: 150GB initial load slow
   - **Mitigation**: BigQuery partitioning by year, incremental post-race updates

6. **Driver Profile Extraction**: 200+ drivers × 1,300 races = complex aggregation
   - **Mitigation**: Vectorized pandas/NumPy, parallel processing per driver

7. **Feature Store Consistency**: BigQuery definitions vs code drift
   - **Mitigation**: Schema versioning, automated tests, feature registry

8. **Dataflow Pipeline**: 10K msgs/sec → memory overhead
   - **Mitigation**: Window management, early stopping, aggressive GC

## Hard Constraints

### Data
- Telemetry only available 2018+ (10Hz); earlier races have partial data
- Ergast API and FastF1must remain available (external dependency)
- Data completeness <95% requires imputation strategy

### Performance
- **P99 Latency**: Must be <500ms (pit wall real-time requirement)
- **End-to-End**: <5s from telemetry to dashboard
- **Uptime**: 99.5% during race weekends (non-negotiable)

### Cost
- **Budget**: ~$200-250/month GCP spend
- **Alert**: $300/month triggers cost review
- **Target**: <$0.001 per prediction for SaaS viability

### Computational
- Monte Carlo with 10K scenarios requires GPU (CPU too slow)
- Dataflow must handle 10K msgs/sec spikes during live telemetry
- Model quantization required to meet latency targets

### Regulatory
- F1 regulations may change (temporal train/test split mitigates)
- Historical data remains relevant only if rules stable
- Commercial use requires FIA licensing

## Data Splits (Temporal to Prevent Leakage)

| Split | Size | Period | Purpose |
|-------|------|--------|---------|
| Train | ~140GB | 1950-2022 (1,300+ races) | Extract profiles, train models |
| Validation | ~5GB | 2023 Q1-Q2 (10 races) | Hyperparameter tuning |
| Test | ~10GB | 2023 Q3-Q4 + 2024 (20+ races) | Final evaluation, ground truth |

## Immediate TODOs (Week 1-2)

### Setup & Infrastructure
- [ ] Run `setup.sh` to configure GCP project
- [ ] Create BigQuery dataset `f1_strategy`
- [ ] Set up service accounts and IAM roles (least-privilege)
- [ ] Configure Cloud Run, Dataflow, Vertex AI access
- [ ] Provision GCS buckets for training artifacts
- [ ] Deploy Terraform infrastructure (training compute, storage)

### Data Ingestion
- [ ] Download Ergast API data (1950-2024) → BigQuery raw tables
- [ ] Download FastF1 telemetry (2018-2024) → BigQuery raw tables
- [ ] Define BigQuery schema (`data/schema.sql`)
- [ ] Validate data completeness (expect ~95%+)

### Documentation
- [x] Create CLAUDE.md (this file)
- [ ] Create docs/data.md
- [ ] Create docs/models.md
- [ ] Create docs/architecture.md
- [ ] Create docs/training-pipeline.md
- [ ] Create docs/metrics.md
- [ ] Create docs/monitoring.md
- [ ] Create docs/roadmap.md
- [ ] Initialize docs/progress.md
- [ ] Create session_summary.md

### Development Environment
- [ ] Install dependencies: `pip install -r requirements.txt`
- [ ] Test Ergast API connectivity
- [ ] Test FastF1 library import
- [ ] Set up local development environment

## Key Decisions & Rationale

### Why GCP over AWS/Azure?
- Integrated BigQuery + Dataflow + Vertex AI ecosystem
- Superior time-series handling in BigQuery
- Cost-effective for our scale (~$200/month vs $400+ on AWS)

### Why Monte Carlo vs Optimization Solver?
- Race dynamics are stochastic (safety cars, crashes, weather)
- Need probability distributions, not point estimates
- 10K scenarios capture uncertainty better than deterministic solver

### Why 4 Separate Models vs End-to-End?
- Modularity: Replace/retrain models independently
- Explainability: Feature importance per task
- Performance: Smaller models faster than monolithic
- Validation: Test each component separately

### Why Temporal Split vs Random?
- Prevents data leakage (no future info in training)
- Reflects real deployment (new races arrive sequentially)
- Tests generalization across eras (regulation changes)

## Working Principles

1. **High-Signal Only**: CLAUDE.md stays ≤5000 tokens; details → docs/
2. **Modular Context**: Large content in dedicated docs/ files
3. **Session Hygiene**: Append to docs/progress.md after each session
4. **Reference, Don't Repeat**: Point to files instead of restating
5. **Spec-Driven**: Follow project specification strictly
6. **Data-Driven Validation**: Every model validated against ground truth
7. **Production-First**: Build for real-time deployment from day one

## Quick Reference

- **Project Spec**: See original specification (stored in docs/)
- **Data Details**: docs/data.md
- **Model Architectures**: docs/models.md
- **System Design**: docs/architecture.md
- **Training Pipeline**: docs/training-pipeline.md
- **KPIs & Metrics**: docs/metrics.md
- **Monitoring Setup**: docs/monitoring.md
- **Timeline**: docs/roadmap.md
- **Session Log**: docs/progress.md
- **Latest Context**: session_summary.md

## Next Steps

1. Complete documentation structure (docs/ files)
2. Run setup.sh for GCP configuration
3. Begin data ingestion (Week 1-2)
4. Start driver profile extraction (Week 3-4)
5. Feature engineering and model training (Week 5-7)

---

**Compaction Protocol**: Run `/compact Focus on code, architecture decisions, and model behavior` every ~40 messages.
**Session End**: Append summary to docs/progress.md, update session_summary.md.
