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

- **Data**: BigQuery (warehouse), Dataflow (streaming), Ergast + FastF1 (sources)
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
- **Budget**: $200-250/month total ($120 training, $50 data, $30 inference, $30 storage, $20 monitoring)
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
| **Setup** | GCP config, BigQuery, IAM, Terraform | Not Started |
| **Data** | Ingestion (Ergast, FastF1), DAG orchestrator, pipeline logging | Not Started |
| **Processing** | Cleaning, feature engineering, driver profiles | Not Started |
| **Training** | 4 ML models (parallel), distributed training infrastructure | Not Started |
| **Serving** | FastAPI, Cloud Run, Monte Carlo sim | Not Started |
| **Ops** | Monitoring, dashboards, alerting | Not Started |

## Next Steps

1. **Week 1-2**: GCP setup, data ingestion, BigQuery schema
2. **Week 3-4**: Driver profile extraction, feature engineering
3. **Week 5-7**: Model training (4 models in parallel), validation
4. **Week 8+**: API deployment, Monte Carlo sim, dashboard

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
