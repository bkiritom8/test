# F1 Strategy Optimizer — Data Pipeline

**Course submission for MLOps Data Pipeline assignment.**
This directory is fully self-contained. All pipeline code, tests, and documentation are here.

---

## Overview

End-to-end MLOps data pipeline for 76 years of F1 data (1950–2026):

- **Ingestion** — Jolpica API (historical results) + FastF1 (10Hz telemetry)
- **Preprocessing** — CSV normalization, Parquet conversion, timedelta handling
- **Validation** — Schema checks and data quality assertions
- **Anomaly Detection** — Outlier detection with optional Slack alerts
- **Feature Engineering** — Lap-by-lap ML state vectors
- **Bias Analysis** — Representation slicing by era, team, circuit, weather

---

## Architecture

```
Jolpica API ──► src/ingestion/ergast_ingestion.py ──► data/raw/jolpica/
FastF1 API  ──► src/ingestion/fastf1_ingestion.py ──► data/raw/fastf1/
                                │
                pipeline/scripts/csv_to_parquet.py
                                │
                        data/processed/  (Parquet)
                                │
              ┌─────────────────┼─────────────────┐
              ▼                 ▼                  ▼
    scripts/validate_data  scripts/anomaly_    ml/features/
                           detection.py        feature_pipeline.py
                                                    │
                                            data/features/
                                                    │
                                   scripts/bias_analysis.py
                                                    │
                                    logs/bias_report.json
```

---

## Folder Structure

```
Data-Pipeline/
├── dags/
│   └── f1_pipeline.py          Airflow DAG (weekly, 7 tasks)
├── data/
│   └── .gitkeep                Data lives in data/ at repo root (gitignored)
├── scripts/
│   ├── validate_data.py        Schema + data quality validation
│   ├── anomaly_detection.py    Outlier and missing-value detection
│   ├── bias_analysis.py        Representation bias via data slicing
│   └── expectations/
│       └── .gitkeep            GE-style JSON suites written here at runtime
├── tests/
│   ├── __init__.py
│   ├── test_ingestion.py       Unit tests for Jolpica ingestion
│   ├── test_csv_to_parquet.py  Unit tests for CSV conversion
│   └── test_preprocessing.py  Unit tests for validation logic
├── logs/
│   └── .gitkeep                anomaly_report.json, bias_report.json written here
├── dvc.yaml                    Pipeline stages for this directory
└── README.md                   This file
```

Also at repo root:
```
dvc.yaml                        Root-level DVC pipeline (full repo stages)
.dvcignore                      DVC ignore rules
src/ingestion/
├── ergast_ingestion.py         Jolpica API client
└── fastf1_ingestion.py         FastF1 telemetry client
data/
└── .gitignore                  Keeps data/ out of git (DVC manages it)
docs/bias.md                    Bias findings and mitigation documentation
```

---

## Prerequisites

- Python 3.10+
- (Optional) GCP account for GCS data access

```bash
# Install all dependencies
pip install -r requirements-f1.txt

# Install pipeline-specific tools
pip install dvc apache-airflow pandas pyarrow numpy tenacity
```

---

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/bkiritom8/F1-Strategy-Optimizer.git
cd F1-Strategy-Optimizer

# 2. Install dependencies
pip install -r requirements-f1.txt

# 3. (Optional) Initialize DVC with GCS remote
dvc init
dvc remote add -d gcs_remote gs://f1optimizer-data-lake
dvc remote modify gcs_remote credentialpath ~/.config/gcloud/application_default_credentials.json

# 4. Pull existing processed data from GCS (if you have access)
dvc pull
```

---

## Running the Pipeline

### Full pipeline (recommended — runs all stages in dependency order)

```bash
dvc repro
```

### Individual stages

```bash
dvc repro ingest_jolpica      # Fetch historical data from Jolpica API
dvc repro ingest_fastf1       # Fetch telemetry from FastF1
dvc repro preprocess          # Convert CSVs → Parquet
dvc repro validate            # Run schema validation
dvc repro detect_anomalies    # Run anomaly detection
dvc repro build_features      # Build ML feature vectors
dvc repro bias_analysis       # Run bias analysis
```

### Without GCP (local mode — using raw/ directory)

```bash
export USE_LOCAL_DATA=true

# If you have the raw CSVs locally in raw/
python pipeline/scripts/csv_to_parquet.py --input-dir raw/ --bucket local

# Then run validation directly
python Data-Pipeline/scripts/validate_data.py --data-dir data/processed
python Data-Pipeline/scripts/anomaly_detection.py --data-dir data/processed
python Data-Pipeline/scripts/bias_analysis.py --data-dir data/processed
```

### Airflow

```bash
# Initialize Airflow database
airflow db init

# Start the scheduler and webserver
airflow scheduler &
airflow webserver --port 8080 &

# Trigger the pipeline
airflow dags trigger f1_data_pipeline

# Or use the web UI at http://localhost:8080
```

The DAG (`Data-Pipeline/dags/f1_pipeline.py`) runs weekly and includes:

| Task | Description |
|---|---|
| `fetch_jolpica` | Jolpica API → data/raw/jolpica/ |
| `fetch_fastf1` | FastF1 → data/raw/fastf1/ |
| `validate_raw` | Check raw files exist and are non-empty |
| `preprocess` | CSV → Parquet normalization |
| `detect_anomalies` | Outlier + missing value checks |
| `build_features` | Lap-by-lap ML feature vectors |
| `bias_analysis` | Representation bias report |

---

## Running Tests

```bash
# All Data-Pipeline tests
pytest Data-Pipeline/tests/ -v

# With coverage
pytest Data-Pipeline/tests/ -v --cov=Data-Pipeline/scripts --cov=src/ingestion --cov-report=term-missing

# All project tests
pytest tests/ ml/tests/ Data-Pipeline/tests/ -v
```

Expected output: all tests pass in < 5 seconds (all network calls are mocked).

---

## Code Quality

All code in this repository passes the following checks with **0 errors**:

```bash
# Linting (PEP 8 + style)
ruff check src/ ml/ pipeline/ Data-Pipeline/ tests/
# Expected: no output (0 errors)

# Auto-formatting check
black --check src/ ml/ pipeline/ Data-Pipeline/ tests/
# Expected: All done! ✨ 🍰 ✨  0 files would be reformatted.

# Type checking
mypy src/ ml/ --ignore-missing-imports
# Expected: Success: no issues found
```

These checks are enforced in CI/CD (`.github/workflows/ci.yml`).

---

## DVC Remote Setup (GCS)

```bash
# Add GCS as the default DVC remote
dvc remote add -d gcs_remote gs://f1optimizer-data-lake
dvc remote modify gcs_remote credentialpath ~/.config/gcloud/application_default_credentials.json

# Push processed data to GCS
dvc push

# Pull data on another machine
dvc pull
```

The DVC pipeline is defined in `dvc.yaml` at the repo root.
Stages: `ingest_jolpica → ingest_fastf1 → preprocess → validate → detect_anomalies → build_features → bias_analysis`

---

## Pipeline Outputs

| File | Description |
|---|---|
| `data/raw/jolpica/` | Raw JSON from Jolpica API |
| `data/raw/fastf1/` | Raw CSVs from FastF1 library |
| `data/processed/*.parquet` | Normalized Parquet files (10 files) |
| `data/features/` | ML-ready state vectors |
| `Data-Pipeline/scripts/expectations/validation_suite.json` | Validation results |
| `Data-Pipeline/logs/anomaly_report.json` | Anomaly detection report |
| `Data-Pipeline/logs/bias_report.json` | Bias analysis report |

---

## Troubleshooting

**`dvc repro` fails at `ingest_jolpica`**
→ Check internet connectivity. Jolpica API rate limit: 500 req/hr.
→ Retry after 1 hour or use `export USE_LOCAL_DATA=true` to skip ingestion.

**`validate_data.py` exits with code 1**
→ Critical schema violation. Check `scripts/expectations/validation_suite.json` for details.

**`anomaly_detection.py` exits with code 2**
→ Critical errors (e.g. missing Driver IDs). Investigate the input Parquet file.

**FastF1 download very slow**
→ FastF1 caches to `data/raw/fastf1_cache/`. First run is slow; subsequent runs are instant.

**`ImportError: fastf1`**
→ Install with `pip install fastf1`. FastF1 ingestion is optional; pipeline degrades gracefully.

**Airflow `ModuleNotFoundError`**
→ Ensure the repo root is on `PYTHONPATH`: `export PYTHONPATH=/path/to/test:$PYTHONPATH`

---

## See Also

- [`docs/bias.md`](../docs/bias.md) — Bias analysis findings and mitigation
- [`docs/architecture.md`](../docs/architecture.md) — Full system architecture
- [`team-docs/DEV_SETUP.md`](../team-docs/DEV_SETUP.md) — Developer onboarding (team-internal)
- [`ml/README.md`](../ml/README.md) — ML models and training documentation
