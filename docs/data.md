# Data Sources, Splits, and Management

**Last Updated**: 2026-02-14

## Overview

The F1 Strategy Optimizer leverages 74 years of comprehensive Formula 1 data spanning 1950-2024, encompassing race results, pit stops, qualifying sessions, and high-frequency telemetry. This document details data sources, acquisition strategy, preprocessing, splits, and management.

## Data Card

| Attribute | Details |
|-----------|---------|
| **Time Period** | 1950-2024 (74 years) |
| **Total Races** | 1,300+ races |
| **Total Lap Records** | 20M+ laps |
| **Training Data Size** | ~150GB (BigQuery) |
| **Primary Formats** | JSON (race results), CSV (telemetry), Time-series (streaming) |
| **Telemetry Frequency** | 10 Hz (FastF1 library, 2018+) |
| **Geographic Coverage** | Global (23 circuits annually) |
| **Drivers Covered** | 800+ drivers across history, ~20 active per season |

## Data Types

### Continuous Variables
- Throttle percentage (0-100%)
- Speed (km/h)
- Fuel load (kg)
- Brake pressure (bar)
- Tire temperature (°C)
- G-forces (lateral, longitudinal)
- Steering angle (degrees)

### Categorical Variables
- Tire compound (Soft/Medium/Hard/Intermediate/Wet)
- Weather conditions (Dry/Wet/Mixed)
- Track surface (Asphalt type)
- DRS status (Open/Closed)
- Safety car status (None/VSC/Full)

### Temporal Variables
- Lap times (ms)
- Sector times (ms)
- Stint duration (laps)
- Race position by lap

### Geospatial Variables
- GPS coordinates (latitude, longitude)
- Track position (distance from start line)
- Corner identification

## Primary Data Sources

### 1. Ergast F1 API (1950-2024)

**Source**: http://ergast.com/mdb/

**Coverage**:
- 1,300+ races
- Race results (positions, points, retirements)
- Pit stop logs (lap, duration, compound)
- Qualifying sessions
- Driver/constructor standings
- Circuit characteristics

**Format**: JSON/XML REST API

**Update Frequency**: Automated post-race (typically within 24 hours)

**Reliability**: Comprehensive historical archive, community-maintained, high accuracy

**Rate Limits**: None specified, but recommend 1 req/sec for bulk downloads

**Sample Endpoint**:
```
GET http://ergast.com/api/f1/2024.json
GET http://ergast.com/api/f1/2024/1/results.json
GET http://ergast.com/api/f1/2024/1/pitstops.json
```

**Fields Used**:
- `raceId`, `year`, `round`, `circuitId`, `date`
- `driverId`, `constructorId`, `grid`, `position`, `points`
- `lap`, `stop`, `duration`, `time`

### 2. FastF1 Library (2018-2024)

**Source**: https://github.com/theOehrly/FastF1

**Coverage**:
- 200+ races with high-frequency telemetry
- 10 Hz telemetry streams (10 data points per second)
- Car telemetry (throttle, brake, speed, gear, DRS)
- Position data (GPS coordinates, track position)
- Timing data (lap times, sector times, speed traps)

**Format**: Python library interface (pandas DataFrames)

**Data Resolution**: 10 Hz (100ms intervals)

**Telemetry Variables**:
- `Throttle`: Percentage (0-100)
- `Brake`: Boolean or pressure
- `Speed`: km/h
- `nGear`: Gear number (1-8)
- `DRS`: DRS status (0-14, with specific codes)
- `RPM`: Engine RPM
- `X`, `Y`, `Z`: GPS coordinates

**Sample Usage**:
```python
import fastf1

session = fastf1.get_session(2024, 'Monaco', 'R')
session.load()
laps = session.laps
telemetry = laps.pick_driver('VER').pick_fastest().get_telemetry()
```

**Limitations**:
- Telemetry only available from 2018 onwards
- Some sessions may have incomplete data (sensor failures)
- Requires internet connection for initial download

### 3. Supplementary Data Sources

**Track Characteristics**:
- Circuit length, number of corners, elevation changes
- Historical lap records
- Typical pit stop loss time
- Source: Manual curation + Ergast API

**Weather Data**:
- Temperature, humidity, precipitation
- Wind speed and direction
- Track temperature
- Source: OpenWeatherMap API (historical) + race reports

**Driver Statistics**:
- Career wins, podiums, championships
- Average finishing position
- DNF rate
- Source: Ergast API aggregation

**Vehicle Setup Specifications**:
- Wing angles (front/rear)
- Suspension settings
- Brake balance presets
- Source: Team technical reports (limited availability)

## Data Rights and Privacy

### Ownership & Licensing

**Ergast API**:
- Public domain data aggregation
- Provided under open-source model
- No commercial restrictions for educational use
- Attribution required: "Data provided by Ergast API"

**FastF1**:
- Community-contributed telemetry extraction
- Non-commercial educational use permitted
- MIT License
- Attribution required: "Telemetry via FastF1 library"

**Commercial Use**:
- Competitive deployment requires licensing agreements with FIA (Fédération Internationale de l'Automobile)
- Educational/research use authorized under fair use policies
- Team-specific deployment requires team consent

### Privacy Considerations

- **No PII**: Data represents aggregated race performance and telemetry
- **Public Figures**: Driver data is professional performance (not personal)
- **GDPR-Exempt**: No personal data processing beyond public professional records
- **Compliance**: Follows fair use for educational/research purposes

### Usage Rights for This Project

- Academic/educational use authorized
- Telemetry sourced from publicly available race broadcasts
- All attributions included in codebase and documentation
- No redistribution of raw data; only derived models and insights

## Data Loading Strategy

### Stage 1: Historical Data Ingestion (Week 1-2)

**Ergast API Download**:
```bash
# Download all seasons 1950-2024
python data/download.py --source ergast --start-year 1950 --end-year 2024

# Expected output:
# - races.json (1,300+ races)
# - results.json (50,000+ results)
# - pitstops.json (100,000+ pit stops)
# - qualifying.json (20,000+ qualifying sessions)
```

**FastF1 Telemetry Download**:
```bash
# Download telemetry 2018-2024
python data/download.py --source fastf1 --start-year 2018 --end-year 2024

# Expected output:
# - telemetry/ (200+ race sessions)
# - laps/ (lap-by-lap timing data)
# - Total: ~120GB uncompressed
```

**BigQuery Upload**:
```bash
# Load into BigQuery raw tables
python data/upload_to_bq.py --dataset f1_strategy --table-prefix raw_

# Tables created:
# - raw_races
# - raw_results
# - raw_pitstops
# - raw_qualifying
# - raw_telemetry
# - raw_laps
```

**Estimated Total**: ~150GB uncompressed, ~50GB compressed in BigQuery

### Stage 2: Data Cleaning & Preprocessing (Week 2-3)

**Missing Data Handling**:
```python
# Occasional sensor failures in telemetry
# Strategy: Linear interpolation for gaps <1 second
# Longer gaps: Mark as NULL, exclude from training

# Example:
telemetry['Throttle'].interpolate(method='linear', limit=10, inplace=True)
```

**Normalization**:
```python
# Normalize throttle/brake values across different car generations
# Mechanical era (1950-1990) vs Hybrid era (2014+)
# Apply standardization: (x - mean) / std per era
```

**Alignment**:
```python
# Align race results with pit stop logs
# Join on (raceId, driverId)
# Validate: Every result should have 0-5 pit stops
```

**Outlier Removal**:
```python
# Remove outliers:
# - Lap times < 30s or > 500s (data errors)
# - DNF laps (crashes, mechanical failures)
# - Safety car periods (artificially slow laps)

# Strategy: Flag laps with lap_time > mean + 3*std
# Remove flagged laps from training
```

**Tire Compound Standardization**:
```python
# Historical tire notation changed in 2011
# Pre-2011: Option/Prime → Post-2011: Soft/Medium/Hard
# Create mapping dictionary:
tire_mapping = {
    'Option': 'Soft',
    'Prime': 'Medium',
    # ... historical mappings
}
```

### Stage 3: Feature Engineering (Week 3-4)

**Tire Age Calculation**:
```python
# Calculate laps since pit stop
# For each stint, assign tire_age = lap - pit_lap

def calculate_tire_age(df):
    df = df.sort_values(['raceId', 'driverId', 'lap'])
    df['tire_age'] = df.groupby(['raceId', 'driverId', 'stint']).cumcount() + 1
    return df
```

**Fuel Remaining Estimation**:
```python
# Model fuel consumption based on telemetry
# Starting fuel: ~110kg (max allowed)
# Consumption rate: ~1.6-2.0 kg/lap (depends on circuit)

def estimate_fuel(df):
    df['fuel_remaining'] = 110 - (df['lap'] * df['fuel_rate'])
    return df
```

**Telemetry Features**:
```python
# Extract per-lap statistics from 10Hz telemetry
features = {
    'mean_throttle': df.groupby('lap')['Throttle'].mean(),
    'std_throttle': df.groupby('lap')['Throttle'].std(),
    'max_speed': df.groupby('lap')['Speed'].max(),
    'brake_count': df.groupby('lap')['Brake'].sum(),  # Number of braking events
    'drs_usage': df.groupby('lap')['DRS'].mean()  # % of lap with DRS active
}
```

**Race Context Features**:
```python
# Position, gap to leader, delta to nearest competitor
df['gap_to_leader'] = df['cumulative_time'] - df.groupby('lap')['cumulative_time'].transform('min')
df['delta_to_next'] = df['cumulative_time'] - df.groupby(['lap', 'position'])['cumulative_time'].shift(1)
```

**Temporal Features**:
```python
# Lap number, stint age, remaining laps
df['lap_number'] = df['lap']
df['stint_age'] = df.groupby(['raceId', 'driverId', 'stint']).cumcount() + 1
df['laps_remaining'] = df['total_laps'] - df['lap']
```

## Data Splitting Strategy

### Temporal Split (Prevents Data Leakage)

| Split | Size | Time Period | Races | Purpose |
|-------|------|-------------|-------|---------|
| **Train** | ~140GB | 1950-2022 | 1,300+ | Extract driver profiles, train baseline models |
| **Validation** | ~5GB | 2023 Q1-Q2 | 10 races | Hyperparameter tuning, model selection |
| **Test** | ~10GB | 2023 Q3-Q4 + 2024 | 20+ races | Final evaluation, ground-truth validation |

### Justification for Temporal Split

1. **Prevents Data Leakage**: No future information used for past predictions
2. **Reflects Real Deployment**: New races arrive sequentially
3. **Tests Generalization**: Validates across regulation changes and eras
4. **Business Alignment**: Models must work on unseen future races

### Split Implementation

```python
# Define split boundaries
TRAIN_END = '2022-12-31'
VAL_START = '2023-01-01'
VAL_END = '2023-06-30'
TEST_START = '2023-07-01'

# Partition BigQuery tables
CREATE TABLE f1_strategy.train AS
SELECT * FROM f1_strategy.processed_data
WHERE race_date <= '2022-12-31';

CREATE TABLE f1_strategy.validation AS
SELECT * FROM f1_strategy.processed_data
WHERE race_date BETWEEN '2023-01-01' AND '2023-06-30';

CREATE TABLE f1_strategy.test AS
SELECT * FROM f1_strategy.processed_data
WHERE race_date >= '2023-07-01';
```

### Validation Strategy

**Cross-Validation**: Not applicable (time-series data requires sequential split)

**Holdout Validation**: Use 2023 Q1-Q2 for hyperparameter tuning

**Final Test**: Evaluate on completely unseen 2023 Q3-Q4 + 2024 races

## Data Management

### Storage Architecture

**BigQuery Partitioning**:
```sql
-- Partition by race_year and race_date for query efficiency
CREATE TABLE f1_strategy.raw_races (
    race_id INT64,
    year INT64,
    round INT64,
    circuit_id STRING,
    race_date DATE,
    race_time TIME,
    ...
)
PARTITION BY race_date
CLUSTER BY year, circuit_id;
```

**Benefits**:
- Query pruning: Only scan relevant partitions
- Cost reduction: Pay only for scanned data
- Performance: 10-100x faster queries on date ranges

### Schema Documentation

**Each table includes**:
- Field definitions with data types
- Units (e.g., kg, km/h, ms)
- Transformation logic applied
- Nullability and default values
- Validation rules

**Example**:
```yaml
# raw_telemetry schema
fields:
  - name: race_id
    type: INT64
    description: Unique identifier for race
    nullable: false

  - name: driver_id
    type: STRING
    description: Three-letter driver code (e.g., VER, HAM)
    nullable: false

  - name: lap
    type: INT64
    description: Lap number (1 to total_laps)
    nullable: false

  - name: throttle
    type: FLOAT64
    description: Throttle percentage (0-100)
    unit: percentage
    nullable: true
    validation: 0 <= value <= 100

  - name: speed
    type: FLOAT64
    description: Speed in kilometers per hour
    unit: km/h
    nullable: true
    validation: 0 <= value <= 400
```

### Versioning

**Dataset Versions**: Tagged by week to track preprocessing changes

```bash
# Version naming convention
dataset_v1_week1   # Initial raw data
dataset_v2_week3   # After cleaning
dataset_v3_week5   # After feature engineering
dataset_v4_week7   # Final training set
```

**Schema Versioning**: Git-tracked schema definitions

```bash
git tag -a data-schema-v1.0 -m "Initial schema definition"
git tag -a data-schema-v1.1 -m "Added tire_age feature"
```

### Quality Checks

**Automated Validation** (runs daily):

```python
# Lap time validation
assert (df['lap_time'] > 30000).all(), "Lap times must be > 30s"
assert (df['lap_time'] < 300000).all(), "Lap times must be < 300s"

# Fuel validation
assert (df['fuel_remaining'] >= 0).all(), "Fuel cannot be negative"
assert (df['fuel_remaining'] <= 150).all(), "Fuel cannot exceed 150kg"

# Position validation
assert (df['position'] >= 1).all(), "Position must be >= 1"
assert (df['position'] <= 24).all(), "Position must be <= 24 (max grid)"

# Completeness check
completeness = df.notnull().sum() / len(df)
assert (completeness > 0.95).all(), "Feature completeness must be >95%"
```

**Quality Metrics Dashboard**:
- Completeness percentage per table
- Null value counts per field
- Outlier detection alerts
- Schema drift detection

### Data Refresh Strategy

**Post-Race Updates** (automated):
```bash
# After each race (23 races per season)
# 1. Download new race data from Ergast API
# 2. Download telemetry from FastF1
# 3. Append to BigQuery raw tables
# 4. Run preprocessing pipeline
# 5. Update feature store
# 6. Trigger model retraining (weekly)

# Cron job: Every Sunday 22:00 UTC (after typical race end)
0 22 * * 0 /path/to/data/refresh.sh
```

**Incremental Updates**:
- Only process new races (not full rebuild)
- Maintain data versioning for rollback
- Validate consistency with historical data

### Backup and Recovery

**BigQuery Snapshots**:
```bash
# Daily snapshots for 7 days
# Weekly snapshots for 4 weeks
# Monthly snapshots for 12 months

bq cp f1_strategy.processed_data f1_strategy.processed_data_$(date +%Y%m%d)
```

**Recovery Procedure**:
1. Identify corrupted table or data issue
2. Restore from most recent snapshot
3. Re-run preprocessing from last known good state
4. Validate data quality before resuming production

## Data Access Patterns

### Training
- Batch reads from BigQuery
- Full table scans with partition pruning
- Export to pandas/TensorFlow Dataset for model training

### Inference
- Real-time feature extraction from streaming telemetry
- Cached driver profiles (static)
- Cached circuit characteristics (static)
- Live race context from Dataflow pipeline

### Dashboard
- Aggregated queries (e.g., top drivers by aggression score)
- Historical race lookups
- Cached results for common queries

## Data Security

**Access Control**:
- Service accounts with least-privilege IAM roles
- BigQuery dataset-level permissions
- Audit logging enabled

**Encryption**:
- Data encrypted at rest (default BigQuery)
- Data encrypted in transit (HTTPS, TLS)

**Compliance**:
- No PII stored
- GDPR-exempt (professional sports data)
- Internal use only (educational project)

## Known Data Issues

### Missing Telemetry
- Pre-2018 races: No high-frequency telemetry available
- Solution: Use lap times and pit stop logs only for historical races

### Sensor Failures
- Occasional telemetry gaps (1-2% of laps)
- Solution: Linear interpolation for gaps <1s, mark as NULL otherwise

### Regulation Changes
- Tire compounds, fuel limits, car specifications change over time
- Solution: Temporal features (year, era) to capture context

### Circuit Layout Changes
- Some circuits modified over decades
- Solution: Track circuit_version in metadata

### Weather Data Granularity
- Historical weather data less precise than modern
- Solution: Categorical binning (Dry/Wet/Mixed) for older races

## Training Pipeline Data Flow

### Overview

ML training runs as an isolated pipeline stage with containerized, distributed execution. Training jobs are orchestrated as scheduled, retryable stages that read from validated storage and write artifacts to dedicated buckets.

### Data Sources for Training

**Input Storage** (READ-ONLY):
```
BigQuery Tables (Processed/Validated Data):
├── f1_strategy.f1_features          # Primary feature store
├── f1_strategy.driver_profiles      # Driver behavioral profiles
├── f1_strategy.train                # Training split (1950-2022)
├── f1_strategy.validation           # Validation split (2023 Q1-Q2)
└── f1_strategy.test                 # Test split (2023 Q3-2024)
```

**Prohibited Sources**:
- ❌ `raw_races`, `raw_results`, `raw_telemetry` (training never reads raw data)
- ❌ `processed_data` (must use versioned train/val/test splits)
- ❌ Any streaming Pub/Sub topics (training is batch-only)

### Data Output Destinations

**Artifact Storage** (WRITE-ONLY):
```
GCS Buckets:
├── gs://f1-strategy-artifacts/
│   ├── models/
│   │   ├── tire_degradation/
│   │   │   ├── v1.0/model.pkl
│   │   │   ├── v1.0/metadata.json
│   │   │   └── v1.0/metrics.json
│   │   ├── fuel_consumption/
│   │   ├── brake_bias/
│   │   └── driving_style/
│   │
│   ├── checkpoints/
│   │   └── [job_id]/epoch_*.ckpt
│   │
│   ├── metrics/
│   │   └── [job_id]/
│   │       ├── training_metrics.csv
│   │       ├── validation_metrics.csv
│   │       └── feature_importance.json
│   │
│   └── logs/
│       └── [job_id]/
│           ├── stdout.log
│           └── stderr.log
```

**Prohibited Destinations**:
- ❌ Training jobs NEVER write to BigQuery `raw_*` tables
- ❌ Training jobs NEVER write to `processed_data` tables
- ❌ Training jobs NEVER modify feature store directly

### Training Job Architecture

**Containerized Execution**:
```yaml
Training Job:
  - Base Image: gcr.io/f1-strategy/training-worker:latest
  - Entrypoint: python pipeline/training/worker.py
  - Resources: Autoscaling worker pool (CPU/GPU abstracted)
  - Networking: Secure internal VPC, no public internet access
  - IAM: Training service account (read: BigQuery, write: GCS artifacts)
```

**Distributed Training**:
- **Horizontal Scaling**: 1-N workers per training job
- **Communication**: Workers communicate via HTTPS (encrypted transport)
- **Coordination**: Parameter server or ring-allreduce (framework-dependent)
- **Fault Tolerance**: Worker failure does not fail job (retry logic)

### Orchestration

**Scheduler**: Vertex AI Pipelines
```python
# Example pipeline definition (Kubeflow DSL)
from kfp.v2 import dsl

@dsl.pipeline(name='f1-training-pipeline')
def training_pipeline(
    data_split: str = 'train',
    model_type: str = 'tire_degradation',
    num_workers: int = 4
):
    # Stage 1: Data validation
    validate_data = dsl.ContainerOp(
        name='validate-data',
        image='gcr.io/f1-strategy/data-validator:latest',
        arguments=['--split', data_split]
    )

    # Stage 2: Distributed training
    train_model = dsl.ContainerOp(
        name='train-model',
        image='gcr.io/f1-strategy/training-worker:latest',
        arguments=[
            '--model', model_type,
            '--data-split', data_split,
            '--num-workers', num_workers
        ]
    ).after(validate_data)

    # Stage 3: Model evaluation
    evaluate_model = dsl.ContainerOp(
        name='evaluate-model',
        image='gcr.io/f1-strategy/evaluator:latest',
        arguments=['--model-uri', train_model.outputs['model_uri']]
    ).after(train_model)
```

**Job Management**:
- **Retries**: Up to 3 retries on failure (configurable)
- **Timeout**: 12 hours max per training job
- **Failure Detection**: Worker heartbeat monitoring (60s interval)
- **Job Restart**: Failed workers automatically restarted
- **Non-Blocking**: Training stages do not block streaming or inference pipelines

**Scheduling**:
```yaml
Training Schedule:
  - Weekly Retraining: Every Monday 02:00 UTC
  - Post-Race Training: Triggered after race data validation (optional)
  - Manual Trigger: Via Vertex AI Pipelines API
  - Concurrent Jobs: Max 4 training jobs in parallel
```

### Infrastructure Provisioning

**Terraform-Managed Resources**:
```hcl
# infrastructure/terraform/training.tf
resource "google_compute_instance_template" "training_worker" {
  name         = "f1-training-worker-template"
  machine_type = var.worker_machine_type  # Abstracted (CPU/GPU)

  disk {
    source_image = "gcr.io/f1-strategy/training-worker:latest"
  }

  network_interface {
    network = google_compute_network.training_vpc.name
    # No external IP (internal-only)
  }

  service_account {
    email  = google_service_account.training_worker.email
    scopes = ["cloud-platform"]
  }
}

resource "google_compute_autoscaler" "training_workers" {
  name   = "f1-training-autoscaler"
  target = google_compute_instance_group_manager.training_workers.id

  autoscaling_policy {
    min_replicas = 0
    max_replicas = 10

    cpu_utilization {
      target = 0.7
    }
  }
}
```

**Autoscaling**:
- **Scale-to-Zero**: Worker pool scales to 0 when no jobs running
- **Scale-Up**: Workers provisioned on-demand when job submitted
- **Scale-Down**: Workers terminated after 10min idle timeout
- **Resource Types**: CPU-optimized, GPU-accelerated, or memory-optimized (abstracted)

### Security

**IAM Roles** (Least-Privilege):
```yaml
Training Service Account: training-worker@f1-strategy.iam.gserviceaccount.com

Permissions:
  BigQuery:
    - roles/bigquery.dataViewer  # READ f1_strategy.* tables
    - roles/bigquery.jobUser     # Execute queries

  Cloud Storage:
    - roles/storage.objectCreator  # WRITE gs://f1-strategy-artifacts/*
    - roles/storage.objectViewer   # READ training data exports (if needed)

  Vertex AI:
    - roles/aiplatform.user        # Submit training jobs

  Denied:
    - ❌ bigquery.dataEditor       # CANNOT modify BigQuery tables
    - ❌ storage.admin             # CANNOT delete artifacts
    - ❌ compute.admin             # CANNOT modify infrastructure
```

**Network Security**:
- **VPC**: Training workers run in isolated VPC
- **Firewall**: Ingress blocked, egress only to GCP APIs
- **Encryption**: All inter-worker communication uses HTTPS/TLS 1.3
- **No Public Access**: Workers have no external IP addresses

**Data Access**:
- **Read**: Training jobs read from BigQuery via IAM-authenticated connections
- **Write**: Artifacts written to GCS with server-side encryption (AES-256)
- **Audit Logging**: All data access logged to Cloud Audit Logs

### Training Data Flow Diagram

```
┌─────────────────────────────────────────────────────┐
│ BigQuery (Processed & Validated Data)              │
│ ├── f1_features (READ-ONLY)                        │
│ ├── train split (READ-ONLY)                        │
│ └── validation split (READ-ONLY)                   │
└──────────────────┬──────────────────────────────────┘
                   │
                   │ Secure Query (IAM Auth)
                   ▼
┌─────────────────────────────────────────────────────┐
│ Training Pipeline (Vertex AI Pipelines)            │
│                                                     │
│  ┌──────────────┐    ┌──────────────┐             │
│  │  Worker 1    │◄──►│  Worker 2    │             │
│  │ (Container)  │    │ (Container)  │             │
│  └──────────────┘    └──────────────┘             │
│         │                    │                     │
│         │  HTTPS/TLS 1.3    │                     │
│         ▼                    ▼                     │
│  ┌──────────────┐    ┌──────────────┐             │
│  │  Worker 3    │◄──►│  Worker N    │             │
│  │ (Container)  │    │ (Container)  │             │
│  └──────────────┘    └──────────────┘             │
└──────────────────┬──────────────────────────────────┘
                   │
                   │ Write Artifacts (IAM Auth, Encrypted)
                   ▼
┌─────────────────────────────────────────────────────┐
│ GCS Artifact Storage (WRITE-ONLY)                  │
│ ├── models/ (trained model binaries)               │
│ ├── checkpoints/ (intermediate states)             │
│ ├── metrics/ (training/validation metrics)         │
│ └── logs/ (execution logs)                         │
└─────────────────────────────────────────────────────┘
```

### Data Validation Before Training

**Pre-Training Checks** (Automated):
```python
# pipeline/training/validate.py
def validate_training_data(split: str) -> bool:
    """Validate data quality before training starts."""

    # Check 1: Data completeness
    query = f"""
        SELECT COUNT(*) as total_rows,
               COUNTIF(lap_time_ms IS NULL) as null_lap_times,
               COUNTIF(tire_age IS NULL) as null_tire_age
        FROM f1_strategy.{split}
    """
    result = bq_client.query(query).result()
    row = list(result)[0]

    completeness = 1 - (row.null_lap_times + row.null_tire_age) / (row.total_rows * 2)
    assert completeness > 0.95, f"Data completeness {completeness:.2%} < 95%"

    # Check 2: Temporal consistency
    query = f"""
        SELECT MIN(race_date) as min_date, MAX(race_date) as max_date
        FROM f1_strategy.{split}
    """
    result = bq_client.query(query).result()
    row = list(result)[0]

    if split == 'train':
        assert row.min_date >= '1950-01-01', "Train split starts too early"
        assert row.max_date <= '2022-12-31', "Train split leaks into validation"

    # Check 3: Feature schema match
    expected_features = ['lap_time_ms', 'tire_age', 'fuel_remaining', ...]
    query = f"SELECT * FROM f1_strategy.{split} LIMIT 1"
    result = bq_client.query(query).result()
    actual_features = [field.name for field in result.schema]

    missing = set(expected_features) - set(actual_features)
    assert not missing, f"Missing features: {missing}"

    return True
```

**Validation Failure Handling**:
- Training job exits with error code
- Alert sent to Slack/PagerDuty
- Logs written to Cloud Logging
- No model artifacts written

### Cost Management

**Training Budget**:
- **Weekly Training**: ~$20-30 (4 workers × 2 hours × $2.50/hr)
- **Monthly Budget**: ~$80-120
- **Cost Alert**: Trigger if single job exceeds $50

**Cost Optimization**:
- Use preemptible VMs for non-critical training
- Cache preprocessed features to reduce BigQuery query costs
- Autoscale to zero when idle
- Use spot instances (up to 70% cost reduction)

### Monitoring & Observability

**Training Metrics** (Exported to Cloud Monitoring):
```yaml
Metrics:
  - training.job.duration_seconds
  - training.job.failure_count
  - training.worker.cpu_utilization
  - training.worker.memory_usage
  - training.data.records_processed
  - training.model.loss
  - training.model.accuracy
```

**Alerts**:
```yaml
Training Alerts:
  - Job Duration > 12 hours → PagerDuty
  - Job Failure Rate > 20% → Slack
  - Worker OOM → Auto-retry with larger instance
  - Data Validation Failure → Block training, alert owner
```

**Logs**:
- All training logs streamed to Cloud Logging
- Queryable via Log Explorer
- Retention: 30 days

## Future Enhancements

- **Real-Time Telemetry**: Integrate live race feeds (requires FIA partnership)
- **Video Analysis**: Extract corner-by-corner driver behavior from onboards
- **Tire Temperature**: More granular tire deg modeling with temp data
- **Setup Database**: Expand vehicle setup specifications (currently limited)
- **Multi-Series**: Extend to Formula E, IndyCar, WEC (similar data structure)
- **Federated Learning**: Train on team-specific data without centralizing
- **AutoML Integration**: Automated hyperparameter tuning via Vertex AI AutoML

---

**See Also**:
- CLAUDE.md: High-level project overview
- docs/training-pipeline.md: Detailed training infrastructure specifications
- docs/models.md: How data flows into ML models
- docs/architecture.md: System design, deployment
- docs/metrics.md: Data quality KPIs
